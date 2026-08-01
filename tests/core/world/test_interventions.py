"""What the player does to a world, and what it costs (#26).

The contract is checkable in advance and was written against it (§8.1). Two tests carry more than
their weight: the refusal being *recorded with its reason*, because a player who was away needs to
know what did not happen (§2.4); and `TestInterventionsRunBeforeTheTickTheyLandOn`, which is the
whole argument for draining the queue before a tick's systems rather than after.
"""

import pytest

from clients.viewer.demo_world import build_demo_world
from core.selection import Selection
from core.world.cull import Cull
from core.world.interventions import Interventions, Record


class Noted:
    """An intervention that records that it ran, and nothing else.

    Authored rather than assembled, for the reason `ConstantDrive` is in the behaviour tests: the
    framework's contract is cost, precondition and effect, and exercising it against a real cull
    would be testing the cull as well.
    """

    def __init__(self, name="noted", price=1.0, refuse=None):
        self.name = name
        self.price = price
        self.refuse = refuse
        self.applied = 0

    def cost(self):
        return self.price

    def refusal(self):
        return self.refuse

    def apply(self, store):
        self.applied += 1


def world(ticks=40, seed=1, founders=60):
    built = build_demo_world(seed=seed, n_entities=founders)
    built.loop.advance(ticks)
    built.loop.interventions = Interventions(balance=100.0)
    return built


def living_of(built, species_id):
    return Selection.from_mask(
        built.store.alive & (built.store.age >= 0) & (built.store.species_id == species_id)
    )


class TestTheLedgerRefusesAnImpossibleStart:
    def test_a_negative_budget_is_rejected(self):
        with pytest.raises(ValueError, match="cannot start negative"):
            Interventions(balance=-1.0)


class TestNothingHappensUntilABoundary:
    def test_a_request_is_queued_rather_than_applied(self):
        """Applying on request would edit columns halfway through whatever the caller was doing —
        the hazard §2.3 makes capacity growth wait for, and the same answer."""
        queue = Interventions(balance=10.0)
        noted = Noted()

        queue.request(noted)

        assert noted.applied == 0
        assert queue.pending == 1
        assert queue.history == []

    def test_the_next_tick_applies_it(self):
        built = world()
        noted = Noted()
        built.loop.interventions.request(noted)

        built.loop.advance(1)

        assert noted.applied == 1
        assert built.loop.interventions.pending == 0

    def test_requests_apply_in_the_order_they_were_asked_for(self):
        """The player asked for these in a sequence and the second may depend on the first having
        happened; anything cleverer would be a scheduling policy nobody asked for."""
        queue = Interventions(balance=100.0)
        order = []
        for name in ("first", "second", "third"):
            noted = Noted(name=name)
            noted.apply = lambda store, name=name: order.append(name)
            queue.request(noted)

        queue.apply_pending(store=None, tick=0)

        assert order == ["first", "second", "third"]


class TestTheLedgerAndTheHistoryAgree:
    def test_an_applied_intervention_is_charged_and_recorded(self):
        queue = Interventions(balance=10.0)
        queue.request(Noted(price=3.5))

        records = queue.apply_pending(store=None, tick=7)

        assert queue.balance == pytest.approx(6.5)
        assert records == [Record(tick=7, name="noted", cost=3.5)]
        assert queue.history == records

    def test_the_balance_always_reconciles_against_the_history(self):
        """A balance that could move without a matching line is not a ledger, and §3.2 needs this
        to survive a restore rather than merely to look right now."""
        queue = Interventions(balance=20.0)
        for price in (2.0, 3.0, 100.0, 4.0):
            queue.request(Noted(price=price))

        queue.apply_pending(store=None, tick=1)

        assert queue.balance == pytest.approx(20.0 - sum(r.cost for r in queue.history))

    def test_a_refusal_is_recorded_with_its_reason(self):
        """The one a player who was away depends on. §2.4 makes absence the normal case, and an
        intervention that quietly did nothing is the obituary §2.7 exists to avoid."""
        queue = Interventions(balance=10.0)
        queue.request(Noted(refuse="the ground is frozen"))

        records = queue.apply_pending(store=None, tick=3)

        assert records == [Record(tick=3, name="noted", cost=0.0, refusal="the ground is frozen")]
        assert not records[0].applied
        assert queue.balance == pytest.approx(10.0), "a refusal must not be charged"

    def test_an_unaffordable_intervention_is_refused_rather_than_overdrawn(self):
        queue = Interventions(balance=2.0)
        queue.request(Noted(price=5.0))

        records = queue.apply_pending(store=None, tick=0)

        assert "balance" in records[0].refusal
        assert queue.balance == pytest.approx(2.0)

    def test_the_interesting_refusal_is_reported_before_the_affordability_one(self):
        """"That would wipe out the species" is more useful than "you cannot afford it", and the
        second is true of everything once the budget is empty."""
        queue = Interventions(balance=0.0)
        queue.request(Noted(price=99.0, refuse="that would wipe out the species"))

        assert queue.apply_pending(store=None, tick=0)[0].refusal == (
            "that would wipe out the species"
        )

    def test_a_refused_intervention_is_not_retried_later(self):
        """Retrying would fire it at some unpredictable later moment, which is worse than not
        firing: the player would be looking at a world changed by a decision they made about a
        different one."""
        queue = Interventions(balance=10.0)
        queue.request(Noted(refuse="not now"))

        queue.apply_pending(store=None, tick=0)
        queue.apply_pending(store=None, tick=1)

        assert queue.pending == 0
        assert len(queue.history) == 1


class TestCulling:
    def test_it_removes_the_number_asked_for(self):
        built = world()
        species = int(built.store.species_id[built.store.alive][0])
        before = len(living_of(built, species))

        built.loop.interventions.request(
            Cull(built.ecology, built.store, species_id=species, count=10, survivors=5)
        )
        built.loop.advance(1)

        assert built.loop.interventions.history[-1].applied
        assert len(living_of(built, species)) <= before - 10

    def test_it_returns_the_bodies_to_the_ground_they_fell_on(self):
        """A cull is decomposition compressed into one tick, and the *soil* is what proves it —
        `total_nutrients()` would hold either way, because the ledger counts an unreturned body as
        exported (see `Cull`, and the gap filed against the invariant). What a skipped return would
        do is sterilise the ground the player used the cull on."""
        built = world()
        species = int(built.store.species_id[built.store.alive][0])
        cull = Cull(built.ecology, built.store, species_id=species, count=10, survivors=5)
        exported_before = built.plants.exported_nutrients
        total_before = built.plants.total_nutrients()

        # Applied directly rather than through a tick: `plant_growth` runs first in the order and
        # converts soil into standing crop, and grazing adds to the export ledger, so neither
        # column reads cleanly at a tick boundary. The queue's integration with the loop is what
        # the tests above cover; this one is about where the bodies went.
        cull.apply(built.store)

        assert built.plants.exported_nutrients < exported_before, "the bodies went nowhere"
        assert built.plants.total_nutrients() == pytest.approx(total_before, rel=1e-9)

    def test_it_refuses_to_take_a_species_below_its_survivor_floor(self):
        """§2.7's rule, and the reason this is the framework's first intervention rather than an
        arbitrary one: the precondition is a real design constraint."""
        built = world()
        species = int(built.store.species_id[built.store.alive][0])
        alive = len(living_of(built, species))

        built.loop.interventions.request(
            Cull(built.ecology, built.store, species_id=species, count=alive, survivors=10)
        )
        built.loop.advance(1)

        assert "§2.7" in built.loop.interventions.history[-1].refusal
        assert len(living_of(built, species)) > 0

    def test_a_species_with_no_living_members_is_refused(self):
        built = world()

        built.loop.interventions.request(
            Cull(built.ecology, built.store, species_id=999, count=1, survivors=1)
        )
        built.loop.advance(1)

        assert "no living members" in built.loop.interventions.history[-1].refusal

    def test_gestating_rows_do_not_count_as_survivors(self):
        """One carries a negative age and has not been born (#20), so counting it would let a cull
        leave a species whose only remaining members are unborn."""
        built = world(ticks=120)
        species = int(built.store.species_id[built.store.alive][0])
        gestating = int(
            (
                built.store.alive
                & (built.store.age < 0)
                & (built.store.species_id == species)
            ).sum()
        )
        assert gestating > 0, "the demo world should be breeding by tick 120"

        cull = Cull(built.ecology, built.store, species_id=species, count=1, survivors=1)

        assert len(cull._living()) == int(
            (
                built.store.alive
                & (built.store.age >= 0)
                & (built.store.species_id == species)
            ).sum()
        )

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"count": 0, "survivors": 1}, "at least one animal"),
            ({"count": 1, "survivors": 0}, "eradication"),
        ],
    )
    def test_a_cull_that_could_empty_a_species_is_rejected_at_construction(
        self, kwargs, message
    ):
        """A survivor floor of zero honours §2.7's letter and breaks it: a population of one is
        extinct on a delay."""
        built = world(ticks=1)
        with pytest.raises(ValueError, match=message):
            Cull(built.ecology, built.store, species_id=0, **kwargs)


class TestInterventionsRunBeforeTheTickTheyLandOn:
    def test_an_intervention_that_broke_conservation_is_caught_on_its_own_tick(self):
        """The whole argument for the ordering. An intervention applied before the systems is in
        front of the same tick's invariant pass (§6), so a conservation law it breaks is reported
        on the tick it was broken rather than surfacing later somewhere that did nothing wrong.
        """
        built = build_demo_world(seed=1, n_entities=40)
        built.loop.advance(20)
        built.loop.debug_checks = True
        built.loop.interventions = Interventions(balance=10.0)

        # Biomass out of nowhere: the naive "emergency feed", and the shape of intervention that
        # genuinely does break the loop. Zeroing an animal's energy would *not* — the ledger counts
        # it as exported either way, which is its own gap and filed separately.
        thief = Noted(name="magic feed")
        thief.apply = lambda store: built.plants.biomass.__iadd__(100.0)
        built.loop.interventions.request(thief)

        with pytest.raises(Exception, match="nutrients_are_conserved"):
            built.loop.advance(1)

    def test_a_cull_lands_before_the_systems_that_read_the_population(self):
        """Applied after them, a cull would sit inert until the following tick — the world on
        screen would be one the player's click had not reached yet. The intervention sees the
        population it was asked about; the tick then runs on what it left behind."""
        built = world()
        species = int(built.store.species_id[built.store.alive][0])
        before = len(living_of(built, species))
        seen = []

        watcher = Noted(name="watcher")
        watcher.apply = lambda store: seen.append(len(living_of(built, species)))
        built.loop.interventions.request(watcher)
        built.loop.interventions.request(
            Cull(built.ecology, built.store, species_id=species, count=10, survivors=5)
        )
        built.loop.advance(1)

        assert seen == [before], "the intervention ran against a tick that had not started"
        assert len(living_of(built, species)) <= before - 10
