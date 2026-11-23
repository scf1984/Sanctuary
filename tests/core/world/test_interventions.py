"""What the player does to a world, and what it costs (#26).

The contract is checkable in advance and was written against it (§8.1). Two tests carry more than
their weight: the refusal being *recorded with its reason*, because a player who was away needs to
know what did not happen (§2.4); and `TestInterventionsRunBeforeTheTickTheyLandOn`, which is the
whole argument for draining the queue before a tick's systems rather than after.
"""

import numpy as np
import pytest

from worlds.demo import build_demo_world
from core.selection import Selection
from core.world.cull import Cull, CullConfig
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


def cull(built, species_id, count, seed=0):
    """A cull held to the world's own guard, drawing from a generator of its own so a test can
    repeat a draw (the world's generator is what the session passes)."""
    return Cull(
        built.ecology,
        built.carrion,
        built.store,
        built.config.cull,
        species_id=species_id,
        count=count,
        rng=np.random.default_rng(seed),
    )


class TestCulling:
    def test_it_removes_the_number_asked_for(self):
        built = world()
        species = int(built.store.species_id[built.store.alive][0])
        before = len(living_of(built, species))

        built.loop.interventions.request(cull(built, species, count=10))
        built.loop.advance(1)

        assert built.loop.interventions.history[-1].applied
        assert len(living_of(built, species)) <= before - 10

    def test_the_bodies_become_carrion_where_each_animal_stood(self):
        """#28, decision 4. Each body is laid whole, at its own cell, and nothing is returned to
        the soil yet: the flesh stays owed on the export ledger while it lies there, and
        `Carrion.decompose` is what pays it back (#214's invariant, below)."""
        built = world()
        species = int(built.store.species_id[built.store.alive][0])
        culling = cull(built, species, count=10)
        victims = culling._victims()
        culling.rng = np.random.default_rng(0)  # the same draw again, for `apply`
        cells = built.terrain.cell_indices(built.store.x[victims], built.store.y[victims])
        expected = np.zeros_like(built.carrion.mass)
        np.add.at(expected, cells, built.store.energy[victims].astype(np.float64))
        carrion_before = built.carrion.mass.copy()
        exported_before = built.plants.exported_nutrients
        total_before = built.plants.total_nutrients()

        # Applied directly rather than through a tick, so nothing grazes or rots in between.
        culling.apply(built.store)

        np.testing.assert_allclose(built.carrion.mass - carrion_before, expected, rtol=1e-12)
        assert built.plants.exported_nutrients == exported_before
        assert built.plants.total_nutrients() == total_before
        assert not built.store.alive[victims].any()

    def test_victims_are_drawn_at_random_from_the_species(self):
        """#28, decision 3. The first build took the lowest rows, a selection rule nobody chose.
        Over many draws every born member is taken about equally often, the lowest rows included."""
        built = world()
        species = int(built.store.species_id[built.store.alive][0])
        members = living_of(built, species).to_indices()
        count, draws = 5, 2_000
        culling = cull(built, species, count=count, seed=7)

        taken = np.concatenate([culling._victims() for _ in range(draws)])

        assert np.isin(taken, members).all()
        frequency = np.bincount(taken, minlength=built.store.capacity)[members] / draws
        expected = count / len(members)
        # Five binomial standard errors per member: fails on a biased draw, not on noise.
        tolerance = 5 * np.sqrt(expected * (1 - expected) / draws)
        assert np.abs(frequency - expected).max() < tolerance

    def test_a_cull_may_take_a_species_down_to_its_guard_and_no_further(self):
        """§2.7's rule, and #28's done-when: the guard rejects an over-large cull. The floor is the
        world's (`CullConfig`), so no request can lower it."""
        built = world()
        species = int(built.store.species_id[built.store.alive][0])
        alive = len(living_of(built, species))
        floor = built.config.cull.survivors
        assert alive > floor + 1, "the fixture needs a species above its guard"

        greedy = cull(built, species, count=alive - floor + 1)
        exact = cull(built, species, count=alive - floor)

        assert "§2.7" in greedy.refusal()
        assert exact.refusal() is None

        built.loop.interventions.request(greedy)
        built.loop.advance(1)
        assert "§2.7" in built.loop.interventions.history[-1].refusal

    def test_the_demo_world_keeps_six(self):
        """The measured number (#28, decision 1): six born animals regrow to 200 within ten
        generations in 20 of 20 runs (docs/spikes/viable-population.md, after #251)."""
        assert build_demo_world(seed=1, n_entities=10).config.cull.survivors == 6

    def test_a_species_with_no_living_members_is_refused(self):
        built = world()

        built.loop.interventions.request(cull(built, species_id=999, count=1))
        built.loop.advance(1)

        assert "no living members" in built.loop.interventions.history[-1].refusal

    def test_gestating_rows_are_neither_survivors_nor_victims(self):
        """One carries a negative age and has not been born (#20), so counting it would let a cull
        leave a species whose only remaining members are unborn; and it is born whether or not its
        mother survives (§2.7), so a cull never reaches it."""
        built = world(ticks=120)
        species = int(built.store.species_id[built.store.alive][0])
        unborn = built.store.alive & (built.store.age < 0) & (built.store.species_id == species)
        assert unborn.any(), "the demo world should be breeding by tick 120"
        born = len(living_of(built, species))

        culling = cull(built, species, count=born - built.config.cull.survivors)

        assert len(culling._living()) == born
        assert not unborn[culling._victims()].any()

    def test_a_cull_of_nobody_is_rejected_at_construction(self):
        built = world(ticks=1)
        with pytest.raises(ValueError, match="at least one animal"):
            cull(built, species_id=0, count=0)

    @pytest.mark.parametrize("survivors", [0, 1])
    def test_a_guard_below_two_is_rejected(self, survivors):
        """Reproduction needs two parents, so a guard leaving one honours §2.7's letter and breaks
        it: a population of one is extinct on a delay."""
        with pytest.raises(ValueError, match="eradicated"):
            CullConfig(survivors=survivors)


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
        # breaks the field's total. Removing an animal without returning its body is the other
        # shape, and the next test.
        thief = Noted(name="magic feed")
        thief.apply = lambda store: built.plants.biomass.__iadd__(100.0)
        built.loop.interventions.request(thief)

        with pytest.raises(Exception, match="nutrients_are_conserved"):
            built.loop.advance(1)

    def test_an_intervention_that_removes_animals_without_their_bodies_is_caught(self):
        """#214's reproduction. Ten animals released with their energy unreturned leave
        `total_nutrients()` exactly where it was — the ledger still owes it — so only the check
        that asks *where* the owed energy is can see that nothing holds it any more."""
        built = build_demo_world(seed=1, n_entities=60)
        built.loop.advance(60)
        built.loop.debug_checks = True
        built.loop.interventions = Interventions(balance=10.0)

        vanish = Noted(name="vanish")
        vanish.apply = lambda store: store.release(store.row_ids()[store.alive][:10])
        built.loop.interventions.request(vanish)

        with pytest.raises(Exception, match="exported_nutrients_are_held"):
            built.loop.advance(1)

    def test_a_cull_leaves_what_it_removes(self):
        """The same removal done properly: `Cull.apply` lays every body as carrion before freeing
        its row, so the export ledger stays fully held, and decomposing it keeps it so."""
        built = build_demo_world(seed=1, n_entities=60)
        built.loop.advance(60)
        built.loop.debug_checks = True
        built.loop.interventions = Interventions(balance=100.0)
        species = int(built.store.species_id[built.store.alive][0])

        built.loop.interventions.request(cull(built, species, count=10))
        built.loop.advance(5)  # must not raise

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
        built.loop.interventions.request(cull(built, species, count=10))
        built.loop.advance(1)

        assert seen == [before], "the intervention ran against a tick that had not started"
        assert len(living_of(built, species)) <= before - 10
