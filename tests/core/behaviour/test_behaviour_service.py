import numpy as np
import pytest

from core.behaviour.service import Behaviour, BehaviourConfig, DriveRegistrationError
from core.entities.store import EntityStore
from core.genetics.expression import GeneticsConfig
from core.genetics.service import Genetics
from core.genetics.species import SpeciesRegistry
from core.selection import Selection
from core.services import ColumnOwnershipError, ColumnRegistry
from core.world.terrain import Terrain

from tests.support.genes import gene_registry

GENE_NAMES = ("mutability", "choice_temperature", "commitment")
GENE_REGISTRY = gene_registry(GENE_NAMES)
GENETICS_CONFIG = GeneticsConfig(mutability_gene="mutability", drift_margin=2.0)

N_CANDIDATES = 8
LOOK_AHEAD = 2.0
SPACING = 2.0 * np.pi / N_CANDIDATES

# The null option is the last column, by construction in `candidate_positions`.
NULL = N_CANDIDATES

# exp(-4) is about 0.018, so a utility gap of 1 becomes 55 in scaled units and swamps the Gumbel
# noise the softmax adds: this animal takes its best option every time. exp(+4) is about 55, which
# flattens the same gap to 0.018 and leaves the draw all but uniform. Temperature is a gene exactly
# so one world can hold both (#114).
DECISIVE = -4.0
RECKLESS = 4.0


class World:
    """A flat world with the two services `Behaviour` reads: genetics, for the temperature gene,
    and terrain, for the bounds candidate positions are clipped into.
    """

    def __init__(self, capacity=8, n_drives=4, grid=9, n_candidates=N_CANDIDATES):
        self.store = EntityStore(
            initial_capacity=capacity, n_drives=n_drives, n_genes=len(GENE_NAMES)
        )
        self.columns = ColumnRegistry()
        self.genes = GENE_REGISTRY
        self.species = SpeciesRegistry(self.genes.vocabulary)
        self.species_id = self.species.register(GENE_NAMES)
        self.genetics = Genetics(
            self.store, self.columns, self.species, self.genes, GENETICS_CONFIG
        )
        self.terrain = Terrain(np.zeros((grid, grid), dtype=np.float32), cell_size=1.0)
        self.behaviour = Behaviour(
            self.store,
            self.columns,
            self.genetics,
            self.genes,
            self.terrain,
            BehaviourConfig(
                n_candidates=n_candidates,
                look_ahead=LOOK_AHEAD,
                commitment_gene="commitment",
                choice_temperature_gene="choice_temperature",
            ),
        )

    def spawn(self, n, temperature=DECISIVE, commitment=0.0, **columns):
        """Allocate `n` entities at the world centre unless placed, at `temperature`."""
        columns.setdefault("species_id", np.full(n, self.species_id, dtype=np.int32))
        columns.setdefault("x", np.full(n, 4.0, dtype=np.float32))
        columns.setdefault("y", np.full(n, 4.0, dtype=np.float32))
        ids = self.store.allocate(n, **columns)
        rows = [self.store._id_to_row[i] for i in ids.tolist()]
        selection = Selection.from_indices(np.array(rows, dtype=np.int64), self.store.capacity)
        genes = np.zeros((n, len(GENE_NAMES)), dtype=np.float32)
        genes[:, GENE_NAMES.index("choice_temperature")] = temperature
        genes[:, GENE_NAMES.index("commitment")] = commitment
        self.genetics.set_genes(selection, genes)
        return selection


class ConstantDrive:
    """A drive whose urgency and appeal are authored by the test rather than read from a world.

    Every drive the simulation ships reads fields — energy, climate, the plant field — that the
    option contest knows nothing about, so exercising the contest itself means supplying both halves
    directly. `urgency` accepts a scalar or a whole column; `appeal` a scalar or one row of option
    values shared by every entity.
    """

    def __init__(self, name, urgency=0.0, appeal=1.0):
        self.name = name
        self._urgency = np.asarray(urgency, dtype=np.float32)
        self._appeal = np.asarray(appeal, dtype=np.float32)

    def urgency(self, selection):
        if self._urgency.ndim == 0:
            return np.full(len(selection), float(self._urgency), dtype=np.float32)
        return self._urgency[selection.to_mask()]

    def appeal(self, selection, x, y):
        return np.broadcast_to(self._appeal, x.shape).astype(np.float32)


def wanting(option, urgency=1.0):
    """A drive that wants exactly one option and is indifferent to the rest."""
    appeal = np.zeros(N_CANDIDATES + 1, dtype=np.float32)
    appeal[option] = 1.0
    return ConstantDrive(f"wants_{option}", urgency=urgency, appeal=appeal)


def uncommitted(selection):
    """A commitment column of zeros — the degenerate case, where every tick decides afresh."""
    return np.zeros(len(selection), dtype=np.float64)


def per_row(store, **row_values):
    """A (capacity,) float32 column with `row_values` placed at the named rows."""
    column = np.zeros(store.capacity, dtype=np.float32)
    for row, value in row_values.items():
        column[int(row)] = value
    return column


class TestColumnOwnership:
    def test_claims_every_column_it_writes(self):
        world = World()
        for column in ("drive_scores", "choice_heading", "choice_moving"):
            assert world.columns.owner_of(column) == "Behaviour"

    def test_a_rival_service_cannot_also_claim_them(self):
        world = World()

        class RivalBehaviour(Behaviour):
            pass

        with pytest.raises(ColumnOwnershipError):
            RivalBehaviour(
                world.store,
                world.columns,
                world.genetics,
                world.genes,
                world.terrain,
                world.behaviour.config,
            )

    def test_behaviour_cannot_write_a_column_it_does_not_own(self):
        world = World()
        selection = world.spawn(1)

        with pytest.raises(ColumnOwnershipError):
            world.behaviour.write("energy", selection, np.zeros(1, dtype=np.float32))


class TestRegistration:
    def test_drive_names_are_reported_in_registration_order(self):
        world = World()
        world.behaviour.register(ConstantDrive("hunger"))
        world.behaviour.register(ConstantDrive("fear"))

        assert world.behaviour.drive_names == ("hunger", "fear")

    def test_two_drives_cannot_share_a_name(self):
        """Names are how the viewer and `breakdown` address a drive; two answering to one name
        would silently report the wrong contribution.
        """
        world = World()
        world.behaviour.register(ConstantDrive("hunger"))

        with pytest.raises(DriveRegistrationError):
            world.behaviour.register(ConstantDrive("hunger"))

    def test_registering_past_the_column_width_fails_loudly(self):
        """The store's drive_scores block is allocated at a fixed width. Overflowing it is a
        world-construction error, and it is caught here rather than at the first tick (§8.7).
        """
        world = World(n_drives=2)
        world.behaviour.register(ConstantDrive("hunger"))
        world.behaviour.register(ConstantDrive("thirst"))

        with pytest.raises(DriveRegistrationError):
            world.behaviour.register(ConstantDrive("fear"))

    def test_a_drive_reporting_the_wrong_urgency_length_fails_loudly(self):
        """A scalar or length-1 return would broadcast into the score column silently, giving every
        entity one animal's motivation.
        """
        world = World()
        selection = world.spawn(3)

        class ScalarUrgency:
            name = "broken"

            def urgency(self, selection):
                return np.float32(1.0)

            def appeal(self, selection, x, y):
                return np.ones_like(x, dtype=np.float32)

        world.behaviour.register(ScalarUrgency())

        with pytest.raises(ValueError, match="urgency"):
            world.behaviour.choose(selection, np.random.default_rng(0))

    def test_a_drive_scoring_one_column_of_appeal_fails_loudly(self):
        """The case NumPy would *not* catch: an (n, 1) return broadcasts cleanly across every
        option, so the drive would silently become indifferent and no shape error would be raised.
        """
        world = World()
        selection = world.spawn(3)

        class OneColumn:
            name = "broken"

            def urgency(self, selection):
                return np.ones(len(selection), dtype=np.float32)

            def appeal(self, selection, x, y):
                return np.ones((x.shape[0], 1), dtype=np.float32)

        world.behaviour.register(OneColumn())

        with pytest.raises(ValueError, match="appeal"):
            world.behaviour.choose(selection, np.random.default_rng(0))


class TestCandidateOptions:
    def test_headings_are_evenly_spaced_around_the_circle(self):
        world = World()
        selection = world.spawn(1)

        headings = world.behaviour.candidate_headings(selection, np.random.default_rng(0))

        assert headings.shape == (1, N_CANDIDATES)
        assert np.diff(headings[0]) == pytest.approx(np.full(N_CANDIDATES - 1, SPACING))

    def test_headings_are_jittered_per_entity(self):
        """Without the jitter every animal evaluates the identical absolute directions, so a
        population converges on the same few headings and moves in lockstep along them. The jitter
        is what makes angular resolution effectively continuous across a population (#114).
        """
        world = World(capacity=64)
        selection = world.spawn(50)

        headings = world.behaviour.candidate_headings(selection, np.random.default_rng(0))

        offsets = headings[:, 0]
        assert len(np.unique(offsets)) == 50
        assert ((offsets >= 0.0) & (offsets < SPACING)).all()

    def test_every_heading_stays_inside_one_turn(self):
        """The column is documented as radians in [0, 2pi), and the jitter is what could push the
        last candidate past a full turn if it were ever allowed to reach a whole spacing.
        """
        world = World(capacity=64)
        selection = world.spawn(50)

        headings = world.behaviour.candidate_headings(selection, np.random.default_rng(1))

        assert ((headings >= 0.0) & (headings < 2.0 * np.pi)).all()

    def test_the_null_option_is_the_animals_own_position_and_comes_last(self):
        world = World()
        selection = world.spawn(2, x=np.float32([3.0, 5.0]), y=np.float32([2.0, 6.0]))
        headings = world.behaviour.candidate_headings(selection, np.random.default_rng(0))

        x, y = world.behaviour.candidate_positions(selection, headings)

        assert x.shape == (2, N_CANDIDATES + 1)
        assert x[:, NULL] == pytest.approx([3.0, 5.0])
        assert y[:, NULL] == pytest.approx([2.0, 6.0])

    def test_candidates_are_clipped_into_the_world(self):
        """`Movement._landing` guarantees animals land exactly on the boundary, so a heading
        pointing outward from one is the ordinary case. The fields a drive samples raise outside
        their bounds, so the clip is what keeps a cornered animal scoreable at all.
        """
        world = World()
        selection = world.spawn(1, x=np.float32([0.0]), y=np.float32([0.0]))
        headings = world.behaviour.candidate_headings(selection, np.random.default_rng(0))

        x, y = world.behaviour.candidate_positions(selection, headings)

        assert ((x >= 0.0) & (x <= world.terrain.world_width)).all()
        assert ((y >= 0.0) & (y <= world.terrain.world_height)).all()

    def test_a_candidate_sits_one_look_ahead_from_the_animal(self):
        world = World()
        selection = world.spawn(1)
        headings = world.behaviour.candidate_headings(selection, np.random.default_rng(0))

        x, y = world.behaviour.candidate_positions(selection, headings)

        reach = np.hypot(x[0, :N_CANDIDATES] - 4.0, y[0, :N_CANDIDATES] - 4.0)
        assert reach == pytest.approx(np.full(N_CANDIDATES, LOOK_AHEAD))


class TestUtilities:
    def test_utility_is_urgency_times_appeal_summed_over_drives(self):
        world = World()
        selection = world.spawn(1)
        world.behaviour.register(wanting(0, urgency=0.5))
        world.behaviour.register(wanting(NULL, urgency=0.25))
        headings = world.behaviour.candidate_headings(selection, np.random.default_rng(0))
        x, y = world.behaviour.candidate_positions(selection, headings)

        total, contributions = world.behaviour.utilities(
            selection, headings, x, y, uncommitted(selection)
        )

        expected = np.zeros(N_CANDIDATES + 1)
        expected[0] = 0.5
        expected[NULL] = 0.25
        assert total[0] == pytest.approx(expected)
        assert contributions["wants_0"][0, 0] == pytest.approx(0.5)
        assert contributions["wants_8"][0, NULL] == pytest.approx(0.25)

    def test_a_mild_appetite_weighs_less_than_an_urgent_one(self):
        """Urgency scaling appeal is what makes a starving animal's food preference outweigh a
        peckish one's without either drive knowing the other exists.
        """
        world = World()
        selection = world.spawn(2)
        world.behaviour.register(
            ConstantDrive(
                "hunger",
                urgency=per_row(world.store, **{str(selection.to_indices()[0]): 1.0,
                                                str(selection.to_indices()[1]): 0.1}),
                appeal=wanting(0)._appeal,
            )
        )
        headings = world.behaviour.candidate_headings(selection, np.random.default_rng(0))
        x, y = world.behaviour.candidate_positions(selection, headings)

        total, _ = world.behaviour.utilities(
            selection, headings, x, y, uncommitted(selection)
        )

        assert total[0, 0] == pytest.approx(1.0)
        assert total[1, 0] == pytest.approx(0.1)


class TestChoosing:
    def test_a_decisive_animal_takes_its_best_option(self):
        """The jitter is drawn from the generator before the Gumbel noise is, so replaying the same
        seed through `candidate_headings` reproduces exactly the headings `choose` scored.
        """
        world = World()
        selection = world.spawn(1)
        world.behaviour.register(wanting(3))
        expected = world.behaviour.candidate_headings(selection, np.random.default_rng(7))

        world.behaviour.choose(selection, np.random.default_rng(7))

        row = selection.to_indices()[0]
        assert world.store.choice_moving[row]
        assert world.store.choice_heading[row] == pytest.approx(expected[0, 3], abs=1e-6)

    def test_appeal_on_the_null_option_keeps_an_animal_where_it_is(self):
        """Rest is an option in the same contest — no mode, no flag, no state column (#114)."""
        world = World()
        selection = world.spawn(1)
        world.behaviour.register(wanting(NULL))

        world.behaviour.choose(selection, np.random.default_rng(0))

        assert not world.store.choice_moving[selection.to_indices()[0]]

    def test_an_animal_that_stays_keeps_the_heading_it_had(self):
        """So a rested animal resumes the way it was going instead of choosing afresh from nothing,
        which is what lets change-aversion hold a direction across the several ticks recovery needs.
        """
        world = World()
        selection = world.spawn(1)
        world.behaviour.register(wanting(3))
        world.behaviour.choose(selection, np.random.default_rng(7))
        walking = world.store.choice_heading[selection.to_indices()[0]]

        world.behaviour.register(wanting(NULL, urgency=10.0))
        world.behaviour.choose(selection, np.random.default_rng(11))

        row = selection.to_indices()[0]
        assert not world.store.choice_moving[row]
        assert world.store.choice_heading[row] == pytest.approx(walking)

    def test_temperature_decides_how_much_an_animal_explores(self):
        """Boltzmann sampling, asserted as a distribution rather than as a draw (§2.2).

        Both cohorts face the identical contest — all the appeal on staying put — and differ only
        in the expressed temperature gene. The cold one always takes the best option; the warm one
        is nearly uniform over the nine, so about one in nine stays.
        """
        world = World(capacity=512)
        decisive = world.spawn(200, temperature=DECISIVE)
        reckless = world.spawn(200, temperature=RECKLESS)
        world.behaviour.register(wanting(NULL))

        world.behaviour.choose(decisive, np.random.default_rng(3))
        world.behaviour.choose(reckless, np.random.default_rng(3))

        assert not world.store.choice_moving[decisive.to_indices()].any()
        stayed = world.store.choice_moving[reckless.to_indices()].sum()
        assert 0 < (200 - stayed) < 200, "the warm cohort was not exploring at all"
        assert (200 - stayed) < 100, "the warm cohort ignored a preference it should still feel"

    def test_an_animal_with_no_reason_to_do_anything_still_chooses(self):
        """Every option scores zero, so the draw is uniform and the animal wanders. It does *not*
        default to standing still: a flat contest is indifference, and freezing on indifference is
        the failure #126 recorded — forty founders scored and not one moving.
        """
        world = World(capacity=512)
        selection = world.spawn(200)
        world.behaviour.register(ConstantDrive("nothing", urgency=0.0))

        world.behaviour.choose(selection, np.random.default_rng(5))

        moved = world.store.choice_moving[selection.to_indices()].sum()
        assert moved > 100, "an unmotivated population froze instead of wandering"

    def test_choosing_leaves_entities_outside_the_selection_untouched(self):
        world = World()
        chooser = world.spawn(1)
        bystander = world.spawn(1)
        world.behaviour.register(wanting(3))

        world.behaviour.choose(chooser, np.random.default_rng(0))

        row = bystander.to_indices()[0]
        assert world.store.choice_heading[row] == pytest.approx(0.0)
        assert not world.store.choice_moving[row]
        assert world.behaviour.scores(bystander) == pytest.approx(np.zeros((1, 4)))


def turn_from(world, selection, previous):
    """The absolute angle between each entity's stored heading and `previous`, wrapped to [0, pi]."""
    delta = world.store.choice_heading[selection.to_indices()] - previous
    return np.abs(np.arctan2(np.sin(delta), np.cos(delta)))


class TestCommitment:
    """The bonus for continuing last tick's bearing, and the fact that it is a gene (#100).

    #114 built the term with a per-world constant; this is what makes its width a heritable trait,
    so a dithering lineage and a lineage that cannot be interrupted are both selected against
    rather than one coefficient being tuned to sit between them.
    """

    def test_an_option_continuing_last_tick_is_worth_more_than_one_reversing_it(self):
        """Rewarded by *how well* the bearing is continued rather than by an equality test: `cos`
        of the turn falls off smoothly, so a slight correction keeps almost all of the bonus and a
        reversal loses it. An option-index comparison could not express that, which is why the
        column stores a heading (#114).
        """
        # Three candidates rather than eight, so the turn each one represents can be named exactly:
        # straight on, a right angle, and a reversal.
        world = World(n_candidates=3)
        selection = world.spawn(1)
        world.store.choice_heading[selection.to_indices()] = 0.0
        headings = np.array([[0.0, np.pi / 2, np.pi]])
        x, y = world.behaviour.candidate_positions(selection, headings)

        total, _ = world.behaviour.utilities(selection, headings, x, y, np.array([0.5]))

        assert total[0, 0] == pytest.approx(0.5)  # straight on: the whole bonus
        assert total[0, 1] == pytest.approx(0.0, abs=1e-9)  # a right-angle turn: none of it
        assert total[0, 2] == pytest.approx(-0.5)  # a reversal: the bonus paid back
        assert total[0, 3] == pytest.approx(0.0)  # the null option continues no direction

    def test_each_animal_holds_its_bearing_by_its_own_commitment(self):
        """The whole of #100: the band is per-entity, so two animals in one vectorized pass weigh
        the same turn differently. A constant could not express a dogged animal beside a flighty
        one, which is what selection needs something to act on.
        """
        world = World(n_candidates=3)
        selection = world.spawn(2)
        world.store.choice_heading[selection.to_indices()] = 0.0
        headings = np.zeros((2, 3))
        x, y = world.behaviour.candidate_positions(selection, headings)

        total, _ = world.behaviour.utilities(
            selection, headings, x, y, np.array([0.9, 0.1])
        )

        assert total[0, 0] == pytest.approx(0.9)
        assert total[1, 0] == pytest.approx(0.1)

    def test_a_challenger_must_clear_the_whole_band_to_break_a_held_bearing(self):
        """Commitment is **hysteresis, not a weight**, because the bonus is applied to the
        *incumbent* bearing rather than to a fixed drive. Holding a heading needs the challenger to
        stay under `+c`, and taking it needs the challenger to exceed `-c`, so the two thresholds
        are separated by `2c` and `commitment` is what sets the band width (#100).

        Asserted on a perpendicular challenger, where the incumbent keeps the whole bonus and the
        challenger gets none of it, so the crossing point is exactly the drive's own urgency.
        """
        band = 0.5
        sideways = np.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        headings = np.array([[0.0, np.pi / 2, np.pi, 3.0 * np.pi / 2]])

        for urgency, holds in ((0.4, True), (0.6, False)):
            world = World(n_candidates=4)
            selection = world.spawn(1)
            world.store.choice_heading[selection.to_indices()] = 0.0
            world.behaviour.register(ConstantDrive("sideways", urgency=urgency, appeal=sideways))
            x, y = world.behaviour.candidate_positions(selection, headings)

            total, _ = world.behaviour.utilities(selection, headings, x, y, np.array([band]))

            assert bool(total[0, 0] > total[0, 1]) is holds, (
                f"a challenger scoring {urgency} against a band of {band} should "
                f"{'not ' if holds else ''}have taken the turn"
            )

    def test_choose_reads_the_bonus_from_the_gene(self):
        """The wiring, asserted as a distribution rather than as a draw (§2.2). Nothing is
        preferred, so only the stored bearing is left to decide: a dogged cohort keeps to the
        candidates flanking the way it was already going, while an uncommitted one draws uniformly
        and scatters over the circle.

        Not "every animal takes the nearest candidate" — the choice is *sampled*, so an animal
        whose two nearest candidates are nearly equidistant sometimes takes the further one. That
        is the temperature doing its job, not the bonus failing to (#114).
        """
        world = World(capacity=512)
        previous = 1.0
        dogged = world.spawn(200, commitment=5.0)
        flighty = world.spawn(200, commitment=0.0)
        world.store.choice_heading[dogged.to_indices()] = previous
        world.store.choice_heading[flighty.to_indices()] = previous
        world.behaviour.register(ConstantDrive("nothing", urgency=0.0))

        world.behaviour.choose(dogged, np.random.default_rng(3))
        world.behaviour.choose(flighty, np.random.default_rng(3))

        held = turn_from(world, dogged, previous)
        scattered = turn_from(world, flighty, previous)
        # One spacing is the furthest the *second*-nearest candidate can ever sit, so this says the
        # cohort never chose past its bearing's immediate neighbours.
        assert (held <= SPACING).all()
        assert held.mean() < SPACING / 2
        # A uniform draw over the circle averages a quarter-turn; anything near it is no preference.
        assert scattered.mean() > 1.0

    def test_a_commitment_gene_that_drifted_negative_is_dogged_in_the_same_way(self):
        """Read as a magnitude, so storage below zero is a strongly committed animal rather than one
        rewarded for reversing. Genes drift freely across zero (§2.5) and the expression mode is
        what keeps the bonus a bonus — nothing clamps the column.

        Asserted as equality against the mirrored cohort rather than statistically, because `abs`
        makes the two *the same animal*: same seed, same jitter, same draw, same heading.
        """
        world = World(capacity=512)
        previous = 1.0
        positive = world.spawn(200, commitment=5.0)
        negative = world.spawn(200, commitment=-5.0)
        world.store.choice_heading[positive.to_indices()] = previous
        world.store.choice_heading[negative.to_indices()] = previous
        world.behaviour.register(ConstantDrive("nothing", urgency=0.0))

        world.behaviour.choose(positive, np.random.default_rng(3))
        world.behaviour.choose(negative, np.random.default_rng(3))

        assert turn_from(world, negative, previous) == pytest.approx(
            turn_from(world, positive, previous)
        )

    def test_a_commitment_gene_that_could_express_negative_is_rejected(self):
        """A negative bonus rewards reversing, which is a spin rather than a preference — and the
        mode is what forbids it, so the world is refused at construction rather than checked per
        tick (§8.7). This is #136's rule about costs applied to a second consumer of the same
        property: what matters is that the phenotype cannot go below zero, not which mode it is.
        """
        registry = gene_registry(("mutability", "choice_temperature", "signature_0"))
        store = EntityStore(initial_capacity=1, n_drives=1, n_genes=len(registry))
        columns = ColumnRegistry()
        species = SpeciesRegistry(registry.vocabulary)

        with pytest.raises(ValueError, match="signature_0"):
            Behaviour(
                store,
                columns,
                Genetics(store, columns, species, registry, GENETICS_CONFIG),
                registry,
                Terrain(np.zeros((9, 9), dtype=np.float32), cell_size=1.0),
                BehaviourConfig(
                    n_candidates=8,
                    look_ahead=1.0,
                    commitment_gene="signature_0",
                    choice_temperature_gene="choice_temperature",
                ),
            )


class TestChosenTarget:
    def test_a_movers_target_is_one_look_ahead_along_its_stored_heading(self):
        """Recomputed from the store rather than carried out of `choose` in a variable, so the
        decision survives as a fact between the two systems §2.1 keeps separate — scoring at
        position 3 in the tick and movement at 4.
        """
        world = World()
        selection = world.spawn(1)
        world.behaviour.register(wanting(3))
        world.behaviour.choose(selection, np.random.default_rng(7))

        target_x, target_y = world.behaviour.chosen_target(selection)

        heading = world.store.choice_heading[selection.to_indices()[0]]
        assert target_x[0] == pytest.approx(4.0 + LOOK_AHEAD * np.cos(heading), abs=1e-5)
        assert target_y[0] == pytest.approx(4.0 + LOOK_AHEAD * np.sin(heading), abs=1e-5)

    def test_a_stayers_target_is_its_own_position(self):
        """`Movement.step` then prices a step of zero and it pays nothing, which is what makes rest
        recover exertion (#107) with nothing anywhere branching on a resting state.
        """
        world = World()
        selection = world.spawn(1)
        world.behaviour.register(wanting(NULL))
        world.behaviour.choose(selection, np.random.default_rng(0))

        target_x, target_y = world.behaviour.chosen_target(selection)

        assert target_x == pytest.approx([4.0])
        assert target_y == pytest.approx([4.0])

    def test_a_target_never_leaves_the_world(self):
        world = World()
        selection = world.spawn(1, x=np.float32([0.0]), y=np.float32([0.0]))
        world.behaviour.register(ConstantDrive("nothing", urgency=0.0))
        world.behaviour.choose(selection, np.random.default_rng(0))

        target_x, target_y = world.behaviour.chosen_target(selection)

        assert 0.0 <= target_x[0] <= world.terrain.world_width
        assert 0.0 <= target_y[0] <= world.terrain.world_height


class TestChoiceStateIsNotInherited:
    def test_a_newborn_has_made_no_choices(self):
        world = World()
        selection = world.spawn(1)

        assert world.store.choice_heading[selection.to_indices()[0]] == pytest.approx(0.0)
        assert not world.store.choice_moving[selection.to_indices()[0]]

    def test_a_reused_row_does_not_leak_its_predecessors_choice(self):
        """§2.1 puts death before reproduction within a tick precisely so freed rows are reusable
        immediately, which makes this the ordinary path once #20 and #21 land, not an edge case.
        """
        world = World()
        first = world.spawn(1)
        world.behaviour.register(wanting(3))
        world.behaviour.choose(first, np.random.default_rng(7))
        assert world.store.choice_moving[first.to_indices()[0]]

        world.store.release(world.store._row_to_id[first.to_mask()])
        second = world.spawn(1)

        assert second.to_mask().tolist() == first.to_mask().tolist()  # the same row, reused
        assert world.store.choice_heading[second.to_indices()[0]] == pytest.approx(0.0)
        assert not world.store.choice_moving[second.to_indices()[0]]


class TestInspection:
    def test_breakdown_reports_each_drives_share_of_the_option_taken(self):
        """Not the bare urgency: two animals with identical hunger, one facing a meadow and one
        facing bare rock, are not equally explained by "hunger 0.6". What §3.3's click-to-inspect
        needs is how much of *this* decision each drive accounts for.
        """
        world = World()
        selection = world.spawn(1)
        world.behaviour.register(wanting(3, urgency=0.75))
        world.behaviour.register(ConstantDrive("ambient", urgency=0.25, appeal=1.0))

        world.behaviour.choose(selection, np.random.default_rng(7))

        assert world.behaviour.breakdown(selection) == {
            "wants_3": pytest.approx([0.75]),
            "ambient": pytest.approx([0.25]),
        }

    def test_a_drive_that_did_not_want_the_chosen_option_reports_nothing_for_it(self):
        """The decomposition is of the option *actually taken*, so a drive that preferred a
        different one contributed nothing to this decision however urgent it was.
        """
        world = World()
        selection = world.spawn(1)
        world.behaviour.register(wanting(3, urgency=1.0))
        world.behaviour.register(wanting(5, urgency=0.2))

        world.behaviour.choose(selection, np.random.default_rng(7))

        breakdown = world.behaviour.breakdown(selection)
        assert breakdown["wants_3"] == pytest.approx([1.0])
        assert breakdown["wants_5"] == pytest.approx([0.0])

    def test_adding_a_drive_needs_no_change_to_the_contest(self):
        """#22's "done when", which #114 inherits unchanged: a new drive is registered, not wired
        into a dispatch chain. The loop runs with one drive and then two, with nothing about the
        call changing.
        """
        world = World()
        selection = world.spawn(1)
        world.behaviour.register(wanting(3, urgency=0.6))

        world.behaviour.choose(selection, np.random.default_rng(7))
        assert set(world.behaviour.breakdown(selection)) == {"wants_3"}

        world.behaviour.register(ConstantDrive("ambient", urgency=0.4, appeal=1.0))
        world.behaviour.choose(selection, np.random.default_rng(7))

        assert world.behaviour.breakdown(selection)["ambient"] == pytest.approx([0.4])

    def test_headings_are_reported_over_a_whole_selection(self):
        world = World(capacity=16)
        selection = world.spawn(6)
        world.behaviour.register(wanting(3))

        world.behaviour.choose(selection, np.random.default_rng(7))

        assert world.behaviour.headings(selection).shape == (6,)
