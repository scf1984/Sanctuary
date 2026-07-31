"""Exertion: the column that records effort, and the fatigue term that reads it (issue #107).

The test the issue asks for is `test_a_sprinting_cohort_outscores_a_resting_one_at_equal_health`,
and it fails against the pre-#107 code by construction: fatigue scored `weight × (1 - health)`, so
two cohorts at equal health scored identically no matter what either had been doing.
"""

import numpy as np
import pytest

from core.behaviour.drives import Fatigue, FatigueConfig
from core.behaviour.exertion import Exertion, ExertionConfig
from core.behaviour.movement import Movement, MovementConfig
from core.ecology.metabolism import Metabolism, MetabolismConfig
from core.ecology.service import Ecology
from core.entities.store import EntityStore
from core.genetics.expression import GeneticsConfig
from core.genetics.service import Genetics
from core.genetics.species import SpeciesRegistry
from core.selection import Selection
from core.services import ColumnOwnershipError, ColumnRegistry
from core.world.climate import Climate, ClimateConfig
from core.world.terrain import Terrain

from tests.support.genes import gene_registry
from tests.support.plants import plant_field

GENE_NAMES = ("size", "speed", "insulation", "mutability")

# Every gene declares how its stored value is read (#104). These are all quantities, so all fold
# across zero; `mutability` is in the vocabulary because inheritance's spread floor is a gene, and
# every world needs one even when — as here — nothing in these tests breeds.
GENETICS_CONFIG = GeneticsConfig(
    mutability_gene="mutability",
    drift_margin=2.0,
)
GENE_REGISTRY = gene_registry(GENE_NAMES, {"insulation": 1.0})

# Nothing but locomotion moves an energy pool here, so a cohort's exertion is attributable to the
# steps under test. Insulation carries a cost because MetabolismConfig requires one, and no cohort
# below expresses it.
FREE_METABOLISM = MetabolismConfig(
    basal_rate=0.0,
    thermoregulation_rate=0.0,
    neutral_temperature=20.0,
    insulation_gene="insulation",
)

MOVEMENT_CONFIG = MovementConfig(
    speed_gene="speed",
    size_gene="size",
    transport_cost=1.0,
    exertion_premium=2.0,
    climb_cost=0.5,
    walking_pace=0.4,
)

GRID = 41
CELL_SIZE = 1.0


def flat_heights():
    return np.zeros((GRID, GRID), dtype=np.float32)


def ramp_heights(gain_per_unit):
    """Ground rising steadily along +x, so an eastward step climbs and the identical westward step
    descends."""
    x = np.arange(GRID, dtype=np.float32) * CELL_SIZE
    return np.broadcast_to(x * gain_per_unit, (GRID, GRID)).astype(np.float32)


class World:
    """A store plus the services exertion sits between: movement fills it, fatigue reads it."""

    def __init__(self, recovery_rate=0.25, exertion_saturation=10.0, heights=None):
        self.store = EntityStore(initial_capacity=64, n_drives=1, n_genes=len(GENE_NAMES))
        self.registry = ColumnRegistry()
        self.genes = GENE_REGISTRY
        self.species = SpeciesRegistry(self.genes.vocabulary)
        self.genetics = Genetics(self.store, self.registry, self.species, self.genes, GENETICS_CONFIG)
        self.terrain = Terrain(
            flat_heights() if heights is None else heights, cell_size=CELL_SIZE
        )
        self.climate = Climate(
            self.terrain,
            ClimateConfig(equator_y=0.0, equator_temperature=20.0, latitude_gradient=0.0),
        )
        self.ecology = Ecology(
            self.store,
            self.registry,
            self.genetics,
            self.climate,
            Metabolism(self.genes, FREE_METABOLISM),
            plant_field(self.terrain, self.climate),
        )
        self.exertion = Exertion(
            self.store, self.registry, ExertionConfig(recovery_rate=recovery_rate)
        )
        self.movement = Movement(
            self.store,
            self.registry,
            self.ecology,
            self.exertion,
            self.genetics,
            self.terrain,
            self.genes,
            MOVEMENT_CONFIG,
        )
        self.fatigue = Fatigue(
            self.store,
            self.exertion,
            FatigueConfig(weight=1.0, exertion_saturation=exertion_saturation),
        )
        self.species_id = self.species.register(GENE_NAMES)

    def place(self, x, y, *, speed=5.0, size=1.0, health=1.0, energy=1e6):
        x = np.atleast_1d(np.asarray(x, dtype=np.float32))
        y = np.broadcast_to(np.atleast_1d(np.asarray(y, dtype=np.float32)), x.shape)
        n = x.shape[0]
        genes = np.zeros((n, len(GENE_NAMES)), dtype=np.float32)
        genes[:, GENE_NAMES.index("speed")] = speed
        genes[:, GENE_NAMES.index("size")] = size

        ids = self.store.allocate(
            n,
            x=x,
            y=y.astype(np.float32),
            energy=np.full(n, energy, dtype=np.float32),
            health=np.full(n, health, dtype=np.float32),
            species_id=np.full(n, self.species_id, dtype=np.int32),
        )
        rows = np.array([self.store._id_to_row[i] for i in ids.tolist()], dtype=np.int64)
        selection = Selection.from_indices(rows, capacity=self.store.capacity)
        self.genetics.set_genes(selection, genes)
        self.movement.settle(selection)
        return selection

    def step_toward(self, selection, target_x, target_y, pace):
        n = len(selection)
        self.movement.step(
            selection,
            np.full(n, target_x, dtype=np.float64),
            np.full(n, target_y, dtype=np.float64),
            pace,
        )


class TestExertionConfig:
    def test_zero_recovery_is_rejected(self):
        """Nothing would ever recover: the first animal to move is permanently exhausted, and the
        drive stops distinguishing anything."""
        with pytest.raises(ValueError, match="recovery_rate"):
            ExertionConfig(recovery_rate=0.0)

    def test_recovery_above_one_is_rejected(self):
        """Shedding more than everything would make exertion negative, and a negative exertion is
        an animal that rests its way to being fresher than fresh."""
        with pytest.raises(ValueError, match="recovery_rate"):
            ExertionConfig(recovery_rate=1.5)

    def test_complete_recovery_in_one_tick_is_allowed(self):
        """The closed end of the range is a real world, not a degenerate one: fatigue then reflects
        only the tick just gone."""
        assert ExertionConfig(recovery_rate=1.0).recovery_rate == 1.0


class TestFatigueConfig:
    def test_zero_saturation_is_rejected(self):
        with pytest.raises(ValueError, match="exertion_saturation"):
            FatigueConfig(weight=1.0, exertion_saturation=0.0)


class TestMovementFillsTheColumn:
    def test_a_newly_placed_animal_has_done_no_work(self):
        world = World()
        cohort = world.place([5.0], 5.0)

        assert world.exertion.exerted(cohort) == pytest.approx([0.0])

    def test_walking_accumulates_exertion(self):
        world = World()
        cohort = world.place([5.0], 5.0)

        world.step_toward(cohort, 9.0, 5.0, pace=MOVEMENT_CONFIG.walking_pace)

        assert world.exertion.exerted(cohort)[0] > 0.0

    def test_standing_still_accumulates_none(self):
        """`Hunger.forage_target` returns the animal's own position whenever nothing edible is in
        sight, so a step onto where you already are is an ordinary tick, not an edge case."""
        world = World()
        cohort = world.place([5.0], 5.0)

        world.step_toward(cohort, 5.0, 5.0, pace=MOVEMENT_CONFIG.walking_pace)

        assert world.exertion.exerted(cohort) == pytest.approx([0.0])

    def test_sprinting_the_same_ground_is_more_tiring_than_walking(self):
        """The exertion premium is per world unit travelled (§2.5), so pace has to show up in the
        record of effort and not only in the bill. Both cohorts cover the same distance."""
        world = World()
        walker = world.place([5.0], 5.0)
        sprinter = world.place([5.0], 9.0)

        world.step_toward(walker, 7.0, 5.0, pace=MOVEMENT_CONFIG.walking_pace)
        world.step_toward(sprinter, 7.0, 9.0, pace=1.0)

        assert world.exertion.exerted(sprinter)[0] > world.exertion.exerted(walker)[0]

    def test_climbing_is_more_tiring_than_the_same_distance_on_the_flat(self):
        """Two worlds rather than one, because a single terrain cannot be both flat and sloped."""
        flat = World()
        sloped = World(heights=ramp_heights(0.5))

        on_flat = flat.place([5.0], 5.0)
        uphill = sloped.place([5.0], 5.0)
        flat.step_toward(on_flat, 9.0, 5.0, pace=0.5)
        sloped.step_toward(uphill, 9.0, 5.0, pace=0.5)

        assert sloped.exertion.exerted(uphill)[0] > flat.exertion.exerted(on_flat)[0]

    def test_body_size_changes_the_bill_but_not_the_tiredness(self):
        """Exertion is work per unit of body size, so one `exertion_saturation` means the same
        tiredness to a mouse and to an elephant — while the elephant still burns more fuel."""
        world = World()
        small = world.place([5.0], 5.0, size=1.0)
        large = world.place([5.0], 9.0, size=4.0)
        before_small = world.ecology.energy(small)[0]
        before_large = world.ecology.energy(large)[0]

        world.step_toward(small, 9.0, 5.0, pace=0.5)
        world.step_toward(large, 9.0, 9.0, pace=0.5)

        assert world.exertion.exerted(large)[0] == pytest.approx(
            world.exertion.exerted(small)[0]
        )
        spent_small = before_small - world.ecology.energy(small)[0]
        spent_large = before_large - world.ecology.energy(large)[0]
        assert spent_large == pytest.approx(4.0 * spent_small, rel=1e-4)


class TestRecovery:
    def test_resting_sheds_a_fixed_fraction(self):
        world = World(recovery_rate=0.25)
        cohort = world.place([5.0], 5.0)
        world.step_toward(cohort, 9.0, 5.0, pace=1.0)
        after_moving = world.exertion.exerted(cohort)[0]

        world.exertion.recover(cohort)

        assert world.exertion.exerted(cohort)[0] == pytest.approx(0.75 * after_moving, rel=1e-5)

    def test_recovery_never_drives_exertion_negative(self):
        """Geometric decay approaches zero without crossing it, so an animal that rests forever
        becomes fresh rather than superhuman — and no clamp is doing that work."""
        world = World(recovery_rate=1.0)
        cohort = world.place([5.0], 5.0)
        world.step_toward(cohort, 9.0, 5.0, pace=1.0)

        for _ in range(5):
            world.exertion.recover(cohort)

        assert world.exertion.exerted(cohort)[0] == pytest.approx(0.0)

    def test_an_idle_animal_recovers_too(self):
        """Recovery runs over the whole living population, not only whoever moved: an animal that
        did not move is exactly the one recovery is for."""
        world = World(recovery_rate=0.5)
        mover = world.place([5.0], 5.0)
        idler = world.place([5.0], 9.0)
        world.step_toward(mover, 9.0, 5.0, pace=1.0)
        both = Selection.from_mask(mover.to_mask() | idler.to_mask())
        exerted = world.exertion.exerted(both).max()

        # The idler never appears in a `step` call, so only a whole-population pass reaches it.
        world.exertion.recover(both)

        assert world.exertion.exerted(both).max() == pytest.approx(0.5 * exerted, rel=1e-5)


class TestFatigueReadsBoth:
    def test_a_sprinting_cohort_outscores_a_resting_one_at_equal_health(self):
        """The test #107 was filed for. At equal health the pre-#107 score was identical for both
        cohorts, because health deficit was the whole of it.

        Saturation is set well above what this scenario accumulates, so the sprinter's score is a
        graded reading rather than a clipped 1.0 — a clipped one would pass this assertion while
        hiding whether the term responds to effort at all.
        """
        world = World(exertion_saturation=200.0)
        sprinter = world.place([5.0], 5.0, health=1.0)
        rester = world.place([5.0], 9.0, health=1.0)

        for _ in range(3):
            world.step_toward(sprinter, 9.0, 5.0, pace=1.0)
            world.step_toward(sprinter, 5.0, 5.0, pace=1.0)
            world.exertion.recover(sprinter)
            world.exertion.recover(rester)

        assert world.fatigue.urgency(sprinter)[0] > world.fatigue.urgency(rester)[0]

    def test_a_healthy_idle_animal_wants_nothing(self):
        """A drive scoring zero cannot win (`Behaviour.winning_drive`), so a fresh healthy animal
        must score exactly zero rather than nearly zero."""
        world = World()
        cohort = world.place([5.0], 5.0, health=1.0)

        assert world.fatigue.urgency(cohort) == pytest.approx([0.0])

    def test_injury_alone_still_scores(self):
        """The term that existed before #107 is unchanged in isolation: an animal that has not
        moved but is hurt has a real reason to rest."""
        world = World()
        cohort = world.place([5.0], 5.0, health=0.25)

        assert world.fatigue.urgency(cohort) == pytest.approx([0.75])

    def test_hurt_and_spent_exceeds_either_alone(self):
        """Noisy-OR, per §2.5: independent reasons to do one thing compound rather than replacing
        each other, and neither alone can saturate the score."""
        # Saturation well above what one sprint costs, so the exhaustion term is partial: at
        # saturation both cohorts would read 1.0 and the comparison would say nothing.
        world = World(exertion_saturation=48.0)
        hurt = world.place([5.0], 5.0, health=0.5)
        spent = world.place([5.0], 9.0, health=1.0)
        both = world.place([5.0], 13.0, health=0.5)

        for cohort in (spent, both):
            world.step_toward(cohort, 9.0, float(world.store.y[cohort.to_mask()][0]), pace=1.0)

        score_hurt = world.fatigue.urgency(hurt)[0]
        score_spent = world.fatigue.urgency(spent)[0]
        score_both = world.fatigue.urgency(both)[0]
        assert score_both > score_hurt
        assert score_both > score_spent

    def test_the_urgency_never_exceeds_its_weight(self):
        """Bounded in [0, 1] before weighting, like every other drive — which is the property that
        lets a third reason to rest be added later without retuning anything else."""
        world = World(exertion_saturation=0.01)
        cohort = world.place([5.0], 5.0, health=0.0)
        world.step_toward(cohort, 20.0, 5.0, pace=1.0)

        assert world.fatigue.urgency(cohort)[0] == pytest.approx(1.0)


class TestOwnership:
    def test_only_exertion_owns_the_column(self):
        """`Movement` hands its work over rather than writing the column, the same relationship it
        has with `Ecology` and the energy pool (§2.3)."""
        world = World()

        assert world.registry.owner_of("exertion") == "Exertion"
        with pytest.raises(ColumnOwnershipError):
            world.movement.write("exertion", world.place([5.0], 5.0), np.zeros(1))

    def test_negative_work_is_rejected(self):
        """Moving must never be a way to become less tired (§8.7)."""
        world = World()
        cohort = world.place([5.0], 5.0)

        with pytest.raises(ValueError, match="non-negative"):
            world.exertion.accumulate(cohort, np.array([-1.0]))

    def test_a_scalar_work_argument_is_rejected(self):
        """A length-1 array broadcasts cleanly across a selection and would credit one animal's
        effort to the whole cohort."""
        world = World()
        cohort = world.place([5.0, 9.0], 5.0)

        with pytest.raises(ValueError, match="shape"):
            world.exertion.accumulate(cohort, np.array([1.0]))


class TestReusedRows:
    def test_a_reused_row_starts_fresh(self):
        """`allocate` resets exertion for the same reason it resets age: a newborn inherits neither
        its predecessor's years nor its tiredness."""
        world = World()
        first = world.place([5.0], 5.0)
        world.step_toward(first, 9.0, 5.0, pace=1.0)
        assert world.exertion.exerted(first)[0] > 0.0
        ids = world.store._row_to_id[first.to_mask()]

        world.store.release(ids)
        second = world.place([5.0], 5.0)

        assert second.to_mask().tolist() == first.to_mask().tolist()  # the same row, reused
        assert world.exertion.exerted(second) == pytest.approx([0.0])
