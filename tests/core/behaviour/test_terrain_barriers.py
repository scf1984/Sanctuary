"""Issue #25's "done when": steep terrain suppresses crossing rates, and a long run of real
movement never trips an invariant.

This is the claim that makes elevation worth its cost in the model (CLAUDE.md §2.6). A mountain
range is supposed to become an isolation barrier — the precondition for #16's emergent speciation —
without anyone placing a barrier: it is expensive to climb, so fewer animals get over it, so the
populations either side mix less. Nothing here authors that. The ridge is terrain and the animals
are identical on both maps; only the ground differs.

Statistical and directional, never exact (CLAUDE.md §6, §8.1). Cohorts are drawn from overlapping
distributions and the assertions are about which map lets more animals across, over many seeds.
Nothing below asserts a crossing count.
"""

import numpy as np
import pytest

from core.behaviour.exertion import Exertion, ExertionConfig
from core.behaviour.movement import Movement, MovementConfig
from core.ecology.metabolism import Metabolism, MetabolismConfig
from core.ecology.service import Ecology
from core.entities.store import EntityStore
from core.genetics.expression import ExpressionMode, GeneticsConfig
from core.genetics.service import Genetics
from core.genetics.species import SpeciesRegistry
from core.genetics.vocabulary import GeneVocabulary
from core.invariants import default_registry
from core.selection import Selection
from core.services import ColumnRegistry
from core.world.climate import Climate, ClimateConfig
from core.world.terrain import Terrain
from core.world.tick import TickLoop


GENE_NAMES = ("size", "speed", "insulation", "mutability")

# Every gene declares how its stored value is read (#104). These are all quantities, so all fold
# across zero; `mutability` is in the vocabulary because inheritance's spread floor is a gene, and
# every world needs one even when — as here — nothing in these tests breeds.
GENETICS_CONFIG = GeneticsConfig(
    expression_modes={name: ExpressionMode.MAGNITUDE for name in GENE_NAMES},
    mutability_gene="mutability",
    drift_margin=2.0,
)

METABOLISM_CONFIG = MetabolismConfig(
    gene_costs={"size": 0.5, "speed": 0.5, "insulation": 1.0, "mutability": 0.0},
    basal_rate=0.2,
    thermoregulation_rate=0.1,
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
WORLD_SPAN = (GRID - 1) * CELL_SIZE

COHORT_SIZE = 150
# Enough to walk the world's width several times over on the flat, so a failure to cross a ridge
# is the climb charge and not simply a short journey. Derived rather than guessed: the flat
# traverse costs transport_cost x span x (1 + exertion_premium x walking_pace) per unit of size.
FLAT_TRAVERSE_COST = (
    MOVEMENT_CONFIG.transport_cost
    * WORLD_SPAN
    * (1.0 + MOVEMENT_CONFIG.exertion_premium * MOVEMENT_CONFIG.walking_pace)
)
STARTING_ENERGY = 3.0 * FLAT_TRAVERSE_COST
MAX_TICKS = 400


def flat_heights():
    return np.zeros((GRID, GRID), dtype=np.float32)


def ridge_heights(peak):
    """A north-south ridge across the middle of the world: elevation depends on x only, so it is a
    wall an animal heading east must climb over and never a hill it can walk around."""
    x = np.arange(GRID, dtype=np.float64) * CELL_SIZE
    profile = peak * np.exp(-(((x - WORLD_SPAN / 2.0) / (WORLD_SPAN / 8.0)) ** 2))
    return np.broadcast_to(profile, (GRID, GRID)).astype(np.float32)


class World:
    def __init__(self, heights, capacity=512):
        self.store = EntityStore(initial_capacity=capacity, n_drives=1, n_genes=len(GENE_NAMES))
        self.registry = ColumnRegistry()
        self.vocabulary = GeneVocabulary(GENE_NAMES)
        self.species = SpeciesRegistry(self.vocabulary)
        self.genetics = Genetics(self.store, self.registry, self.species, self.vocabulary, GENETICS_CONFIG)
        self.terrain = Terrain(heights, cell_size=CELL_SIZE)
        self.climate = Climate(
            self.terrain,
            ClimateConfig(equator_y=0.0, equator_temperature=20.0, latitude_gradient=0.0),
        )
        self.ecology = Ecology(
            self.store,
            self.registry,
            self.genetics,
            self.climate,
            Metabolism(self.vocabulary, METABOLISM_CONFIG, GENETICS_CONFIG.expression_modes),
        )
        self.exertion = Exertion(self.store, self.registry, ExertionConfig(recovery_rate=0.5))
        self.movement = Movement(
            self.store,
            self.registry,
            self.ecology,
            self.exertion,
            self.genetics,
            self.terrain,
            self.vocabulary,
            MOVEMENT_CONFIG,
        )
        self.species_id = self.species.register(GENE_NAMES)

    def add_cohort(self, rng, n, x, energy=STARTING_ENERGY):
        """A cohort spread along the western edge, with individual variation in size and speed so
        the comparison has to survive overlapping distributions rather than two uniform blocks."""
        genes = np.zeros((n, len(GENE_NAMES)), dtype=np.float32)
        genes[:, GENE_NAMES.index("speed")] = np.clip(rng.normal(2.0, 0.3, n), 0.1, None)
        genes[:, GENE_NAMES.index("size")] = np.clip(rng.normal(1.0, 0.15, n), 0.1, None)

        ids = self.store.allocate(
            n,
            x=np.full(n, x, dtype=np.float32),
            y=rng.uniform(0.0, WORLD_SPAN, n).astype(np.float32),
            energy=np.full(n, energy, dtype=np.float32),
            species_id=np.full(n, self.species_id, dtype=np.int32),
        )
        rows = np.array([self.store._id_to_row[i] for i in ids.tolist()], dtype=np.int64)
        selection = Selection.from_indices(rows, capacity=self.store.capacity)
        self.genetics.set_genes(selection, genes)
        self.movement.settle(selection)
        return selection


def crossings(heights, seed, max_ticks=MAX_TICKS):
    """How many of a westward cohort reach the far edge, walking due east every tick.

    Every animal wants the same thing and heads straight for it, so the only thing that can stop
    one is what the ground charged on the way. Metabolic upkeep runs alongside, because an animal
    slowed to a shuffle by a climb is still paying to be alive — that interaction is the point,
    not a contaminant.
    """
    rng = np.random.default_rng(seed)
    world = World(heights)
    cohort = world.add_cohort(rng, COHORT_SIZE, x=0.0)
    east_x = np.full(COHORT_SIZE, WORLD_SPAN, dtype=np.float64)
    east_y = world.store.y[cohort.to_mask()].astype(np.float64)

    for _ in range(max_ticks):
        world.movement.step(cohort, east_x, east_y, MOVEMENT_CONFIG.walking_pace)
        world.ecology.drain(cohort)

    return int((world.store.x[cohort.to_mask()] >= WORLD_SPAN).sum())


class TestSteepTerrainSuppressesCrossingRates:
    @pytest.mark.parametrize("seed", range(8))
    def test_a_ridge_stops_animals_that_flat_ground_lets_through(self, seed):
        """The same cohort, the same distance, the same energy — only the relief differs."""
        assert crossings(ridge_heights(peak=400.0), seed) < crossings(flat_heights(), seed)

    @pytest.mark.parametrize("seed", range(8))
    def test_flat_ground_is_crossed_by_essentially_everyone(self, seed):
        """The other half of the comparison: without it, a ridge "suppressing" crossings could be
        a cohort that never had the energy to make the traverse at all.
        """
        assert crossings(flat_heights(), seed) > 0.9 * COHORT_SIZE

    @pytest.mark.parametrize("seed", range(4))
    def test_a_higher_ridge_is_a_stronger_barrier_than_a_lower_one(self, seed):
        """Barrier strength is continuous in relief, which is what lets #16's isolation be a matter
        of degree rather than a wall someone placed.
        """
        gentle = crossings(ridge_heights(peak=60.0), seed)
        severe = crossings(ridge_heights(peak=400.0), seed)

        assert severe < gentle

    def test_animals_stopped_by_the_ridge_are_stranded_on_it_rather_than_teleported_past(self):
        """Guards the mechanism, not just the count: a barrier that worked by deleting animals or
        by snapping them to the target would produce the same crossing statistic.
        """
        rng = np.random.default_rng(3)
        world = World(ridge_heights(peak=400.0))
        cohort = world.add_cohort(rng, COHORT_SIZE, x=0.0)
        east_x = np.full(COHORT_SIZE, WORLD_SPAN, dtype=np.float64)
        east_y = world.store.y[cohort.to_mask()].astype(np.float64)

        for _ in range(MAX_TICKS):
            world.movement.step(cohort, east_x, east_y, MOVEMENT_CONFIG.walking_pace)
            world.ecology.drain(cohort)

        x = world.store.x[cohort.to_mask()]
        stalled = x < WORLD_SPAN
        assert stalled.any()
        # They got somewhere — they are partway up the west face, not still on the start line and
        # not over the top.
        assert (x[stalled] > 0.0).all()
        assert (world.ecology.energy(cohort)[stalled] == 0.0).all()


class TestALongRunOfMovementHoldsEveryInvariant:
    def test_a_thousand_ticks_of_wandering_never_leaves_the_world(self):
        """Movement runs as a registered system under the real tick loop with debug checks on, so
        #7's harness — including "no alive entity leaves world bounds" and "no alive entity has
        negative energy" — is evaluated after every one of the 1000 ticks, not just at the end.

        The targets are redrawn each tick from anywhere in the world, so animals are pushed at the
        boundary repeatedly rather than settling in the middle. That is what makes the bounds
        invariant a real assertion here: `step` never clamps a position, and this is the run that
        would catch it if a step could overshoot its target onto ground `Terrain` cannot sample.
        """
        rng = np.random.default_rng(11)
        world = World(ridge_heights(peak=200.0))
        # Deep pools: the claim under test is geometric, and a cohort that starves in tick 30 stops
        # exercising the boundary for the remaining 970.
        cohort = world.add_cohort(rng, COHORT_SIZE, x=0.0, energy=1e7)

        def wander():
            world.movement.step(
                cohort,
                rng.uniform(0.0, WORLD_SPAN, COHORT_SIZE),
                rng.uniform(0.0, WORLD_SPAN, COHORT_SIZE),
                MOVEMENT_CONFIG.walking_pace,
            )
            world.ecology.drain(cohort)

        loop = TickLoop(
            world.store,
            systems=[wander],
            invariants=default_registry(0.0, WORLD_SPAN, 0.0, WORLD_SPAN),
            debug_checks=True,
        )

        loop.advance(1000)

        assert loop.tick_count == 1000
        # The renderer interpolates between these two snapshots (§2.1); a run in which nothing
        # moved would satisfy every invariant above and mean nothing.
        assert not np.array_equal(loop.previous_positions[0], loop.current_positions[0])

    def test_the_surface_lock_holds_for_every_entity_after_a_long_run(self):
        """Surface-locked z (§2.6) is an invariant of this service rather than of the harness: the
        harness watches the world's rules, and "z follows terrain" is true only until flight
        unlocks it, which §2.6 stages deliberately.
        """
        rng = np.random.default_rng(12)
        world = World(ridge_heights(peak=200.0))
        cohort = world.add_cohort(rng, COHORT_SIZE, x=0.0, energy=1e7)

        for _ in range(200):
            world.movement.step(
                cohort,
                rng.uniform(0.0, WORLD_SPAN, COHORT_SIZE),
                rng.uniform(0.0, WORLD_SPAN, COHORT_SIZE),
                MOVEMENT_CONFIG.walking_pace,
            )

        mask = cohort.to_mask()
        ground = world.terrain.elevation_at(world.store.x[mask], world.store.y[mask])
        assert world.store.z[mask] == pytest.approx(ground, rel=1e-5)
