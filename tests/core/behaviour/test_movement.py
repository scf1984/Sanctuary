"""Movement: straight-line integration toward a target, priced by distance, pace, size and climb
(issue #25).

The contract here is checkable in advance, so these were written before the implementation
(CLAUDE.md §8.1). What is *not* test-driven is what the coefficients should be — that is ecological
tuning — so nothing below asserts an energy figure. Every assertion is either a geometric identity
(where a step lands) or a direction (uphill costs more than downhill, sprinting costs more per
world unit than walking).
"""

import numpy as np
import pytest

from core.behaviour.exertion import Exertion, ExertionConfig
from core.behaviour.movement import Movement, MovementConfig
from core.ecology.metabolism import Metabolism, MetabolismConfig
from core.ecology.service import Ecology
from core.entities.store import EntityStore
from core.genetics.service import Genetics
from core.genetics.species import SpeciesRegistry
from core.genetics.vocabulary import GeneVocabulary
from core.selection import Selection
from core.services import ColumnRegistry, ColumnOwnershipError
from core.world.climate import Climate, ClimateConfig
from core.world.terrain import Terrain


GENE_NAMES = ("size", "speed", "insulation")

# Free metabolism: every gene costs nothing and there is no basal rate, so the only thing that
# moves an energy pool in these tests is the locomotion charge under test. Insulation still carries
# a positive cost because MetabolismConfig requires it to (a free insulation gene is a free lunch),
# and no cohort below expresses a non-zero value for it.
FREE_METABOLISM = MetabolismConfig(
    gene_costs={"size": 0.0, "speed": 0.0, "insulation": 1.0},
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


def flat_heights(elevation=0.0):
    return np.full((GRID, GRID), elevation, dtype=np.float32)


def ramp_heights(gain_per_unit):
    """Ground rising steadily along +x: elevation = gain_per_unit * x, so a step east climbs and
    the identical step west descends."""
    x = np.arange(GRID, dtype=np.float32) * CELL_SIZE
    return np.broadcast_to(x * gain_per_unit, (GRID, GRID)).astype(np.float32)


def ravine_heights(depth, floor_x, wall_width):
    """Flat ground with a V-shaped ravine: elevation dips to -depth at `floor_x` and climbs back.

    The shape #113 exists for. A step that starts and ends on the rim has **zero net elevation
    change**, so pricing it from its endpoints alone charges nothing for the climb out — while an
    animal that walked it did every world unit of that climb.
    """
    x = np.arange(GRID, dtype=np.float32) * CELL_SIZE
    dip = np.clip(1.0 - np.abs(x - floor_x) / wall_width, 0.0, 1.0)
    return np.broadcast_to(-depth * dip, (GRID, GRID)).astype(np.float32)


def ridge_heights(height, crest_x, width):
    """The mirror of `ravine_heights`: a hump cresting at `crest_x`."""
    return -ravine_heights(height, crest_x, width)


class World:
    """A store plus the services movement needs, wired the way a real world would wire them."""

    def __init__(self, heights, movement_config=MOVEMENT_CONFIG, capacity=256):
        self.store = EntityStore(initial_capacity=capacity, n_drives=1, n_genes=len(GENE_NAMES))
        self.registry = ColumnRegistry()
        self.vocabulary = GeneVocabulary(GENE_NAMES)
        self.species = SpeciesRegistry(self.vocabulary)
        self.genetics = Genetics(self.store, self.registry, self.species)
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
            Metabolism(self.vocabulary, FREE_METABOLISM),
        )
        # Exertion is where movement records effort (#107); nothing here asserts on it, so the
        # recovery rate is arbitrary. `tests/core/behaviour/test_exertion.py` is what covers it.
        self.exertion = Exertion(self.store, self.registry, ExertionConfig(recovery_rate=0.5))
        self.movement = Movement(
            self.store,
            self.registry,
            self.ecology,
            self.exertion,
            self.genetics,
            self.terrain,
            self.vocabulary,
            movement_config,
        )
        self.species_id = self.species.register(GENE_NAMES)

    def place(self, x, y, *, speed, size=1.0, energy=1e6):
        """Allocate one entity per (x, y) pair and settle it onto the surface."""
        x = np.atleast_1d(np.asarray(x, dtype=np.float32))
        y = np.atleast_1d(np.asarray(y, dtype=np.float32))
        n = x.shape[0]
        genes = np.zeros((n, len(GENE_NAMES)), dtype=np.float32)
        genes[:, GENE_NAMES.index("speed")] = speed
        genes[:, GENE_NAMES.index("size")] = size

        ids = self.store.allocate(
            n,
            x=x,
            y=np.broadcast_to(y, (n,)).astype(np.float32),
            energy=np.full(n, energy, dtype=np.float32),
            species_id=np.full(n, self.species_id, dtype=np.int32),
        )
        rows = np.array([self.store._id_to_row[i] for i in ids.tolist()], dtype=np.int64)
        selection = Selection.from_indices(rows, capacity=self.store.capacity)
        self.genetics.set_genes(selection, genes)
        self.movement.settle(selection)
        return selection

    def position(self, selection):
        mask = selection.to_mask()
        return self.store.x[mask], self.store.y[mask], self.store.z[mask]

    def energy(self, selection):
        return self.ecology.energy(selection)

    def step_toward(self, selection, target_x, target_y, pace):
        n = len(selection)
        self.movement.step(
            selection,
            np.full(n, target_x, dtype=np.float64),
            np.full(n, target_y, dtype=np.float64),
            pace,
        )


class TestConfigRejectsValuesThatWouldBreakTheEnergyBudget:
    def test_free_transport_is_rejected(self):
        """A zero transport cost makes distance free, which removes the only thing that stops an
        animal crossing the world every tick and defeats §2.5's hard budget."""
        with pytest.raises(ValueError, match="transport_cost"):
            MovementConfig(
                speed_gene="speed",
                size_gene="size",
                transport_cost=0.0,
                exertion_premium=2.0,
                climb_cost=0.5,
                walking_pace=0.4,
            )

    def test_a_negative_exertion_premium_is_rejected(self):
        """Negative would make sprinting cheaper per world unit than walking — the inverse of §2.5."""
        with pytest.raises(ValueError, match="exertion_premium"):
            MovementConfig(
                speed_gene="speed",
                size_gene="size",
                transport_cost=1.0,
                exertion_premium=-0.1,
                climb_cost=0.5,
                walking_pace=0.4,
            )

    def test_a_negative_climb_cost_is_rejected(self):
        """Negative climb cost is energy created by walking uphill, which §2.5's closed loop
        forbids outright."""
        with pytest.raises(ValueError, match="climb_cost"):
            MovementConfig(
                speed_gene="speed",
                size_gene="size",
                transport_cost=1.0,
                exertion_premium=2.0,
                climb_cost=-1.0,
                walking_pace=0.4,
            )

    @pytest.mark.parametrize("pace", [0.0, 1.5])
    def test_walking_pace_must_be_a_real_fraction_of_top_speed(self, pace):
        with pytest.raises(ValueError, match="walking_pace"):
            MovementConfig(
                speed_gene="speed",
                size_gene="size",
                transport_cost=1.0,
                exertion_premium=2.0,
                climb_cost=0.5,
                walking_pace=pace,
            )


class TestAStepGoesWhereItWasAimed:
    def test_it_travels_pace_times_top_speed_along_the_straight_line(self):
        world = World(flat_heights())
        walker = world.place(10.0, 10.0, speed=5.0)

        world.step_toward(walker, 30.0, 10.0, pace=1.0)

        x, y, _ = world.position(walker)
        assert x == pytest.approx(15.0)
        assert y == pytest.approx(10.0)

    def test_a_diagonal_target_moves_it_on_both_axes(self):
        """The prototype's `Vector.angle` computed `atan2(x, y)` — arguments reversed — so a
        diagonal came out mirrored about the 45-degree line. Movement never goes through an angle
        at all (a unit vector is what integration needs), and this is the assertion that would
        have caught that bug."""
        world = World(flat_heights())
        walker = world.place(0.0, 0.0, speed=5.0)

        # Target at (3, 4): distance 5, so one full-pace tick lands exactly on it.
        world.step_toward(walker, 3.0, 4.0, pace=1.0)

        x, y, _ = world.position(walker)
        assert x == pytest.approx(3.0)
        assert y == pytest.approx(4.0)

    def test_it_stops_on_the_target_rather_than_overshooting(self):
        world = World(flat_heights())
        walker = world.place(10.0, 10.0, speed=50.0)

        world.step_toward(walker, 12.0, 10.0, pace=1.0)

        x, y, _ = world.position(walker)
        assert (x, y) == pytest.approx((12.0, 10.0))

    def test_an_animal_already_on_its_target_neither_moves_nor_pays(self):
        """`Hunger.forage_target` returns the animal's own position when it can see no food, so a
        zero-length step is a normal tick and not an edge case."""
        world = World(flat_heights())
        walker = world.place(10.0, 10.0, speed=5.0)
        before = world.energy(walker).copy()

        world.step_toward(walker, 10.0, 10.0, pace=1.0)

        x, y, _ = world.position(walker)
        assert (x, y) == pytest.approx((10.0, 10.0))
        assert world.energy(walker) == pytest.approx(before)

    def test_a_walking_pace_covers_less_ground_than_a_sprint(self):
        world = World(flat_heights())
        walker = world.place(0.0, 10.0, speed=5.0)
        sprinter = world.place(0.0, 10.0, speed=5.0)

        world.step_toward(walker, 40.0, 10.0, pace=MOVEMENT_CONFIG.walking_pace)
        world.step_toward(sprinter, 40.0, 10.0, pace=1.0)

        assert world.position(walker)[0] < world.position(sprinter)[0]

    def test_top_speed_is_the_expressed_speed_gene(self):
        """Expressed, not stored: a species that does not express speed does not move, exactly as
        it does not pay for speed (CLAUDE.md §2.5)."""
        world = World(flat_heights())
        speedy = world.place(0.0, 10.0, speed=7.0)

        assert world.movement.top_speed(speedy) == pytest.approx(7.0)

        sessile_id = world.species.register(("size", "insulation"))
        world.genetics.speciate(speedy, sessile_id)

        assert world.movement.top_speed(speedy) == pytest.approx(0.0)


class TestCreaturesStaySurfaceLocked:
    def test_settle_puts_a_freshly_placed_animal_on_the_ground(self):
        world = World(flat_heights(elevation=250.0))
        walker = world.place(10.0, 10.0, speed=5.0)

        assert world.position(walker)[2] == pytest.approx(250.0)

    def test_z_follows_the_terrain_after_a_step(self):
        world = World(ramp_heights(gain_per_unit=10.0))
        walker = world.place(0.0, 10.0, speed=4.0)

        world.step_toward(walker, 40.0, 10.0, pace=1.0)

        x, _, z = world.position(walker)
        assert z == pytest.approx(world.terrain.elevation_at(x, np.float64(10.0)), rel=1e-5)
        assert z > 0.0


class TestWhatAStepCosts:
    """CLAUDE.md §2.5: every one of these is a term of the energy budget, and each is what makes
    some environment or some build cheaper than another without a designer choosing."""

    def _cost_of_one_step(self, world, walker, target_x, target_y, pace):
        before = world.energy(walker).copy()
        world.step_toward(walker, target_x, target_y, pace)
        return float(before[0] - world.energy(walker)[0])

    def test_uphill_costs_more_than_the_same_step_downhill(self):
        """Issue #25's "done when", and the reason elevation is in the model at all (§2.6): if
        relief did not price movement, terrain would be decoration."""
        world = World(ramp_heights(gain_per_unit=10.0))
        climber = world.place(20.0, 10.0, speed=4.0)
        descender = world.place(20.0, 10.0, speed=4.0)

        uphill = self._cost_of_one_step(world, climber, 40.0, 10.0, pace=1.0)
        downhill = self._cost_of_one_step(world, descender, 0.0, 10.0, pace=1.0)

        assert uphill > downhill

    def test_a_steeper_climb_costs_more_than_a_gentle_one(self):
        gentle = World(ramp_heights(gain_per_unit=2.0))
        steep = World(ramp_heights(gain_per_unit=20.0))
        gentle_climber = gentle.place(0.0, 10.0, speed=4.0)
        steep_climber = steep.place(0.0, 10.0, speed=4.0)

        gentle_cost = self._cost_of_one_step(gentle, gentle_climber, 40.0, 10.0, pace=1.0)
        steep_cost = self._cost_of_one_step(steep, steep_climber, 40.0, 10.0, pace=1.0)

        assert steep_cost > gentle_cost

    def test_a_bigger_animal_pays_more_for_the_same_journey(self):
        world = World(flat_heights())
        small = world.place(0.0, 10.0, speed=4.0, size=1.0)
        large = world.place(0.0, 10.0, speed=4.0, size=3.0)

        small_cost = self._cost_of_one_step(world, small, 40.0, 10.0, pace=1.0)
        large_cost = self._cost_of_one_step(world, large, 40.0, 10.0, pace=1.0)

        assert large_cost > small_cost

    def test_sprinting_costs_more_per_metre_than_walking(self):
        """This is §2.5's "effort is charged, not just distance". Cost rising with *distance* alone
        would make a chase merely long, not expensive; it is the per-unit premium that makes a
        predator pay for every chase it loses and prey pay for every escape.
        """
        world = World(flat_heights())
        walker = world.place(0.0, 10.0, speed=5.0)
        sprinter = world.place(0.0, 10.0, speed=5.0)

        walk_cost = self._cost_of_one_step(world, walker, 40.0, 10.0, MOVEMENT_CONFIG.walking_pace)
        sprint_cost = self._cost_of_one_step(world, sprinter, 40.0, 10.0, 1.0)

        walk_distance = float(world.position(walker)[0][0])
        sprint_distance = float(world.position(sprinter)[0][0])

        assert walk_cost / walk_distance < sprint_cost / sprint_distance

    def test_an_unexpressed_size_gene_is_neither_charged_nor_carried(self):
        """Cost follows the expressed phenotype (§2.5), the same rule metabolism obeys."""
        world = World(flat_heights())
        weightless_id = world.species.register(("speed", "insulation"))
        heavy = world.place(0.0, 10.0, speed=4.0, size=3.0)
        weightless = world.place(0.0, 10.0, speed=4.0, size=3.0)
        world.genetics.speciate(weightless, weightless_id)

        heavy_cost = self._cost_of_one_step(world, heavy, 40.0, 10.0, pace=1.0)
        weightless_cost = self._cost_of_one_step(world, weightless, 40.0, 10.0, pace=1.0)

        assert weightless_cost == pytest.approx(0.0)
        assert heavy_cost > 0.0


class TestHungerClosesOffOptions:
    """CLAUDE.md §2.5: "a starving animal can neither run nor hide". The pool is not merely a
    readout that gets low — it gates what the animal is physically able to do."""

    def test_an_empty_pool_leaves_an_animal_unable_to_move(self):
        world = World(flat_heights())
        starved = world.place(10.0, 10.0, speed=5.0, energy=0.0)

        world.step_toward(starved, 40.0, 10.0, pace=1.0)

        x, y, _ = world.position(starved)
        assert (x, y) == pytest.approx((10.0, 10.0))

    def test_a_nearly_empty_pool_covers_only_what_it_can_pay_for(self):
        world = World(flat_heights())
        fed = world.place(0.0, 10.0, speed=5.0, energy=1e6)
        faint = world.place(0.0, 10.0, speed=5.0, energy=1.0)

        world.step_toward(fed, 40.0, 10.0, pace=1.0)
        world.step_toward(faint, 40.0, 10.0, pace=1.0)

        assert 0.0 < world.position(faint)[0] < world.position(fed)[0]

    def test_moving_never_drives_a_pool_negative(self):
        """The floor is `Ecology.spend`'s, and the invariant harness asserts it every tick; this
        pins that movement routes its charge through the owner of `energy` rather than around it.
        """
        world = World(ramp_heights(gain_per_unit=20.0))
        walkers = world.place(
            np.zeros(16), np.full(16, 10.0), speed=5.0, energy=np.float32(0.5)
        )

        for _ in range(20):
            world.step_toward(walkers, 40.0, 10.0, pace=1.0)

        assert (world.energy(walkers) >= 0.0).all()


class TestMovementDoesNotWriteWhatItDoesNotOwn:
    def test_it_cannot_write_the_energy_column_directly(self):
        """`energy` is Ecology's (#17). Movement charges through `spend`, and the registry is what
        makes that a caught error rather than a convention (CLAUDE.md §2.3)."""
        world = World(flat_heights())
        walker = world.place(10.0, 10.0, speed=5.0)

        with pytest.raises(ColumnOwnershipError, match="energy"):
            world.movement.write("energy", walker, np.zeros(len(walker), dtype=np.float32))


class TestAStepOntoTheWorldEdge:
    """The pricing pass and the move must agree on where a step lands (#128).

    `step` snaps to the target on arrival precisely because float error on ``x + unit * distance``
    can put the result a fraction past the boundary, where `Terrain.elevation_at` raises. The
    pricing pass computed the same point *without* snapping, so a target on the world edge raised
    out of the middle of a tick. Nothing caught it because #25's own tests aim at interior targets;
    the first thing to feed a real `Hunger.forage_target` into `step` hit it at once, since
    `Plants.perceive` reports patches on the terrain grid and its outermost column sits at exactly
    `world_width`.
    """

    def test_a_diagonal_step_onto_the_far_corner_is_priced_without_raising(self):
        world = World(flat_heights())
        corner_x = world.terrain.world_width
        corner_y = world.terrain.world_height
        # Reach far exceeding the distance, so the step arrives and the snap is what has to apply.
        walker = world.place(0.0, 0.0, speed=1000.0)

        world.step_toward(walker, corner_x, corner_y, pace=1.0)

        x, y, _z = world.position(walker)
        assert x[0] == pytest.approx(corner_x)
        assert y[0] == pytest.approx(corner_y)

    def test_many_diagonal_steps_onto_the_edge_all_survive(self):
        """One step lands past the boundary only when rounding breaks the wrong way — about 0.13% of
        the time — so a single assertion would pass on a broken build. A cohort spread along the
        edge is what makes the failure certain rather than likely.
        """
        world = World(flat_heights())
        edge_x = world.terrain.world_width
        starts_y = np.linspace(0.0, world.terrain.world_height, 200, dtype=np.float32)
        walkers = world.place(np.zeros(200, dtype=np.float32), starts_y, speed=1000.0)

        for target_y in np.linspace(0.0, world.terrain.world_height, 12):
            world.step_toward(walkers, edge_x, float(target_y), pace=1.0)

        x, _y, _z = world.position(walkers)
        assert (x <= world.terrain.world_width).all()

    def test_a_step_it_cannot_afford_still_lands_inside_the_world(self):
        """The unaffordable path integrates a *fraction* of the step, so it does not arrive and the
        snap does not apply — the landing point has to be in bounds on its own."""
        world = World(flat_heights())
        walker = world.place(0.0, 0.0, speed=1000.0, energy=np.float32(1.0))

        world.step_toward(walker, world.terrain.world_width, world.terrain.world_height, pace=1.0)

        x, y, _z = world.position(walker)
        assert 0.0 <= x[0] <= world.terrain.world_width
        assert 0.0 <= y[0] <= world.terrain.world_height


class TestAStepIsPricedAlongItsWholePath:
    """Issue #113: a step is charged for every cell it crosses, not for its two endpoints.

    Endpoint sampling nets a descent against a climb, so terrain between the ends is invisible and
    an animal crosses it for free. That matters more than it sounds: impassable ground is expressed
    *entirely* through its cost, so a barrier the cost function never sees is not a barrier at all.
    """

    def _cost_of_one_step(self, world, walker, target_x, target_y, pace):
        before = world.energy(walker).copy()
        world.step_toward(walker, target_x, target_y, pace)
        return float(before[0] - world.energy(walker)[0])

    def test_crossing_a_ravine_is_charged_for_the_climb_out(self):
        """The defect, stated directly. Rim to rim is zero net elevation change, so the old rule
        charged the flat rate for a step that descended into a gorge and climbed out of it."""
        world = World(ravine_heights(depth=6.0, floor_x=20.0, wall_width=6.0))
        crosser = world.place(14.0, 20.0, speed=12.0)
        flat_world = World(flat_heights())
        flat_walker = flat_world.place(14.0, 20.0, speed=12.0)

        across = self._cost_of_one_step(world, crosser, 26.0, 20.0, pace=1.0)
        level = self._cost_of_one_step(flat_world, flat_walker, 26.0, 20.0, pace=1.0)

        assert across > level, (
            "a rim-to-rim crossing cost the same as flat ground: the ravine between the "
            "endpoints was never seen"
        )

    def test_crossing_a_ridge_is_charged_for_the_ascent(self):
        world = World(ridge_heights(height=6.0, crest_x=20.0, width=6.0))
        crosser = world.place(14.0, 20.0, speed=12.0)
        flat_world = World(flat_heights())
        flat_walker = flat_world.place(14.0, 20.0, speed=12.0)

        over = self._cost_of_one_step(world, crosser, 26.0, 20.0, pace=1.0)
        level = self._cost_of_one_step(flat_world, flat_walker, 26.0, 20.0, pace=1.0)

        assert over > level

    def test_a_step_costs_what_the_same_ground_covered_in_pieces_costs(self):
        """The property the whole issue reduces to: **path cost is additive under subdivision**.

        One long step over broken ground must cost what the same ground costs walked in short
        steps, because it is the same journey. Endpoint sampling breaks this — the pieces each see
        their own local relief while the whole sees only its ends — and that failure is what lets an
        animal cross a ravine for free by taking one big stride.
        """
        heights = ravine_heights(depth=5.0, floor_x=20.0, wall_width=5.0)
        one_world = World(heights)
        strider = one_world.place(14.0, 20.0, speed=12.0)
        one_world.step_toward(strider, 26.0, 20.0, pace=1.0)
        long_step = float(1e6 - one_world.energy(strider)[0])

        many_world = World(heights)
        walker = many_world.place(14.0, 20.0, speed=1.0)
        for _ in range(12):
            many_world.step_toward(walker, 26.0, 20.0, pace=1.0)
        short_steps = float(1e6 - many_world.energy(walker)[0])

        assert long_step == pytest.approx(short_steps, rel=0.02)

    def test_a_diagonal_crossing_sees_the_ground_it_crosses(self):
        """Diagonals cross more cells than either axis alone, so the enumeration has to visit
        crossings on both."""
        world = World(ravine_heights(depth=5.0, floor_x=20.0, wall_width=5.0))
        crosser = world.place(14.0, 14.0, speed=20.0)
        flat_world = World(flat_heights())
        flat_walker = flat_world.place(14.0, 14.0, speed=20.0)

        across = self._cost_of_one_step(world, crosser, 26.0, 26.0, pace=1.0)
        level = self._cost_of_one_step(flat_world, flat_walker, 26.0, 26.0, pace=1.0)

        assert across > level

    def test_flat_ground_still_costs_exactly_its_distance(self):
        """The haul term must not be double counted by being accumulated per cell: summing a
        linear cost over sub-segments is the same number, and a regression here would show up as
        every animal in every world paying more to walk."""
        world = World(flat_heights())
        walker = world.place(10.0, 10.0, speed=7.0)

        cost = self._cost_of_one_step(world, walker, 17.0, 10.0, pace=1.0)

        expected = 1.0 * MOVEMENT_CONFIG.transport_cost * 7.0 * (
            1.0 + MOVEMENT_CONFIG.exertion_premium * 1.0
        )
        assert cost == pytest.approx(expected, rel=1e-6)

    def test_a_step_inside_one_cell_is_unchanged(self):
        """No crossing to enumerate: the answer must be the plain endpoint arithmetic it always was."""
        world = World(ramp_heights(gain_per_unit=2.0))
        walker = world.place(10.2, 10.0, speed=0.5)

        cost = self._cost_of_one_step(world, walker, 10.7, 10.0, pace=1.0)

        distance = 0.5
        climb = 2.0 * distance
        expected = 1.0 * (
            MOVEMENT_CONFIG.transport_cost
            * distance
            * (1.0 + MOVEMENT_CONFIG.exertion_premium)
            + MOVEMENT_CONFIG.climb_cost * climb
        )
        assert cost == pytest.approx(expected, rel=1e-6)

    def test_an_animal_runs_out_where_the_climb_starts_not_beyond_it(self):
        """The affordability gate wants the integral too: with a cumulative cost along the path,
        "where did it run out" is well defined, and an animal with barely enough energy stops at
        the foot of a wall rather than partway up it in proportion to a flat average."""
        world = World(ridge_heights(height=8.0, crest_x=24.0, width=4.0))
        # Enough for the flat approach and a little of the climb, nowhere near enough for the crest.
        walker = world.place(14.0, 20.0, speed=20.0, energy=6.0)

        world.step_toward(walker, 34.0, 20.0, pace=1.0)
        x, _, _ = world.position(walker)

        assert 14.0 < float(x[0]) < 24.0
        assert world.energy(walker)[0] == pytest.approx(0.0, abs=1e-6)

    def test_no_step_is_charged_more_than_the_pool_holds(self):
        world = World(ravine_heights(depth=9.0, floor_x=20.0, wall_width=4.0))
        walkers = world.place(
            np.full(16, 12.0), np.linspace(8.0, 30.0, 16), speed=16.0, energy=3.0
        )

        world.step_toward(walkers, 28.0, 20.0, pace=1.0)

        assert (world.energy(walkers) >= 0.0).all()
