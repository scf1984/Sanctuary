"""Water lost, water found, water drunk — and the death that falls out of it (#156).

Two groups carry the design rather than the arithmetic. **Nothing kills anybody here**: the last
class asserts that a dry animal dies through `Ecology.starving` and `Death`, which are unchanged,
because a second mortality path is what this mechanic was most at risk of growing. And the column
is a **deficit**, so the first class asserts what a recycled row means — a newborn that inherited a
stranger's thirst would die in its first tick, and nothing else in the world would report it.
"""

import numpy as np
import pytest

from core.ecology.hydration import Hydration, HydrationConfig
from core.selection import Selection
from core.services import ColumnRegistry
from core.world.climate import Climate, ClimateConfig
from core.world.diffusion import DiffusionConfig
from core.world.terrain import Terrain
from core.world.water import Water
from core.entities.store import EntityStore

GRID = 11


def world(loss_rate=0.1, heat_scaling=0.0, drink_rate=0.5, temperature=20.0, lake=True):
    """A flat world with a lake in one corner, or none at all."""
    terrain = Terrain(np.zeros((GRID, GRID), dtype=np.float32), cell_size=1.0)
    climate = Climate(
        terrain,
        ClimateConfig(
            equator_y=0.0, equator_temperature=temperature, latitude_gradient=0.0
        ),
    )
    depth = np.zeros((GRID, GRID), dtype=np.float32)
    if lake:
        depth[1:3, 1:3] = 2.0
    water = Water(
        depth,
        np.zeros((GRID, GRID), dtype=np.int8),
        np.ones((GRID, GRID), dtype=np.float32),
        cell_size=1.0,
    )
    store = EntityStore(initial_capacity=16, n_drives=5, n_genes=1)
    hydration = Hydration(
        store,
        ColumnRegistry(),
        terrain,
        climate,
        water,
        HydrationConfig(
            loss_rate=loss_rate,
            heat_scaling=heat_scaling,
            neutral_temperature=20.0,
            drink_rate=drink_rate,
            reachability=DiffusionConfig(range=4.0, climb_penalty=0.5),
        ),
    )
    return store, hydration


def spawn(store, *positions):
    n = len(positions)
    ids = store.allocate(
        n,
        x=np.array([p[0] for p in positions], dtype=np.float32),
        y=np.array([p[1] for p in positions], dtype=np.float32),
        z=np.zeros(n, dtype=np.float32),
    )
    rows = [store._id_to_row[i] for i in ids.tolist()]
    return Selection.from_indices(np.array(rows, dtype=np.int64), store.capacity)


class TestTheColumnIsADeficit:
    def test_a_new_animal_is_not_thirsty(self):
        store, hydration = world()

        assert hydration.deficit(spawn(store, (5.0, 5.0)))[0] == pytest.approx(0.0)

    def test_a_recycled_row_does_not_inherit_a_strangers_thirst(self):
        """The whole reason the deficit is stored rather than the reserve. Stored as a level, the
        `_CLEARED_ON_ALLOCATE` reset would mean "born completely dry" and every young would die in
        its first tick — with nothing in the world reporting why (§8.7)."""
        store, hydration = world()
        first = spawn(store, (5.0, 5.0))
        for _ in range(10):
            hydration.lose(first)
        assert hydration.deficit(first)[0] > 0.5
        store.release(store.row_ids()[first.to_mask()])

        assert hydration.deficit(spawn(store, (5.0, 5.0)))[0] == pytest.approx(0.0)


class TestWaterIsLost:
    def test_an_animal_dries_out_over_ticks(self):
        store, hydration = world(loss_rate=0.1)
        animal = spawn(store, (5.0, 5.0))

        hydration.lose(animal)
        hydration.lose(animal)

        assert hydration.deficit(animal)[0] == pytest.approx(0.2, rel=1e-5)

    def test_heat_dries_an_animal_faster(self):
        """Where the old thirst score's temperature term went: heat does not make an animal want
        water, it makes an animal lose water, and wanting follows from having lost."""
        cool_store, cool = world(heat_scaling=0.1, temperature=20.0)
        hot_store, hot = world(heat_scaling=0.1, temperature=30.0)

        cool.lose(spawn(cool_store, (5.0, 5.0)))
        hot.lose(spawn(hot_store, (5.0, 5.0)))

        assert hot.store.dehydration[0] == pytest.approx(2.0 * cool.store.dehydration[0], rel=1e-5)

    def test_cold_does_not_hydrate_anybody(self):
        """Floored at zero rather than allowed to go negative: cold air merely fails to dry an
        animal out. Unfloored, a cold world would refill a reserve out of the weather."""
        store, hydration = world(loss_rate=0.1, heat_scaling=0.5, temperature=0.0)
        animal = spawn(store, (5.0, 5.0))

        hydration.lose(animal)

        assert hydration.deficit(animal)[0] == pytest.approx(0.1, rel=1e-5)

    def test_an_animal_cannot_lose_more_water_than_it_has(self):
        """The cap is what keeps the upkeep multiplier bounded: past 1 a long-dry animal would be
        charged arbitrarily rather than merely fatally."""
        store, hydration = world(loss_rate=0.4)
        animal = spawn(store, (5.0, 5.0))

        for _ in range(20):
            hydration.lose(animal)

        assert hydration.deficit(animal)[0] == pytest.approx(1.0)


class TestDrinking:
    def test_standing_at_water_restores_a_rate_not_a_refill(self):
        """A drink takes several ticks, so the water's edge is somewhere an animal has to *stay* —
        which, while lakes cannot shrink (#165), is the only cost a waterhole can charge."""
        store, hydration = world(loss_rate=0.5, drink_rate=0.2)
        animal = spawn(store, (1.5, 1.5))
        hydration.lose(animal)
        hydration.lose(animal)

        hydration.drink(animal)

        assert hydration.deficit(animal)[0] == pytest.approx(0.8, rel=1e-5)

    def test_dry_ground_gives_nothing(self):
        store, hydration = world(loss_rate=0.5)
        animal = spawn(store, (8.0, 8.0))
        hydration.lose(animal)

        hydration.drink(animal)

        assert hydration.deficit(animal)[0] == pytest.approx(0.5, rel=1e-5)

    def test_a_full_animal_cannot_overdrink(self):
        """Floored at zero, because a negative deficit would make a drinking animal *cheaper* to
        run than a watered one — a free lunch reached by standing in a lake."""
        store, hydration = world(drink_rate=0.5)
        animal = spawn(store, (1.5, 1.5))

        hydration.drink(animal)

        assert hydration.deficit(animal)[0] == pytest.approx(0.0)

    def test_only_the_animals_at_water_drink(self):
        store, hydration = world(loss_rate=0.5, drink_rate=0.5)
        both = spawn(store, (1.5, 1.5), (8.0, 8.0))
        hydration.lose(both)

        hydration.drink(both)

        assert hydration.deficit(both) == pytest.approx([0.0, 0.5], rel=1e-5)


class TestWaterAdvertisesItself:
    """Why the field is diffused rather than sampled as a predicate at each candidate heading.

    Candidates sit one `look_ahead` from the animal and lakes are sparse, so a binary "is there
    water here" reading means thirst can only steer when a candidate lands *on* water — which is
    #126 again in a new place, and the exact failure this issue exists to end.
    """

    def test_water_is_detectable_from_off_the_lake(self):
        _, hydration = world()

        assert hydration.reachable_at(np.array([4.0]), np.array([4.0]))[0] > 0.0

    def test_the_reading_falls_off_with_distance(self):
        _, hydration = world()

        near = hydration.reachable_at(np.array([3.0]), np.array([3.0]))[0]
        far = hydration.reachable_at(np.array([9.0]), np.array([9.0]))[0]

        assert near > far

    def test_a_world_with_no_water_advertises_none(self):
        _, hydration = world(lake=False)

        assert hydration.reachable.max() == pytest.approx(0.0)

    def test_a_block_of_candidate_headings_costs_one_call(self):
        """#114 samples `(n_entities, n_options)` at once, mirroring `Plants.forage_at` — one call
        for the whole population's whole option set rather than one per option (§2.3)."""
        _, hydration = world()
        block = np.full((3, 8), 4.0)

        assert hydration.reachable_at(block, block).shape == (3, 8)


class TestDeathFallsOutRatherThanBeingARule:
    """The design decision this mechanic was most at risk of getting wrong.

    There is no dehydration check anywhere and no thirst-specific mortality. A dry animal costs
    more to run, empties its pool, and dies through paths that already existed — the shape §2.5
    settles for senescence, and the reason the repository has one answer to "how does a failing
    body kill its owner".
    """

    def test_hydration_never_touches_the_energy_pool(self):
        """Not one line of this service writes `energy`. If it did, dehydration would be a second
        way to starve rather than a multiplier on the first."""
        store, hydration = world(loss_rate=0.5, drink_rate=0.5)
        animal = spawn(store, (1.5, 1.5))
        store.energy[animal.to_indices()] = 100.0

        hydration.lose(animal)
        hydration.drink(animal)

        assert store.energy[animal.to_indices()[0]] == pytest.approx(100.0)

    def test_nothing_here_can_release_a_row(self):
        """A completely dry animal is still alive as far as this service is concerned. What kills
        it is `Ecology.upkeep` charging more than it holds, several ticks later and elsewhere."""
        store, hydration = world(loss_rate=1.0)
        animal = spawn(store, (8.0, 8.0))

        hydration.lose(animal)

        assert hydration.deficit(animal)[0] == pytest.approx(1.0)
        assert store.alive[animal.to_indices()[0]]


class TestTheConfigRefusesTheDegenerate:
    @pytest.mark.parametrize("rate", [0.0, -0.1, 1.5])
    def test_an_impossible_loss_rate_is_refused(self, rate):
        with pytest.raises(ValueError, match="loss_rate"):
            HydrationConfig(
                loss_rate=rate,
                heat_scaling=0.0,
                neutral_temperature=20.0,
                drink_rate=0.5,
                reachability=DiffusionConfig(range=4.0, climb_penalty=0.5),
            )

    @pytest.mark.parametrize("rate", [0.0, -0.1, 1.5])
    def test_an_impossible_drink_rate_is_refused(self, rate):
        with pytest.raises(ValueError, match="drink_rate"):
            HydrationConfig(
                loss_rate=0.1,
                heat_scaling=0.0,
                neutral_temperature=20.0,
                drink_rate=rate,
                reachability=DiffusionConfig(range=4.0, climb_penalty=0.5),
            )

    def test_a_hydrating_heat_scaling_is_refused(self):
        with pytest.raises(ValueError, match="heat_scaling"):
            HydrationConfig(
                loss_rate=0.1,
                heat_scaling=-1.0,
                neutral_temperature=20.0,
                drink_rate=0.5,
                reachability=DiffusionConfig(range=4.0, climb_penalty=0.5),
            )
