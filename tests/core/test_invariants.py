import numpy as np
import pytest

from core.ecology.plants import Plants, PlantsConfig
from core.world.diffusion import DiffusionConfig
from core.entities.store import EntityStore
from core.invariants import (
    Invariant,
    InvariantRegistry,
    InvariantViolation,
    Violation,
    default_registry,
    no_alive_entity_has_negative_energy,
    no_alive_entity_occupies_a_free_row,
    no_entity_leaves_world_bounds,
    nutrients_are_conserved,
)
from core.world.climate import Climate, ClimateConfig
from core.world.terrain import Terrain
from core.world.tick import TickLoop
from core.world.water import Water

PLANTS_CONFIG = PlantsConfig(
    solar_constant=10.0,
    latitude_tilt=0.02,
    min_growth_temperature=0.0,
    optimal_growth_temperature=25.0,
    max_growth_temperature=45.0,
    nutrient_per_biomass=0.1,
    initial_soil_nutrients=100.0,
    senescence_rate=0.05,
    saturation_accumulation=50.0,
    max_rooting_depth=0.5,
    forage_diffusion=DiffusionConfig(range=4.0, climb_penalty=0.5),
)


def make_store(initial_capacity=8, n_drives=2, n_genes=3):
    return EntityStore(initial_capacity=initial_capacity, n_drives=n_drives, n_genes=n_genes)


def make_plants():
    """A plant field over flat, uniformly warm terrain.

    Built here rather than imported from `tests/core/ecology/test_plants.py`: pytest resolves
    test modules by basename (#85), so importing one test module from another is fragile, and
    these tests need nothing from that file's tuning beyond a field that grows.
    """
    terrain = Terrain(np.zeros((11, 11), dtype=np.float32), cell_size=1.0)
    climate = Climate(
        terrain,
        ClimateConfig(equator_y=0.0, equator_temperature=25.0, latitude_gradient=0.0),
    )
    return Plants(terrain, climate, Water.generate(terrain), PLANTS_CONFIG)


def settle(plants, ticks):
    for _ in range(ticks):
        plants.grow()


def holds(_store):
    return None


class TestInvariantRegistry:
    def test_check_all_passes_silently_when_nothing_violates(self):
        store = make_store()
        store.allocate(2)
        registry = InvariantRegistry()
        registry.register("always_ok", holds)
        registry.check_all(store, tick=1)  # must not raise

    def test_check_all_raises_on_first_violating_invariant(self):
        store = make_store()
        registry = InvariantRegistry()
        registry.register("bad", lambda s: Violation("broke", np.array([0], dtype=np.int64)))
        registry.register("never_reached", lambda s: Violation("also broke"))

        with pytest.raises(InvariantViolation) as excinfo:
            registry.check_all(store, tick=7)

        assert excinfo.value.tick == 7
        assert excinfo.value.invariant_name == "bad"
        assert excinfo.value.violation.rows.tolist() == [0]

    def test_violation_message_names_tick_invariant_and_detail(self):
        violation = InvariantViolation(3, "some_invariant", Violation("nutrients drifted by 4.0"))
        message = str(violation)
        assert "3" in message
        assert "some_invariant" in message
        assert "nutrients drifted by 4.0" in message

    def test_register_preserves_order(self):
        registry = InvariantRegistry()
        registry.register("a", holds)
        registry.register("b", holds)
        assert [inv.name for inv in registry._invariants] == ["a", "b"]

    def test_invariant_is_a_plain_named_predicate(self):
        invariant = Invariant("named", holds)
        assert invariant.name == "named"
        assert invariant.check is holds


class TestViolation:
    def test_rows_default_to_empty_for_invariants_over_things_without_rows(self):
        # The #91 case: a field-level breach has a description and no offending entity.
        violation = Violation("the plant field lost 3 nutrient units")
        assert violation.rows.size == 0
        assert violation.rows.dtype == np.int64

    def test_two_violations_do_not_share_a_rows_array(self):
        first = Violation("a")
        second = Violation("b")
        assert first.rows is not second.rows


class TestNoAliveEntityOccupiesAFreeRow:
    def test_passes_for_a_store_used_only_through_allocate_and_release(self):
        store = make_store(initial_capacity=4)
        ids = store.allocate(4)
        store.release(ids[:2])
        assert no_alive_entity_occupies_a_free_row(store) is None

    def test_flags_a_row_left_alive_after_being_freed_behind_the_store_s_back(self):
        store = make_store(initial_capacity=4)
        ids = store.allocate(2)
        row = store._id_to_row[ids[0].item()]
        store.release(np.array([ids[0]]))
        # Simulate a service writing `alive` directly instead of calling release() --
        # exactly the desync this invariant exists to catch.
        store.alive[row] = True

        violation = no_alive_entity_occupies_a_free_row(store)
        assert violation is not None
        assert violation.rows.tolist() == [row]
        assert str(row) in violation.detail


class TestNoAliveEntityHasNegativeEnergy:
    def test_passes_when_all_alive_entities_have_non_negative_energy(self):
        store = make_store()
        store.allocate(2, energy=np.array([0.0, 5.0], dtype=np.float32))
        assert no_alive_entity_has_negative_energy(store) is None

    def test_flags_alive_entities_with_negative_energy(self):
        store = make_store()
        ids = store.allocate(2, energy=np.array([-1.0, 5.0], dtype=np.float32))
        row = store._id_to_row[ids[0].item()]
        violation = no_alive_entity_has_negative_energy(store)
        assert violation is not None
        assert violation.rows.tolist() == [row]

    def test_ignores_negative_energy_on_dead_rows(self):
        store = make_store()
        ids = store.allocate(1, energy=np.array([5.0], dtype=np.float32))
        store.release(ids)
        row = store._id_to_row.get(ids[0].item())
        assert row is None  # released; row's stale energy value is irrelevant now
        # Directly corrupt the freed row's energy -- a dead entity going negative must not fire.
        store.energy[0] = -1.0
        assert no_alive_entity_has_negative_energy(store) is None


class TestNoEntityLeavesWorldBounds:
    def test_passes_when_every_alive_entity_is_within_bounds(self):
        store = make_store()
        store.allocate(
            2,
            x=np.array([0.0, 10.0], dtype=np.float32),
            y=np.array([0.0, 10.0], dtype=np.float32),
        )
        check = no_entity_leaves_world_bounds(min_x=0, max_x=10, min_y=0, max_y=10)
        assert check(store) is None

    def test_flags_alive_entities_outside_each_edge(self):
        store = make_store(initial_capacity=8)
        ids = store.allocate(
            4,
            x=np.array([-1.0, 5.0, 11.0, 5.0], dtype=np.float32),
            y=np.array([5.0, -1.0, 5.0, 11.0], dtype=np.float32),
        )
        rows = [store._id_to_row[i] for i in ids.tolist()]
        check = no_entity_leaves_world_bounds(min_x=0, max_x=10, min_y=0, max_y=10)
        violation = check(store)
        assert violation is not None
        assert sorted(violation.rows.tolist()) == sorted(rows)

    def test_ignores_out_of_bounds_positions_on_dead_rows(self):
        store = make_store()
        ids = store.allocate(1, x=np.array([-100.0], dtype=np.float32))
        store.release(ids)
        check = no_entity_leaves_world_bounds(min_x=0, max_x=10, min_y=0, max_y=10)
        assert check(store) is None


class TestNutrientsAreConserved:
    """The invariant #91 exists to make registrable: a law over grid cells, not over rows."""

    def test_holds_across_growth_and_senescence(self):
        plants = make_plants()
        check = nutrients_are_conserved(plants)
        store = make_store()
        for _ in range(200):
            plants.grow()
            assert check(store) is None

    def test_holds_when_grazing_carries_nutrients_out_of_the_field(self):
        plants = make_plants()
        settle(plants, ticks=50)
        check = nutrients_are_conserved(plants)
        store = make_store()

        plants.graze(np.full(2, 5.0), np.full(2, 5.0), np.ones(2))

        # Grazed nutrients left the soil and the biomass but stayed on the export ledger, so the
        # total is untouched -- this is the closure claim §2.5 makes, now asserted per tick.
        assert plants.exported_nutrients > 0.0
        assert check(store) is None

    def test_flags_nutrients_created_from_nowhere(self):
        plants = make_plants()
        check = nutrients_are_conserved(plants)
        store = make_store()

        plants.soil_nutrients[5, 5] += 1.0

        violation = check(store)
        assert violation is not None
        assert violation.rows.size == 0  # a field breach names no entity
        assert "drift" in violation.detail

    def test_flags_nutrients_destroyed(self):
        plants = make_plants()
        settle(plants, ticks=20)
        check = nutrients_are_conserved(plants)
        store = make_store()

        # Biomass removed without being ledgered -- exactly what a feeding system (#19) that
        # forgot the export ledger would do.
        plants.biomass[5, 5] = 0.0

        assert check(store) is not None

    def test_a_leak_below_the_tolerance_is_not_reported(self):
        plants = make_plants()
        check = nutrients_are_conserved(plants, relative_tolerance=1e-6)
        store = make_store()

        plants.soil_nutrients[5, 5] += plants.total_nutrients() * 1e-9

        assert check(store) is None


class TestDefaultRegistry:
    def test_combines_the_entity_checks(self):
        store = make_store()
        store.allocate(1, x=np.array([-1.0], dtype=np.float32))
        registry = default_registry(min_x=0, max_x=10, min_y=0, max_y=10)

        with pytest.raises(InvariantViolation) as excinfo:
            registry.check_all(store, tick=1)

        assert excinfo.value.invariant_name == "no_entity_leaves_world_bounds"

    def test_passes_for_a_clean_store(self):
        store = make_store()
        store.allocate(1, x=np.array([5.0], dtype=np.float32), y=np.array([5.0], dtype=np.float32))
        registry = default_registry(min_x=0, max_x=10, min_y=0, max_y=10)
        registry.check_all(store, tick=1)  # must not raise

    def test_omits_nutrient_conservation_when_the_world_has_no_plant_field(self):
        registry = default_registry(min_x=0, max_x=10, min_y=0, max_y=10)
        assert "nutrients_are_conserved" not in [inv.name for inv in registry._invariants]

    def test_registers_nutrient_conservation_when_given_a_plant_field(self):
        plants = make_plants()
        store = make_store()
        registry = default_registry(min_x=0, max_x=10, min_y=0, max_y=10, plants=plants)

        registry.check_all(store, tick=1)  # must not raise

        plants.soil_nutrients[5, 5] += 1.0
        with pytest.raises(InvariantViolation) as excinfo:
            registry.check_all(store, tick=2)
        assert excinfo.value.invariant_name == "nutrients_are_conserved"


class TestFieldInvariantsUnderTheTickLoop:
    """#91's actual deliverable: a grid-level law checked after *every* tick by the real loop.

    Before this, `Plants.total_nutrients()` could only be asserted at the end of a test, so a leak
    that opened mid-run and closed again -- or one that only appears during a long offline
    catch-up (§2.4) -- left nothing behind to trip.
    """

    def make_world(self):
        plants = make_plants()
        settle(plants, ticks=50)  # seed before building the invariant: it captures the total now
        store = make_store()
        return plants, store

    def test_five_hundred_ticks_of_growth_and_grazing_never_trip_the_invariant(self):
        plants, store = self.make_world()

        def grow_and_graze():
            plants.grow()
            plants.graze(np.full(2, 5.0), np.array([5.0, 7.0]), np.full(2, 0.5))

        loop = TickLoop(
            store,
            systems=[grow_and_graze],
            invariants=default_registry(0.0, 10.0, 0.0, 10.0, plants=plants),
            debug_checks=True,
        )

        loop.advance(500)

        assert loop.tick_count == 500
        assert plants.exported_nutrients > 0.0  # the run really did move nutrients around

    def test_a_system_that_leaks_nutrients_trips_on_the_tick_it_leaks(self):
        plants, store = self.make_world()

        def grow_and_lose_biomass():
            plants.grow()
            # Biomass deleted without crediting the export ledger -- the mistake #19 and #21 are
            # most likely to make, and previously invisible to the harness.
            plants.biomass[5, 5] = 0.0

        loop = TickLoop(
            store,
            systems=[grow_and_lose_biomass],
            invariants=default_registry(0.0, 10.0, 0.0, 10.0, plants=plants),
            debug_checks=True,
        )

        with pytest.raises(InvariantViolation) as excinfo:
            loop.advance(500)

        assert excinfo.value.invariant_name == "nutrients_are_conserved"
        assert excinfo.value.tick == 1
        assert loop.tick_count == 1
