import numpy as np
import pytest

from core.ecology.carrion import Carrion, CarrionConfig
from core.ecology.plants import Plants, PlantsConfig
from core.world.diffusion import DiffusionConfig
from core.entities.store import EntityStore
from core.invariants import (
    Invariant,
    InvariantRegistry,
    InvariantViolation,
    Violation,
    default_registry,
    exported_nutrients_are_held,
    no_alive_entity_has_negative_energy,
    no_alive_entity_occupies_a_free_row,
    no_entity_exceeds_its_top_speed,
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


class TestExportedNutrientsAreHeld:
    """#214: the export ledger means "grazed out and *not yet returned*", and the second half is
    only true if every outstanding unit is somewhere — in a living body or in carrion on the ground.
    `nutrients_are_conserved` cannot see a body that vanished holding energy, because the ledger
    still sums to the same total; this is the check that can.
    """

    def make_world(self, energies=(10.0, 20.0)):
        plants = make_plants()
        carrion = Carrion(plants.terrain, plants, CarrionConfig(decay_rate=0.1))
        store = make_store()
        n = len(energies)
        ids = store.allocate(
            n,
            x=np.full(n, 5.0, dtype=np.float32),
            y=np.full(n, 5.0, dtype=np.float32),
            energy=np.array(energies, dtype=np.float32),
        )
        # Founding is the one way bodies come to be on the ledger without having grazed, exactly
        # as `build_world` records its founders.
        plants.record_founding_stock(float(sum(energies)))
        return plants, carrion, store, ids

    def test_holds_when_every_outstanding_unit_is_in_a_body(self):
        plants, carrion, store, _ = self.make_world()
        assert exported_nutrients_are_held(plants, carrion)(store) is None

    def test_a_body_released_without_returning_its_energy_trips(self):
        """The issue's reproduction: the animal is gone, its energy is still owed, and nothing
        holds it. `nutrients_are_conserved` passes on exactly this state."""
        plants, carrion, store, ids = self.make_world()
        check = exported_nutrients_are_held(plants, carrion)
        assert nutrients_are_conserved(plants)(store) is None

        store.release(ids[1:])

        violation = check(store)
        assert violation is not None
        assert violation.rows.size == 0
        assert "shortfall" in violation.detail
        assert nutrients_are_conserved(plants)(store) is None  # still blind to it

    def test_a_body_whose_energy_is_returned_before_release_does_not_trip(self):
        """What `Cull.apply` does, and the correct way to remove an animal."""
        plants, carrion, store, ids = self.make_world()
        check = exported_nutrients_are_held(plants, carrion)

        row = store.alive.nonzero()[0][1]
        plants.return_nutrients(np.array([5.0]), np.array([5.0]), store.energy[[row]])
        store.release(ids[1:])

        assert check(store) is None

    def test_energy_burned_and_excreted_does_not_trip(self):
        """`Ecology.spend`: the energy is destroyed and its nutrients go back to the soil."""
        plants, carrion, store, _ = self.make_world()
        check = exported_nutrients_are_held(plants, carrion)

        store.energy[0] -= 4.0
        plants.return_nutrients(np.array([5.0]), np.array([5.0]), np.array([4.0]))

        assert check(store) is None

    def test_a_wound_lying_as_carrion_is_still_held(self):
        """`Ecology.kill` then `Carrion.deposit`: flesh off a body and onto the ground is the one
        in-flight term, and it is still owed until it rots."""
        plants, carrion, store, _ = self.make_world()
        check = exported_nutrients_are_held(plants, carrion)

        store.energy[1] -= 15.0
        carrion.deposit(np.array([5.0]), np.array([5.0]), np.array([15.0]))
        assert check(store) is None

        carrion.decompose()  # rots back through the ledger, so still nothing unaccounted
        assert check(store) is None

    def test_a_gestating_young_holds_what_its_parents_gave_it(self):
        """A row with a negative age is alive and unborn (#20); its endowment was moved, not
        burned, so it counts as held exactly like any other body."""
        plants, carrion, store, _ = self.make_world()
        check = exported_nutrients_are_held(plants, carrion)

        store.allocate(1, age=np.array([-5], dtype=np.int64), energy=np.array([6.0], dtype=np.float32))
        store.energy[0] -= 6.0

        assert check(store) is None

    def test_energy_from_nowhere_trips(self):
        plants, carrion, store, _ = self.make_world()
        check = exported_nutrients_are_held(plants, carrion)

        store.energy[0] += 1.0

        assert check(store) is not None

    def test_a_leak_the_float32_tolerance_hid_now_trips(self):
        """Two thousandths of an energy unit against 30 owed is 7e-5 relative, under the 1e-4 this
        check needed while `energy` was float32. In float64 the allowance is rounding-sized
        (`plants.LEDGER_ABSOLUTE_TOLERANCE`, 1e-3 energy units at this world's exchange rate), so
        the same leak is now visible (#240)."""
        plants, carrion, store, _ = self.make_world()
        check = exported_nutrients_are_held(plants, carrion)

        store.energy[0] += 2e-3

        assert check(store) is not None

    def test_a_world_owing_only_rounding_with_nothing_alive_holds(self):
        """A world that dies out is left owing its last float64 rounding with no body to hold it.
        #240's starving worlds ended owing 2.7e-12 energy units. A purely relative tolerance can
        never accept that, because it is relative to zero."""
        plants, carrion, store, ids = self.make_world()
        check = exported_nutrients_are_held(plants, carrion)
        rows = store.alive.nonzero()[0]
        plants.return_nutrients(np.full(2, 5.0), np.full(2, 5.0), store.energy[rows])
        store.release(ids)
        plants.exported_nutrients = 2.7e-12 * plants.config.nutrient_per_biomass

        assert check(store) is None

    def test_a_dead_row_holding_stale_energy_is_not_counted(self):
        """A released row keeps whatever its `energy` cell last held; only the living hold
        anything."""
        plants, carrion, store, ids = self.make_world()
        check = exported_nutrients_are_held(plants, carrion)

        row = store.alive.nonzero()[0][0]
        plants.return_nutrients(np.array([5.0]), np.array([5.0]), store.energy[[row]])
        store.release(ids[:1])

        assert store.energy[row] > 0.0
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

    def test_omits_the_top_speed_bound_when_the_world_has_no_movement_service(self):
        """Optional on the same terms as the plant field: the bound needs a phenotype, so a bare
        store cannot be asked the question at all."""
        registry = default_registry(min_x=0, max_x=10, min_y=0, max_y=10)
        assert "no_entity_exceeds_its_top_speed" not in [i.name for i in registry._invariants]

    def test_registers_the_top_speed_bound_when_given_a_movement_service(self):
        store = make_store()
        store.allocate(1)
        store.velocity_x[0] = 9.0
        registry = default_registry(
            min_x=0, max_x=10, min_y=0, max_y=10, movement=FixedTopSpeed(3.0)
        )

        with pytest.raises(InvariantViolation) as excinfo:
            registry.check_all(store, tick=1)
        assert excinfo.value.invariant_name == "no_entity_exceeds_its_top_speed"

    def test_registers_nutrient_conservation_when_given_a_plant_field(self):
        plants = make_plants()
        store = make_store()
        registry = default_registry(min_x=0, max_x=10, min_y=0, max_y=10, plants=plants)

        registry.check_all(store, tick=1)  # must not raise

        plants.soil_nutrients[5, 5] += 1.0
        with pytest.raises(InvariantViolation) as excinfo:
            registry.check_all(store, tick=2)
        assert excinfo.value.invariant_name == "nutrients_are_conserved"

    def test_omits_the_export_ledger_check_when_the_world_has_no_carrion(self):
        """The check needs every place an outstanding unit can lie. A world without a carrion
        field cannot say where meat is, so it is not asked."""
        registry = default_registry(min_x=0, max_x=10, min_y=0, max_y=10, plants=make_plants())
        assert "exported_nutrients_are_held" not in [i.name for i in registry._invariants]

    def test_registers_the_export_ledger_check_when_given_carrion(self):
        plants = make_plants()
        carrion = Carrion(plants.terrain, plants, CarrionConfig(decay_rate=0.1))
        store = make_store()
        ids = store.allocate(1, energy=np.array([10.0], dtype=np.float32))
        plants.record_founding_stock(10.0)
        registry = default_registry(
            min_x=0, max_x=10, min_y=0, max_y=10, plants=plants, carrion=carrion
        )
        registry.check_all(store, tick=1)  # must not raise

        store.release(ids)
        with pytest.raises(InvariantViolation) as excinfo:
            registry.check_all(store, tick=2)
        assert excinfo.value.invariant_name == "exported_nutrients_are_held"


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


class FixedTopSpeed:
    """Whatever `Movement` a world has, this invariant asks it exactly one question.

    Stubbed rather than assembled because building a real `Movement` needs a genetics stack, a
    terrain, an ecology and a metabolism — none of which this predicate reads. The same argument
    `ConstantDrive` makes in the behaviour tests: exercise the contract, not the collaborators.
    """

    def __init__(self, speed):
        self.speed = float(speed)

    def top_speed(self, selection):
        return np.full(len(selection), self.speed, dtype=np.float64)


class TestNoEntityExceedsItsTopSpeed:
    """§2.5 rejected a speed cap, so the bound is an argument rather than a clamp: velocity is
    written from the displacement that happened, which is never longer than the one intended.
    This is that argument asserted (§6, §8.2) — it would catch a future writer of `velocity_x`
    that set the column from an intention instead of an outcome.
    """

    def moving(self, store, velocity_x, velocity_y=0.0):
        """Allocate `n` rows and set their velocity behind `Movement`'s back.

        The columns are deliberately not seedable — a newborn is at rest and a reused row must not
        inherit what its predecessor was carrying — so writing them directly is both the only way
        to stage this and an exact rehearsal of the bug the invariant exists to catch: something
        other than `Movement.step` putting a number in the column.
        """
        rows = np.arange(len(velocity_x))
        store.allocate(len(velocity_x))
        store.velocity_x[rows] = np.asarray(velocity_x, dtype=np.float32)
        store.velocity_y[rows] = velocity_y

    def test_passes_when_everything_travels_within_its_own_top_speed(self):
        store = make_store()
        self.moving(store, [0.0, 1.5, 3.0])

        assert no_entity_exceeds_its_top_speed(FixedTopSpeed(3.0))(store) is None

    def test_measures_the_whole_velocity_and_not_one_axis(self):
        """A per-axis check would let a diagonal carry `sqrt(2)` times the top speed, which is the
        grid leaking into the physics — the same defect the magnitude cap in `_accelerate` avoids.
        """
        store = make_store()
        self.moving(store, [2.5], velocity_y=2.5)

        violation = no_entity_exceeds_its_top_speed(FixedTopSpeed(3.0))(store)

        assert violation is not None
        assert violation.rows.tolist() == [0]

    def test_flags_only_the_rows_that_are_over(self):
        store = make_store()
        self.moving(store, [1.0, 9.0, 2.0])

        violation = no_entity_exceeds_its_top_speed(FixedTopSpeed(3.0))(store)

        assert violation.rows.tolist() == [1]
        assert "top speed" in violation.detail

    def test_ignores_a_dead_row_that_kept_its_last_velocity(self):
        """`release` leaves the movement columns alone, exactly as it leaves `x` and `y` (#119), so
        a freed row still holds whatever it was carrying when it died."""
        store = make_store()
        self.moving(store, [9.0])
        store.release(store.row_ids()[[0]])

        assert no_entity_exceeds_its_top_speed(FixedTopSpeed(3.0))(store) is None

    def test_an_empty_world_holds_the_invariant(self):
        assert no_entity_exceeds_its_top_speed(FixedTopSpeed(3.0))(make_store()) is None
