import numpy as np
import pytest

from core.entities.store import EntityStore
from core.invariants import (
    Invariant,
    InvariantRegistry,
    InvariantViolation,
    default_registry,
    no_alive_entity_has_negative_energy,
    no_alive_entity_occupies_a_free_row,
    no_entity_leaves_world_bounds,
)


def make_store(initial_capacity=8, n_drives=2, n_genes=3):
    return EntityStore(initial_capacity=initial_capacity, n_drives=n_drives, n_genes=n_genes)


class TestInvariantRegistry:
    def test_check_all_passes_silently_when_nothing_violates(self):
        store = make_store()
        store.allocate(2)
        registry = InvariantRegistry()
        registry.register("always_ok", lambda s: np.empty(0, dtype=np.int64))
        registry.check_all(store, tick=1)  # must not raise

    def test_check_all_raises_on_first_violating_invariant(self):
        store = make_store()
        registry = InvariantRegistry()
        registry.register("bad", lambda s: np.array([0], dtype=np.int64))
        registry.register("never_reached", lambda s: np.array([1], dtype=np.int64))

        with pytest.raises(InvariantViolation) as excinfo:
            registry.check_all(store, tick=7)

        assert excinfo.value.tick == 7
        assert excinfo.value.invariant_name == "bad"
        assert excinfo.value.offending_rows.tolist() == [0]

    def test_violation_message_names_tick_invariant_and_rows(self):
        violation = InvariantViolation(3, "some_invariant", np.array([2, 5], dtype=np.int64))
        message = str(violation)
        assert "3" in message
        assert "some_invariant" in message
        assert "[2, 5]" in message

    def test_register_preserves_order(self):
        registry = InvariantRegistry()
        registry.register("a", lambda s: np.empty(0, dtype=np.int64))
        registry.register("b", lambda s: np.empty(0, dtype=np.int64))
        assert [inv.name for inv in registry._invariants] == ["a", "b"]

    def test_invariant_is_a_plain_named_predicate(self):
        predicate = lambda s: np.empty(0, dtype=np.int64)  # noqa: E731
        invariant = Invariant("named", predicate)
        assert invariant.name == "named"
        assert invariant.check is predicate


class TestNoAliveEntityOccupiesAFreeRow:
    def test_passes_for_a_store_used_only_through_allocate_and_release(self):
        store = make_store(initial_capacity=4)
        ids = store.allocate(4)
        store.release(ids[:2])
        assert no_alive_entity_occupies_a_free_row(store).size == 0

    def test_flags_a_row_left_alive_after_being_freed_behind_the_store_s_back(self):
        store = make_store(initial_capacity=4)
        ids = store.allocate(2)
        row = store._id_to_row[ids[0].item()]
        store.release(np.array([ids[0]]))
        # Simulate a service writing `alive` directly instead of calling release() --
        # exactly the desync this invariant exists to catch.
        store.alive[row] = True

        offending = no_alive_entity_occupies_a_free_row(store)
        assert offending.tolist() == [row]


class TestNoAliveEntityHasNegativeEnergy:
    def test_passes_when_all_alive_entities_have_non_negative_energy(self):
        store = make_store()
        store.allocate(2, energy=np.array([0.0, 5.0], dtype=np.float32))
        assert no_alive_entity_has_negative_energy(store).size == 0

    def test_flags_alive_entities_with_negative_energy(self):
        store = make_store()
        ids = store.allocate(2, energy=np.array([-1.0, 5.0], dtype=np.float32))
        row = store._id_to_row[ids[0].item()]
        assert no_alive_entity_has_negative_energy(store).tolist() == [row]

    def test_ignores_negative_energy_on_dead_rows(self):
        store = make_store()
        ids = store.allocate(1, energy=np.array([5.0], dtype=np.float32))
        store.release(ids)
        row = store._id_to_row.get(ids[0].item())
        assert row is None  # released; row's stale energy value is irrelevant now
        # Directly corrupt the freed row's energy -- a dead entity going negative must not fire.
        store.energy[0] = -1.0
        assert no_alive_entity_has_negative_energy(store).size == 0


class TestNoEntityLeavesWorldBounds:
    def test_passes_when_every_alive_entity_is_within_bounds(self):
        store = make_store()
        store.allocate(2, x=np.array([0.0, 10.0], dtype=np.float32), y=np.array([0.0, 10.0], dtype=np.float32))
        check = no_entity_leaves_world_bounds(min_x=0, max_x=10, min_y=0, max_y=10)
        assert check(store).size == 0

    def test_flags_alive_entities_outside_each_edge(self):
        store = make_store(initial_capacity=8)
        ids = store.allocate(
            4,
            x=np.array([-1.0, 5.0, 11.0, 5.0], dtype=np.float32),
            y=np.array([5.0, -1.0, 5.0, 11.0], dtype=np.float32),
        )
        rows = [store._id_to_row[i] for i in ids.tolist()]
        check = no_entity_leaves_world_bounds(min_x=0, max_x=10, min_y=0, max_y=10)
        assert sorted(check(store).tolist()) == sorted(rows)

    def test_ignores_out_of_bounds_positions_on_dead_rows(self):
        store = make_store()
        ids = store.allocate(1, x=np.array([-100.0], dtype=np.float32))
        store.release(ids)
        check = no_entity_leaves_world_bounds(min_x=0, max_x=10, min_y=0, max_y=10)
        assert check(store).size == 0


class TestDefaultRegistry:
    def test_combines_all_three_checks(self):
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
