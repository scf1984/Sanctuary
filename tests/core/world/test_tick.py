import time

import numpy as np
import pytest

from core.entities.store import EntityStore
from core.invariants import (
    InvariantRegistry,
    InvariantViolation,
    no_alive_entity_has_negative_energy,
)
from core.world.tick import TickLoop


def make_store(initial_capacity=4, n_drives=2, n_genes=3):
    return EntityStore(initial_capacity=initial_capacity, n_drives=n_drives, n_genes=n_genes)


class TestSystemOrder:
    def test_systems_run_in_registration_order(self):
        store = make_store()
        calls = []
        loop = TickLoop(
            store,
            systems=[lambda: calls.append("a"), lambda: calls.append("b"), lambda: calls.append("c")],
        )
        loop.advance(1)
        assert calls == ["a", "b", "c"]

    def test_systems_attribute_reflects_constructor_order(self):
        store = make_store()

        def first():
            pass

        def second():
            pass

        loop = TickLoop(store, systems=[second, first])
        assert loop.systems == (second, first)

    def test_each_system_runs_once_per_tick(self):
        store = make_store()
        counts = {"n": 0}
        loop = TickLoop(store, systems=[lambda: counts.__setitem__("n", counts["n"] + 1)])
        loop.advance(5)
        assert counts["n"] == 5


class TestAdvance:
    def test_rejects_negative_n_ticks(self):
        store = make_store()
        loop = TickLoop(store, systems=[])
        with pytest.raises(ValueError):
            loop.advance(-1)

    def test_zero_ticks_is_a_noop_for_the_counter(self):
        store = make_store()
        loop = TickLoop(store, systems=[])
        loop.advance(0)
        assert loop.tick_count == 0

    def test_batched_advance_matches_stepwise_advance(self):
        # CLAUDE.md §2.1: offline catch-up must be the same simulation as live play, not a
        # second code path — so batching ticks into one call must not change how many ticks
        # or how many system invocations occurred.
        batched_store = make_store()
        batched_count = {"n": 0}
        batched_loop = TickLoop(
            batched_store, systems=[lambda: batched_count.__setitem__("n", batched_count["n"] + 1)]
        )
        batched_loop.advance(1000)

        stepwise_store = make_store()
        stepwise_count = {"n": 0}
        stepwise_loop = TickLoop(
            stepwise_store, systems=[lambda: stepwise_count.__setitem__("n", stepwise_count["n"] + 1)]
        )
        for _ in range(1000):
            stepwise_loop.advance(1)

        assert batched_loop.tick_count == stepwise_loop.tick_count == 1000
        assert batched_count["n"] == stepwise_count["n"] == 1000

    def test_tick_count_accumulates_across_calls(self):
        store = make_store()
        loop = TickLoop(store, systems=[])
        loop.advance(3)
        loop.advance(4)
        assert loop.tick_count == 7


class TestInterpolationState:
    def test_initial_snapshots_match_store_at_construction(self):
        store = make_store()
        store.allocate(2, x=np.array([1.0, 2.0], dtype=np.float32))
        loop = TickLoop(store, systems=[])
        assert loop.previous_positions[0].tolist() == loop.current_positions[0].tolist()
        assert loop.current_positions[0].tolist()[:2] == [1.0, 2.0]

    def test_advance_moves_current_forward_and_previous_holds_prior_state(self):
        store = make_store()
        ids = store.allocate(1, x=np.array([0.0], dtype=np.float32))
        row = store._id_to_row[ids[0].item()]

        def move():
            store.x[row] += 1.0

        loop = TickLoop(store, systems=[move])
        loop.advance(1)
        assert loop.previous_positions[0][row] == 0.0
        assert loop.current_positions[0][row] == 1.0

        loop.advance(1)
        assert loop.previous_positions[0][row] == 1.0
        assert loop.current_positions[0][row] == 2.0

    def test_snapshots_are_copies_not_live_views(self):
        store = make_store()
        ids = store.allocate(1, x=np.array([5.0], dtype=np.float32))
        row = store._id_to_row[ids[0].item()]
        loop = TickLoop(store, systems=[])
        snapshot = loop.current_positions[0]

        store.x[:] = -1.0

        assert snapshot[row] == 5.0
        assert loop.current_positions[0] is snapshot


class TestOccupancySnapshots:
    """Occupancy is snapshotted with the positions, because a position alone cannot be read (#119).

    `release` leaves x/y/z untouched, so a coordinate says nothing about whether a live entity is
    standing there — or whether the one standing there now is the one that was there last tick.
    """

    def test_free_rows_read_minus_one_at_construction(self):
        store = make_store(initial_capacity=4)
        store.allocate(2)
        loop = TickLoop(store, systems=[])
        assert (loop.current_row_ids >= 0).sum() == 2
        assert loop.previous_row_ids.tolist() == loop.current_row_ids.tolist()

    def test_a_death_during_advance_shows_in_current_but_not_previous(self):
        store = make_store(initial_capacity=4)
        ids = store.allocate(2)
        row = store._id_to_row[ids[0].item()]
        loop = TickLoop(store, systems=[lambda: store.release(ids[:1])])

        loop.advance(1)

        assert loop.previous_row_ids[row] == ids[0]
        assert loop.current_row_ids[row] == -1

    def test_a_birth_during_advance_shows_in_current_but_not_previous(self):
        store = make_store(initial_capacity=2)
        store.allocate(1)
        births: list[np.ndarray] = []
        loop = TickLoop(store, systems=[lambda: births.append(store.allocate(1))])

        loop.advance(1)

        row = store._id_to_row[births[0][0].item()]
        assert loop.previous_row_ids[row] == -1
        assert loop.current_row_ids[row] == births[0][0]

    def test_a_row_recycled_within_one_advance_reports_both_occupants(self):
        """The case `alive` cannot see: the row is occupied at both ends, by different entities.

        §2.1 puts death before reproduction inside a single tick precisely so freed rows are
        reusable immediately, so this is the ordinary path once #20 and #21 land, not an edge case.
        """
        store = make_store(initial_capacity=1)
        first = store.allocate(1)
        row = store._id_to_row[first[0].item()]
        second: list[np.ndarray] = []

        def die_then_breed():
            store.release(first)
            second.append(store.allocate(1))

        loop = TickLoop(store, systems=[die_then_breed])
        loop.advance(1)

        assert store.alive[row], "the row is occupied at both ends of the interval"
        assert loop.previous_row_ids[row] == first[0]
        assert loop.current_row_ids[row] == second[0][0]

    def test_occupancy_snapshots_are_copies_not_live_views(self):
        store = make_store(initial_capacity=2)
        ids = store.allocate(2)
        loop = TickLoop(store, systems=[])
        snapshot = loop.current_row_ids

        store.release(ids)

        assert (store.row_ids() == -1).all(), "the store itself has forgotten them"
        assert (snapshot >= 0).all(), "the snapshot still remembers who was there"
        assert loop.current_row_ids is snapshot


class TestDebugChecks:
    def test_disabled_by_default_even_with_a_broken_system(self):
        store = make_store()
        ids = store.allocate(1, energy=np.array([0.0], dtype=np.float32))
        row = store._id_to_row[ids[0].item()]

        def create_energy_from_nowhere():
            store.energy[row] -= 1.0

        loop = TickLoop(store, systems=[create_energy_from_nowhere])
        loop.advance(3)  # must not raise: debug_checks is off
        assert store.energy[row] == -3.0

    def test_debug_checks_requires_a_registry(self):
        store = make_store()
        with pytest.raises(ValueError):
            TickLoop(store, systems=[], debug_checks=True)

    def test_a_deliberately_broken_system_trips_a_registered_invariant(self):
        store = make_store()
        ids = store.allocate(1, energy=np.array([0.0], dtype=np.float32))
        row = store._id_to_row[ids[0].item()]

        def create_energy_from_nowhere():
            store.energy[row] -= 1.0

        registry = InvariantRegistry()
        registry.register(
            "no_alive_entity_has_negative_energy", no_alive_entity_has_negative_energy
        )
        loop = TickLoop(
            store, systems=[create_energy_from_nowhere], invariants=registry, debug_checks=True
        )

        with pytest.raises(InvariantViolation) as excinfo:
            loop.advance(5)

        # Fails on the first tick the broken system corrupts, not after running all 5.
        assert excinfo.value.tick == 1
        assert loop.tick_count == 1
        assert excinfo.value.violation.rows.tolist() == [row]

    def test_a_correct_system_never_trips_the_invariants(self):
        store = make_store()
        store.allocate(1, energy=np.array([10.0], dtype=np.float32))
        registry = InvariantRegistry()
        registry.register(
            "no_alive_entity_has_negative_energy", no_alive_entity_has_negative_energy
        )
        loop = TickLoop(store, systems=[], invariants=registry, debug_checks=True)
        loop.advance(10)  # must not raise
        assert loop.tick_count == 10

    def test_disabled_overhead_is_negligible_next_to_enabled(self):
        # Same store size and tick count on both sides; the only difference is debug_checks.
        # Disabled must stay a small fraction of enabled's cost, since enabled repeats a
        # 10,000-row vectorized predicate every tick and disabled does only the boolean branch.
        n_entities = 10_000
        n_ticks = 200

        def build_loop(**kwargs):
            store = make_store(initial_capacity=n_entities)
            store.allocate(n_entities, energy=np.ones(n_entities, dtype=np.float32))
            return TickLoop(store, systems=[], **kwargs)

        loop_off = build_loop()
        start = time.perf_counter()
        loop_off.advance(n_ticks)
        off_elapsed = time.perf_counter() - start

        registry = InvariantRegistry()
        registry.register(
            "no_alive_entity_has_negative_energy", no_alive_entity_has_negative_energy
        )
        loop_on = build_loop(invariants=registry, debug_checks=True)
        start = time.perf_counter()
        loop_on.advance(n_ticks)
        on_elapsed = time.perf_counter() - start

        assert off_elapsed < on_elapsed
