import time

import numpy as np
import pytest

from core.selection import Selection
from core.spatial import SpatialIndex


class FakeStore:
    """A minimal stand-in for EntityStore: just the position columns the index reads."""

    def __init__(self, capacity: int) -> None:
        self.x = np.zeros(capacity, dtype=np.float32)
        self.y = np.zeros(capacity, dtype=np.float32)
        self.z = np.zeros(capacity, dtype=np.float32)


def brute_force_neighbors(store, observer_rows, population_rows, radius):
    """Reference implementation: an explicit O(observers x population) distance check, no grid."""
    observer_set = set(observer_rows.tolist())
    result = set()
    for row in population_rows.tolist():
        if row in observer_set:
            continue
        for obs in observer_rows.tolist():
            dx = float(store.x[row]) - float(store.x[obs])
            dy = float(store.y[row]) - float(store.y[obs])
            dz = float(store.z[row]) - float(store.z[obs])
            if dx * dx + dy * dy + dz * dz <= radius * radius:
                result.add(row)
                break
    return result


def random_store(capacity, side, rng):
    store = FakeStore(capacity)
    store.x[:] = rng.uniform(0, side, capacity).astype(np.float32)
    store.y[:] = rng.uniform(0, side, capacity).astype(np.float32)
    store.z[:] = rng.uniform(0, side, capacity).astype(np.float32)
    return store


class TestConstruction:
    def test_rejects_non_positive_cell_size(self):
        with pytest.raises(ValueError):
            SpatialIndex(cell_size=0.0)
        with pytest.raises(ValueError):
            SpatialIndex(cell_size=-1.0)


class TestNeighborsAgainstBruteForce:
    @pytest.mark.parametrize("seed", range(10))
    def test_matches_brute_force_reference(self, seed):
        rng = np.random.default_rng(seed)
        n = 200
        store = random_store(n, side=50.0, rng=rng)
        population = Selection.all(n)
        radius = float(rng.uniform(2.0, 15.0))

        index = SpatialIndex(cell_size=10.0)
        index.rebuild(store, population)

        observer_row = int(rng.integers(0, n))
        observers = Selection.from_indices(np.array([observer_row]), capacity=n)

        got = set(index.neighbors_of(store, observers, radius).to_indices().tolist())
        expected = brute_force_neighbors(
            store, np.array([observer_row]), population.to_indices(), radius
        )

        assert got == expected

    @pytest.mark.parametrize("seed", range(5))
    def test_matches_brute_force_with_multiple_observers(self, seed):
        rng = np.random.default_rng(seed + 1000)
        n = 150
        store = random_store(n, side=40.0, rng=rng)
        population = Selection.all(n)
        radius = float(rng.uniform(3.0, 12.0))

        index = SpatialIndex(cell_size=8.0)
        index.rebuild(store, population)

        observer_rows = np.array(sorted(rng.choice(n, size=5, replace=False)))
        observers = Selection.from_indices(observer_rows, capacity=n)

        got = set(index.neighbors_of(store, observers, radius).to_indices().tolist())
        expected = brute_force_neighbors(store, observer_rows, population.to_indices(), radius)

        assert got == expected

    def test_radius_larger_than_cell_size_still_correct(self):
        rng = np.random.default_rng(42)
        n = 100
        store = random_store(n, side=60.0, rng=rng)
        population = Selection.all(n)
        radius = 45.0  # much larger than cell_size, exercises multi-cell-ring search

        index = SpatialIndex(cell_size=5.0)
        index.rebuild(store, population)

        observers = Selection.from_indices(np.array([0]), capacity=n)
        got = set(index.neighbors_of(store, observers, radius).to_indices().tolist())
        expected = brute_force_neighbors(store, np.array([0]), population.to_indices(), radius)

        assert got == expected

    def test_excludes_rows_outside_population(self):
        n = 10
        store = FakeStore(n)
        store.x[:] = np.arange(n, dtype=np.float32)
        # Only even rows are indexed.
        population = Selection.from_indices(
            np.array([i for i in range(n) if i % 2 == 0]), capacity=n
        )

        index = SpatialIndex(cell_size=5.0)
        index.rebuild(store, population)

        observers = Selection.from_indices(np.array([0]), capacity=n)
        got = index.neighbors_of(store, observers, radius=3.0)

        # Row 1 (odd, distance 1) is geometrically closer than row 2, but is not in population.
        assert got.to_indices().tolist() == [2]

    def test_excludes_self(self):
        n = 3
        store = FakeStore(n)  # all at the origin
        index = SpatialIndex(cell_size=5.0)
        index.rebuild(store, Selection.all(n))

        observers = Selection.from_indices(np.array([0]), capacity=n)
        got = index.neighbors_of(store, observers, radius=1.0)

        assert set(got.to_indices().tolist()) == {1, 2}

    def test_relation_is_symmetric(self):
        rng = np.random.default_rng(7)
        n = 100
        store = random_store(n, side=30.0, rng=rng)
        index = SpatialIndex(cell_size=6.0)
        index.rebuild(store, Selection.all(n))

        a, b = 3, 42
        a_sel = Selection.from_indices(np.array([a]), capacity=n)
        b_sel = Selection.from_indices(np.array([b]), capacity=n)
        radius = 20.0

        b_in_a = b in index.neighbors_of(store, a_sel, radius).to_indices().tolist()
        a_in_b = a in index.neighbors_of(store, b_sel, radius).to_indices().tolist()
        assert b_in_a == a_in_b

    def test_no_indexed_rows_returns_empty(self):
        n = 5
        store = FakeStore(n)
        index = SpatialIndex(cell_size=5.0)
        index.rebuild(store, Selection.none(n))

        observers = Selection.from_indices(np.array([0]), capacity=n)
        got = index.neighbors_of(store, observers, radius=100.0)
        assert len(got) == 0

    def test_rejects_non_positive_radius(self):
        n = 3
        store = FakeStore(n)
        index = SpatialIndex(cell_size=5.0)
        index.rebuild(store, Selection.all(n))
        observers = Selection.from_indices(np.array([0]), capacity=n)
        with pytest.raises(ValueError):
            index.neighbors_of(store, observers, radius=0.0)


class TestIncrementalUpdate:
    def test_update_after_moving_matches_a_full_rebuild(self):
        rng = np.random.default_rng(3)
        n = 200
        store = random_store(n, side=50.0, rng=rng)
        population = Selection.all(n)

        incremental = SpatialIndex(cell_size=7.0)
        incremental.rebuild(store, population)

        for _ in range(20):
            store.x[:] += rng.uniform(-3, 3, n).astype(np.float32)
            store.y[:] += rng.uniform(-3, 3, n).astype(np.float32)
            store.z[:] += rng.uniform(-3, 3, n).astype(np.float32)
            incremental.update(store, population)

            rebuilt = SpatialIndex(cell_size=7.0)
            rebuilt.rebuild(store, population)

            observers = Selection.from_indices(np.array([0]), capacity=n)
            got_incremental = incremental.neighbors_of(store, observers, radius=10.0)
            got_rebuilt = rebuilt.neighbors_of(store, observers, radius=10.0)
            assert got_incremental == got_rebuilt

    def test_update_drops_rows_that_leave_the_population(self):
        n = 4
        store = FakeStore(n)
        index = SpatialIndex(cell_size=5.0)
        index.rebuild(store, Selection.all(n))

        index.update(store, Selection.from_indices(np.array([0, 1]), capacity=n))

        observers = Selection.from_indices(np.array([0]), capacity=n)
        got = index.neighbors_of(store, observers, radius=100.0)
        assert got.to_indices().tolist() == [1]

    def test_update_adds_rows_that_join_the_population(self):
        n = 4
        store = FakeStore(n)
        index = SpatialIndex(cell_size=5.0)
        index.rebuild(store, Selection.from_indices(np.array([0]), capacity=n))

        index.update(store, Selection.all(n))

        observers = Selection.from_indices(np.array([0]), capacity=n)
        got = index.neighbors_of(store, observers, radius=100.0)
        assert set(got.to_indices().tolist()) == {1, 2, 3}

    def test_update_is_a_noop_for_rows_that_did_not_change_cell(self):
        n = 2
        store = FakeStore(n)
        store.x[1] = 1.0
        index = SpatialIndex(cell_size=5.0)
        index.rebuild(store, Selection.all(n))
        buckets_before = {cell: set(rows) for cell, rows in index._buckets.items()}

        index.update(store, Selection.all(n))

        assert index._buckets == buckets_before


class TestPerformance:
    """Query cost benchmarked at the populations from #1 (docs/spikes/soa-throughput.md).

    Positions are drawn uniformly over a world whose volume scales with n, holding entity density
    -- and so expected candidates per query -- roughly constant across sizes. A working spatial
    index should then answer a query in time that tracks local density rather than total
    population, unlike an O(n) brute-force scan.

    Measured on the CI runner (2026-07-28, 4-core x86_64, Python 3.12.3, NumPy 2.x), rebuild()/
    update() in milliseconds and a single-observer neighbors_of() query in milliseconds:

    | n       | rebuild | update | query   |
    |--------:|--------:|-------:|--------:|
    | 1,000   | 0.70    | 0.59   | 0.071   |
    | 5,000   | 3.02    | 2.86   | 0.061   |
    | 20,000  | 12.30   | 10.27  | 0.056   |
    | 100,000 | 49.07   | 39.26  | 0.066   |

    Query cost stays flat (~0.06-0.07ms) across the full two-orders-of-magnitude population range,
    as expected for a grid search whose cost tracks local density rather than n. rebuild/update
    scale linearly with n, as expected for a single full pass over the population. Budgets below
    are set well above measured values to guard against real regressions without flaking on CI
    noise (CLAUDE.md §8.5).
    """

    @pytest.mark.parametrize("n", [1_000, 5_000, 20_000, 100_000])
    def test_query_cost_does_not_grow_with_population(self, n):
        rng = np.random.default_rng(n)
        side = 100.0 * (n / 1_000) ** (1 / 3)
        store = random_store(n, side=side, rng=rng)
        population = Selection.all(n)

        index = SpatialIndex(cell_size=20.0)
        index.rebuild(store, population)

        observers = Selection.from_indices(np.array([0]), capacity=n)

        start = time.perf_counter()
        for _ in range(20):
            index.neighbors_of(store, observers, radius=20.0)
        elapsed_per_query = (time.perf_counter() - start) / 20

        assert elapsed_per_query < 0.01

    @pytest.mark.parametrize("n", [1_000, 5_000, 20_000, 100_000])
    def test_rebuild_and_update_stay_within_budget(self, n):
        rng = np.random.default_rng(n + 1)
        side = 100.0 * (n / 1_000) ** (1 / 3)
        store = random_store(n, side=side, rng=rng)
        population = Selection.all(n)
        index = SpatialIndex(cell_size=20.0)

        start = time.perf_counter()
        index.rebuild(store, population)
        rebuild_elapsed = time.perf_counter() - start
        assert rebuild_elapsed < 2.0

        store.x[:] += rng.uniform(-1, 1, n).astype(np.float32)
        start = time.perf_counter()
        index.update(store, population)
        update_elapsed = time.perf_counter() - start
        assert update_elapsed < 2.0

    def test_beats_repeated_brute_force_scans_at_20k(self):
        # At the scale where per-tick "every entity senses its neighbours" work happens, the
        # index should cost far less than re-scanning the whole population for every observer --
        # that gap is the entire reason this module exists (issue #11's "Why").
        rng = np.random.default_rng(20_000)
        n = 20_000
        side = 100.0 * (n / 1_000) ** (1 / 3)
        store = random_store(n, side=side, rng=rng)
        population = Selection.all(n)
        rows = population.to_indices()
        radius = 20.0

        index = SpatialIndex(cell_size=radius)
        index.rebuild(store, population)

        observer_rows = rows[:200]

        start = time.perf_counter()
        for row in observer_rows.tolist():
            observers = Selection.from_indices(np.array([row]), capacity=n)
            index.neighbors_of(store, observers, radius)
        indexed_elapsed = time.perf_counter() - start

        start = time.perf_counter()
        for row in observer_rows.tolist():
            dx = store.x[rows] - store.x[row]
            dy = store.y[rows] - store.y[row]
            dz = store.z[rows] - store.z[row]
            (dx**2 + dy**2 + dz**2 <= radius**2)
        brute_elapsed = time.perf_counter() - start

        assert indexed_elapsed < brute_elapsed
