"""Benchmark harness for issue #79: find where SpatialIndex actually beats a brute-force scan.

Throwaway spike code -- not part of the simulation core (CLAUDE.md 8.3). It exists to produce the
numbers docs/spikes/spatial-index-crossover.md is built from, and should not be imported by
anything else.

The test this replaces asserted `indexed_elapsed < brute_elapsed` at n=20,000 with no margin. That
is a claim about a *crossover point*, and nobody had measured where the crossover is. This measures
it.

Both paths answer the same question -- "for each of 200 observers, which entities lie within
radius?" -- at a range of population sizes:

  indexed: 200 separate SpatialIndex.neighbors_of() calls, one observer each. Cost is dominated by
      Python-level work per call (dict lookups over the cell neighbourhood, a set union, a sort,
      a Selection construction) and is expected to be flat in n, because the number of candidate
      rows in the searched cells depends on local density, not on total population.

  brute:   200 vectorized full-population distance passes, matching the reference loop in the test.
      This is a single fast NumPy pass over n rows, expected to grow linearly in n.

Population density is held constant across sizes -- world side scales as n**(1/3) -- so a larger n
means a larger world at the same crowding, not a denser one. That is the ecologically meaningful
comparison (CLAUDE.md 2.3: population is emergent from area x productivity), and it is what makes
the indexed path's cost flat rather than growing.

Timing method: each measurement is the median of REPEATS timed runs after a warmup run. Warmup
matters more than usual here -- the indexed path allocates a large dict/set structure on first
touch, and issue #79's failure reproduced 15/15 in isolation but passed inside the full suite,
which is exactly a cold-vs-warm artifact.

Usage:
    python docs/spikes/spatial_index_crossover_bench.py
"""

from __future__ import annotations

import statistics
import time

import numpy as np

from core.selection import Selection
from core.spatial import SpatialIndex

SIZES = (1_000, 5_000, 20_000, 50_000, 100_000, 200_000)
OBSERVERS = 200
RADIUS = 20.0
REPEATS = 7
DENSITY_REFERENCE_SIDE = 100.0  # side length at n=1,000; scaled as n**(1/3) to hold density fixed


class FakeStore:
    """Just the position columns SpatialIndex reads, matching tests/core/test_spatial.py."""

    __slots__ = ("x", "y", "z")

    def __init__(self, capacity: int) -> None:
        self.x = np.zeros(capacity, dtype=np.float32)
        self.y = np.zeros(capacity, dtype=np.float32)
        self.z = np.zeros(capacity, dtype=np.float32)


def random_store(capacity: int, side: float, rng: np.random.Generator) -> FakeStore:
    store = FakeStore(capacity)
    store.x[:] = rng.uniform(0, side, capacity).astype(np.float32)
    store.y[:] = rng.uniform(0, side, capacity).astype(np.float32)
    store.z[:] = rng.uniform(0, side, capacity).astype(np.float32)
    return store


def time_indexed(index, store, observer_rows, n) -> float:
    start = time.perf_counter()
    for row in observer_rows:
        observers = Selection.from_indices(np.array([row]), capacity=n)
        index.neighbors_of(store, observers, RADIUS)
    return time.perf_counter() - start


def time_brute(store, observer_rows, rows) -> float:
    start = time.perf_counter()
    for row in observer_rows:
        dx = store.x[rows] - store.x[row]
        dy = store.y[rows] - store.y[row]
        dz = store.z[rows] - store.z[row]
        (dx**2 + dy**2 + dz**2 <= RADIUS**2)
    return time.perf_counter() - start


def median_of(fn, *args) -> float:
    fn(*args)  # warmup: first touch pays page faults and structure allocation, see module docstring
    return statistics.median(fn(*args) for _ in range(REPEATS))


def main() -> None:
    print(f"observers={OBSERVERS} radius={RADIUS} repeats={REPEATS} (median reported)")
    print()
    print("| n | indexed (ms) | brute (ms) | speedup | candidates/query |")
    print("|---:|---:|---:|---:|---:|")

    for n in SIZES:
        rng = np.random.default_rng(n)
        side = DENSITY_REFERENCE_SIDE * (n / 1_000) ** (1 / 3)
        store = random_store(n, side=side, rng=rng)
        population = Selection.all(n)
        rows = population.to_indices()
        observer_rows = rows[:OBSERVERS].tolist()

        index = SpatialIndex(cell_size=RADIUS)
        index.rebuild(store, population)

        indexed = median_of(time_indexed, index, store, observer_rows, n)
        brute = median_of(time_brute, store, observer_rows, rows)

        # How many rows the grid actually hands to the distance filter per query -- the number that
        # explains why the indexed path is flat in n, so record it alongside the timings.
        sample = Selection.from_indices(np.array([observer_rows[0]]), capacity=n)
        candidates = len(index.neighbors_of(store, sample, RADIUS).to_indices())

        print(
            f"| {n:,} | {indexed * 1e3:.2f} | {brute * 1e3:.2f} | "
            f"{brute / indexed:.2f}x | {candidates} |"
        )


if __name__ == "__main__":
    main()
