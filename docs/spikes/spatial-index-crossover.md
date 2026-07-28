# Spike: where the spatial index actually beats a brute-force scan

Tracks issue #79.

## Status: measured

## Why

`tests/core/test_spatial.py::TestPerformance::test_beats_repeated_brute_force_scans_at_20k`
asserted, with no margin:

```python
assert indexed_elapsed < brute_elapsed
```

at n = 20,000. That is a claim about a *crossover point* — the population above which the grid
search costs less than re-scanning everything — and the crossover had never been measured. Per
CLAUDE.md §8.5 a performance claim needs a benchmark, so this spike measures it before the test is
rewritten.

## Reproduction of the failure

On Windows 11 / Python 3.12.10 / NumPy 2.5.1, the failing test behaves differently depending on
what ran before it:

| how it was run | result |
|---|---|
| `pytest` (full suite, 464 tests) | passes |
| `pytest tests/core/test_spatial.py -k brute_force` (alone) | **fails 15 / 15** |

That split is the whole story: the assertion is decided by cache and allocator warmth, not by a
property of the index. Running the test in isolation pays first-touch page faults and dict/set
allocation inside the indexed path that a full-suite run has already amortized.

## Method

`docs/spikes/spatial_index_crossover_bench.py` is throwaway spike code (CLAUDE.md §8.3) — not part
of `core/`, not imported by anything else. It times the same question both ways at a range of
population sizes:

- **indexed** — 200 separate `SpatialIndex.neighbors_of()` calls, one observer each.
- **brute** — 200 vectorized full-population distance passes, exactly the loop the old test used.

Population density is held constant: world side scales as `n**(1/3)`, so a larger `n` is a larger
world at the same crowding rather than a denser one. That is the ecologically meaningful comparison
(CLAUDE.md §2.3 — population is emergent from area × productivity), and it is what lets the indexed
path's cost stay flat while brute force grows.

Each figure is the **median of 7 timed runs after a warmup run**, because the warmup is precisely
what the old test was accidentally measuring.

## Results

Measured on Windows 11 / Python 3.12.10 / NumPy 2.5.1, 200 observers, radius 20.0. Three
independent runs:

| n | indexed (ms) | brute (ms) | speedup run 1 | run 2 | run 3 | candidates/query |
|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 12.4–16.3 | 3.0–3.4 | 0.24x | 0.19x | 0.25x | 38 |
| 5,000 | 13.9–15.6 | 6.5–8.1 | 0.47x | 0.52x | 0.47x | 24 |
| 20,000 | 16.0–26.6 | 19.7–28.5 | **1.11x** | **1.07x** | **1.50x** | 39 |
| 50,000 | 20.7–31.6 | 55.4–84.2 | 2.68x | 2.67x | 2.73x | 32 |
| 100,000 | 23.5–27.8 | 134.6–223.1 | 4.84x | 7.77x | 9.28x | 30 |
| 200,000 | 27.5–47.7 | 447.4–592.7 | 21.19x | 16.28x | 12.00x | 31 |

### Reading

**The crossover is at n ≈ 20,000 — exactly where the old test asserted.** At that size the two
paths are within 7–50% of each other, so which one wins is decided by machine and cache state. The
test was sitting on the knee of the curve and asserting a coin flip.

**Below the crossover the index is genuinely slower**, by 4–5x at n = 1,000. This is not a defect.
A NumPy brute-force pass over 1,000 rows is one fast vectorized operation, while the indexed path
pays Python-level per-observer overhead — dict lookups across the cell neighbourhood, a set union,
a sort, and a `Selection` construction. The index buys asymptotic behaviour, and asymptotic wins
do not apply at small n.

**Brute force is linear in n, as expected**: 1,000 → 200,000 is a 200x population increase and
brute force grows 199x (2.97 ms → 592.69 ms).

**The indexed path is sub-linear, and near-flat**: over the same 200x population increase it grows
about 2.3x (12.4 ms → 28.0 ms). `candidates/query` stays at 24–39 rows across the whole range,
which is the reason — the grid hands the distance filter a constant-size candidate set determined
by local density, never by total population. The residual growth is dict/set size and cache
pressure, not query work.

## What the replacement test asserts

The property the index actually has is *scaling*, not "faster than brute force at any particular
size". The rewritten test therefore measures at 20,000 and 100,000 and asserts two things, both
with the margin this table shows is real:

1. **A required speedup past the crossover** — at n = 100,000, indexed is at least **2x** faster
   than brute force. Measured: 4.84x / 7.77x / 9.28x, so the smallest observed margin is 2.4x
   clear of the threshold.
2. **Divergent scaling** — going from 20,000 to 100,000 (5x population), brute-force cost grows at
   least 3x while indexed cost grows at most 3x. Measured brute growth: 6.8x / 6.4x / 9.3x.
   Measured indexed growth: 1.57x / 0.88x / 1.51x. Both thresholds sit roughly 2x clear of the
   worst observed value.

n = 100,000 is the right size to assert at: it is the design ceiling used throughout CLAUDE.md
§2.3 and the top of the range benchmarked in [`soa-throughput.md`](soa-throughput.md), so it is a
population the game is expected to run at rather than a figure chosen to make the test pass.

The test performs a warmup pass before every timed measurement. Without it the test measures
allocator state, which is what made the original flaky.
