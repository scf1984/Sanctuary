# Spike: SoA throughput and the catch-up budget

Tracks issue #1.

## Status: measured

## Why

Every offline-advancement decision in `CLAUDE.md` §2.1 and §2.4 rests on an estimate of
~10⁷ entity-updates/sec from a NumPy structure-of-arrays core. That number has never been
measured. This spike measures it, and checks what it implies for closing a 7-day absence.

## Method

`docs/spikes/soa_throughput_bench.py` is throwaway spike code (CLAUDE.md §8.3) — not part of
`core/`, not to be imported by anything else. It builds one representative tick that touches every
entity the way the real core will (CLAUDE.md §2.3):

1. position integration (`x += vx * dt`, clamped to world bounds)
2. a spatial-hash neighbour lookup (grid-cell hashing, standing in for `InteractionGrid`)
3. an energy upkeep decrement, scaled by local crowding
4. a threshold comparison (`energy <= 0`, i.e. starving)
5. a masked selection applying the consequence (refeed, standing in for the free-list
   replacement a real tick performs)

The same five steps are implemented twice: once over global NumPy arrays (structure-of-arrays),
once over a plain Python list of `__slots__` objects with a per-tick `dict`-based neighbour grid.
Both are benchmarked at **1,000 / 5,000 / 20,000 / 100,000** rows, with a warmup period before
timing to avoid first-touch page-fault skew. A doubling-copy (`np.concatenate`) is timed
separately at each size, representing the array-growth mitigation in CLAUDE.md §2.3 item 1.

7-day catch-up wall-clock is derived from the measured SoA per-tick time: at 1 tick = 1 sim-minute
(CLAUDE.md §2.1), a 7-day absence owes `7 × 24 × 60 = 10,080` ticks, so
`wall_clock_seconds = 10,080 × seconds_per_soa_tick`.

## Results

Measured on the CI runner: 4-core x86_64, Python 3.12.3, NumPy 2.5.1. Two independent runs agreed
within a few percent at every size; the table below is the first run.

| n | SoA updates/s | Python updates/s | SoA/Python ratio | growth copy (ms) | 7-day catch-up (s) |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 6,931,331 | 819,685 | 8.5 | 0.01 | 1.45 |
| 5,000 | 11,629,380 | 806,031 | 14.4 | 0.04 | 4.33 |
| 20,000 | 9,187,577 | 844,882 | 10.9 | 0.53 | 21.94 |
| 100,000 | 9,308,281 | 850,611 | 10.9 | 2.53 | 108.29 |

The 5,000-row SoA figure is the high point of the curve rather than the 1,000-row one, which is
consistent with the SoA cost being dominated by fixed per-call NumPy overhead at small n (that
overhead amortizes better at 5,000 than at 1,000) until true O(n) costs — the neighbour-lookup sort
in particular — start to dominate at 20,000+. Growth-copy cost scales linearly with n, as expected
for a `np.concatenate`, and stays under 3 ms even at 100,000 rows — negligible next to a tick.

## Recommendation

**Confirm the `CLAUDE.md` §2.1 ratio table as-is.** Measured SoA throughput is 6.9M–11.6M
entity-updates/sec across the tested range, converging to ~9.3M/sec at 100,000 rows — the same
order of magnitude as the ~10⁷/sec estimate the tick ratios were built on. At that population, a
7-day (10,080-tick) absence takes ~108 seconds of wall-clock to simulate, and every smaller
population tested resolves in a few seconds or less. That is comfortably inside any offline
catch-up budget the game will plausibly need (§2.4), with roughly two orders of magnitude of
headroom before the SoA rate would need to become a design concern.

The SoA-over-Python-objects speedup (8.5–14.4x) is unrelated to the §2.1 feeding-event ratio
(10² vs. reality's 10³–10⁴) — that ratio is a deliberate clock-compression choice, not a throughput
figure, and this spike does not bear on it.

No change to `CLAUDE.md` §2.1 is needed. This report is linked from that section.
