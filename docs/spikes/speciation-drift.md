# Spike: drift rate and the speciation threshold

Tracks issue #16.

## Status: measured

## Why

`CLAUDE.md` §2.5 says isolated populations accumulate genetic distance until they can no longer
interbreed, and issue #16's "done when" asks for a statistical test proving an isolated population
reliably speciates while a mixed one does not. That test needs two numbers — a distance threshold
and a generation count — and picking them by eye would make the test either vacuous (threshold so
low the control also crosses it) or flaky (threshold sitting inside the isolated arm's spread).
This spike measures the distributions both arms actually produce so the constants cite a
measurement (`CLAUDE.md` §8.5).

It also answers a design question the mechanic depends on: **does drift under
`core.genetics.inheritance` separate populations at all?** Nothing forces it to. The inheritance
rule draws offspring from a Gaussian whose spread is the parents' own disagreement, so a closed
population's variance decays as it converges — if it decayed fast enough, both halves would freeze
before they had moved apart, and speciation by isolation would never happen without a mutation
floor.

## Method

`docs/spikes/speciation_drift_spike.py`. Throwaway spike code (`CLAUDE.md` §8.3) — not part of
`core/`, not imported by anything. Unlike `soa_throughput_bench.py` it imports the real
`inherit_genes`, because the quantity being measured *is* that function's drift behaviour.

Two arms share founders, population size, gene count, and generation count. The only difference is
whether genes cross the boundary between the two halves:

- **isolated** — two sub-populations of 40, each breeding only within itself.
- **mixed** (control) — the same 80 creatures breeding as one pool, then split into the same two
  arbitrary halves for measurement. This is the "would any two subsets of one population look
  diverged anyway?" baseline.

Each generation, every creature is replaced by an offspring of two parents drawn at random with
replacement from its own pool. Distance is Euclidean between the two halves' mean gene vectors —
the same quantity `core.genetics.distance.centroid_between` computes. 200 seeds; 6 genes; founders
drawn uniformly from [0.5, 1.5]; `inherit_gain` 1.05.

## Results

Python 3.12.10, NumPy 2.5.1. Distances are in expressed-phenotype units.

| Generation | Isolated: min / p5 / median / p95 | Mixed: median / p95 / max |
|---|---|---|
| 20 | 0.076 / 0.170 / 0.307 / 0.470 | 0.031 / 0.044 / 0.055 |
| 50 | 0.097 / 0.195 / 0.344 / 0.524 | 0.026 / 0.041 / 0.050 |
| 100 | 0.155 / 0.248 / 0.401 / 0.609 | 0.026 / 0.044 / 0.060 |

Share of the 200 runs in which each arm crossed a candidate threshold:

| Generation | Threshold | Isolated | Mixed |
|---|---|---|---|
| 20 | 0.12 | 98.5% | 0.0% |
| 50 | 0.10 | 99.5% | 0.0% |
| 50 | **0.12** | **99.5%** | **0.0%** |
| 50 | 0.15 | 99.0% | 0.0% |
| 100 | 0.12 | 100.0% | 0.0% |

## What this settles

- **Drift does separate isolated populations, and it does so fast.** Most of the separation is in
  place by generation 20; the isolated median then creeps from 0.307 to 0.401 over the next 80
  generations rather than growing like a free random walk. That is the variance decay above: each
  pool converges internally, so the gap between the two pools is largely set early and then holds.
  Speciation by isolation therefore works at this population size without a mutation floor — but
  the *rate* is front-loaded, which is worth remembering when tuning how long a fence must stand
  before it produces a new species.
- **The control never crosses.** Across 200 seeds and all three checkpoints, two arbitrary halves
  of one interbreeding pool never exceeded 0.060. Divergence is caused by the isolation, not by
  the measurement.
- **Chosen constants: 50 generations, threshold 0.12** — used by
  `tests/core/genetics/test_speciation.py`. The gap between the isolated 5th percentile (0.195) and
  the mixed maximum (0.050) is a factor of ~4, and 0.12 sits roughly in the middle on a log scale:
  ~1.6x above the highest control run and ~1.6x below the 5th percentile of the isolated arm. The
  test asserts 36/40 seeds rather than 40/40, leaving room for the one-in-two-hundred isolated run
  that stalls.

These constants are the test's, not the core's. `has_diverged` and `interbreeding_probability` take
the threshold as a parameter; nothing in `core/` hardcodes 0.12. A world with a different gene
count, population size, or `inherit_gain` will need its own number, and re-running this spike is
how to get it.
