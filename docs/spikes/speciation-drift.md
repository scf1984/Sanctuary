# Spike: drift rate and the speciation threshold

Tracks issue #16. **Re-measured for #104**, which replaced all three parts of the inheritance rule
this spike measures; the original figures are in git history and no longer describe the code.

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
`core.genetics.inheritance` separate populations at all, and does it keep doing so?** The first run
of this spike found that it does not keep doing so. The rule then in force drew an offspring from a
distribution whose spread was the parents' own disagreement, so a closed pool's variance decayed as
it converged, and the spike recorded the consequence correctly — "each pool converges internally, so
the gap between the two pools is largely set early and then holds". That observation became #104,
which added a `mutability` floor under the spread. So this spike now reports a second quantity
alongside the between-pool distance: the **within-pool spread**, which is the thing that was
collapsing.

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
drawn uniformly from [0.5, 1.5]; `drift_margin` 2.0; `mutability` swept, since it is the parameter
whose effect is in question.

## Results

Python 3.12.10, NumPy 2.5.1. Distances are in expressed-phenotype units.

### The collapse, and what stops it

Median within-pool standard deviation of one isolated pool — the variance selection has left to act
on. `mutability = 0` is the control: no floor, which is the rule #104 replaced.

| mutability | gen 20 | gen 50 | gen 100 |
|---|---|---|---|
| **0.0** (control) | 0.0809 | 0.0113 | **0.0004** |
| 0.005 | 0.0831 | 0.0161 | 0.0092 |
| 0.01 | 0.0840 | 0.0227 | 0.0184 |
| 0.02 | 0.0855 | 0.0388 | 0.0359 |
| 0.05 | 0.1146 | 0.0894 | 0.0913 |

The control loses **99.5%** of its spread by generation 100 and is still falling; every floored run
instead settles and holds between generations 50 and 100. That plateau is a mutation-drift balance:
finite-pool sampling loses roughly `σ²/N` per generation and the floor puts the same amount back, so
the equilibrium sits near `σ ≈ 1.8 × mutability` rather than at zero or at infinity.

### Between-pool distance, at `mutability = 0.02`

| Generation | Isolated: min / p5 / median / p95 | Mixed: median / p95 / max |
|---|---|---|
| 20 | 0.100 / 0.239 / 0.456 / 0.729 | 0.063 / 0.156 / 0.224 |
| 50 | 0.145 / 0.272 / 0.503 / 0.837 | 0.025 / 0.061 / 0.161 |
| 100 | 0.122 / 0.278 / 0.515 / 0.890 | 0.021 / 0.037 / 0.052 |

Share of the 200 runs in which each arm crossed a candidate threshold:

| Generation | Threshold | Isolated | Mixed |
|---|---|---|---|
| 20 | 0.12 | 99.5% | 11.0% |
| 50 | 0.08 | 100.0% | 2.5% |
| 50 | **0.12** | **100.0%** | **0.5%** |
| 50 | 0.20 | 99.0% | 0.0% |
| 100 | 0.12 | 100.0% | 0.0% |

## What this settles

- **Drift separates isolated populations, and now it keeps separating them.** The isolated median
  rises 0.456 → 0.503 → 0.515 across the checkpoints where the control's rises 0.435 → 0.473 → 0.475
  and then stops, because the control has frozen. At `mutability = 0.05` the isolated median keeps
  climbing throughout (0.463 → 0.589 → 0.703), which is what a molecular clock requires (§2.5).
- **Twenty generations is too early now.** The mixed arm crosses 0.12 in 11% of runs at generation 20
  and only 0.5% by generation 50, because a mixed pool's halves start with the founders' own spread
  and need time to mix it away. Under the old rule the control never crossed at any checkpoint; the
  cost of keeping variance alive is that the control is noisy early. **A fence must stand for tens of
  generations before its result is trustworthy**, which is a fact about the mechanic and not about
  this test.
- **The control's false-positive rate is not zero, and the test says so.** One run in 200 at
  generation 50 has two arbitrary halves of one interbreeding pool drift past 0.12. That is real
  behaviour — divergence is a distance between centroids and nothing forbids one — so
  `test_an_interbreeding_population_reliably_does_not_diverge` bounds a rate rather than asserting
  zero.
- **Chosen constants: 50 generations, `mutability` 0.02, threshold 0.12** — used by
  `tests/core/genetics/test_speciation.py`. The isolated arm crossed in 200/200 runs with a minimum
  of 0.145, comfortably above the threshold; the mixed arm's median is 0.025, a factor of ~5 below
  it. Generation 100 separates better still (mixed max 0.052) but doubles the test's runtime for a
  distinction it already makes.

### A finding that outlived the measurement

Setting the drift clamp to the numerical backstop it was documented as being made the underlying draw
**variance-expanding**: within-pool spread grew from 2.4 at generation 20 to 4657 at generation 100,
even with no mutability floor at all. The cause is the coefficient relating the draw's spread to the
parental gap. For parents from a pool of variance `σ²`, their midpoint carries `σ²/2` and a draw of
standard deviation `k|a−b|` adds `2k²σ²`, so one generation multiplies variance by `(1/2 + 2k²)` —
neutral only at `k = 1/2`. The ported legacy value was `k = 1/√2`, which inflates variance by half
again every generation, measured at 1.4996 over 200,000 pairs against 0.995 for `k = 1/2`.

**The old rule converged because its two halves were wrong in opposite directions**: an expanding
draw held down by a clamp tight enough to crush the excess. So the clamp was load-bearing, not the
backstop its docstring claimed, and the "conceptually sound" legacy formula (`CLAUDE.md` §1) was two
errors cancelling. `core.genetics.inheritance` now derives `k = 1/2` rather than porting it.

These constants are the test's, not the core's. `has_diverged` and `interbreeding_probability` take
the threshold as a parameter; nothing in `core/` hardcodes 0.12. A world with a different gene
count, population size, `mutability` or `drift_margin` will need its own number, and re-running this
spike is how to get it.
