# The interbreeding threshold, and what it was doing instead

`ConceptionConfig.speciation_threshold` shipped at **8.0**, chosen without measurement while
building #191. This is what it was actually doing, and how the replacement was derived.

## What it was doing

`interbreeding_probability` falls linearly from 1 at identical phenotypes to 0 at the threshold.
Measured on the demo world, 150 founders, 400 ticks, 3,000 sampled pairs, two seeds:

| | seed 0 | seed 1 |
|---|---:|---:|
| median pairwise distance | 30.2 | 27.9 |
| p90 | 61.1 | 56.5 |
| p99 | 73.4 | 71.9 |
| max | 82.4 | 79.5 |
| **pairs that could breed at all, at threshold 8.0** | **6.2%** | **7.5%** |
| median interbreeding probability | 0.000 | 0.000 |

So the gate was rejecting **94% of candidate couples** — not on grounds of reproductive isolation,
but because the threshold sat a quarter of the way to a typical distance between two members of one
healthy population.

## The cost, measured

500 ticks, 150 founders, two seeds:

| threshold | living at the end | conceptions |
|---:|---:|---:|
| 8.0 | 234 | 140 |
| 300.0 | 2,835 | 3,430 |

**Reproduction was suppressed roughly 24-fold.** Every population measurement taken before this fix
was made against a throttled world — including #127's reserve figure and the overshoot-and-settle
curve in `capacity-growth.md`. Those want re-taking.

## The replacement, and the rule behind it

**Four times the 99th-percentile within-population pairwise distance**, which measures ~73, giving
**300**.

The rule is the point, not the number. The gate exists to stop *diverged populations* interbreeding
(§2.5: "reproductive isolation degrades, it does not switch... by the time two populations separate
they were already hybridising at a negligible rate"). So it has to sit above the spread a single
healthy population already carries, or it stops being about isolation and becomes a brake on
ordinary breeding — which is exactly what happened.

At 300: a median pair breeds with probability 0.90, the most distant pair observed within one
population with 0.73, and 100% of pairs can breed at all.

## This is provisional, and must be re-derived after #193

The metric it is calibrated against is dominated by two genes. `docs/spikes/gene-distance-composition.md`
measured `maturity_age` at 64% of squared pairwise distance and `gestation_length` at 34% — 98.4%
between them, with the entire cue block at 1.2%. Whatever #193 settles will change what distance
*means*, and this number with it.

## The guard

`tests/clients/viewer/test_demo_world.py::TestTheShippedTuningIsNotDegenerate` asserts both
directions: most couples in the shipped world can breed, **and** a genuinely diverged pair still
cannot. Either alone is passable by a broken configuration — a threshold of zero satisfies the
second, and one of infinity satisfies the first.

That test is the thing that was missing. The old value passed every test in the suite, because
nothing asserted the gate was open.

## Pattern worth naming

This is the fourth constant this session whose name described something it was not doing:

| constant | shipped as | was actually |
|---|---|---|
| `speciation_threshold` | isolation gate | a 24× brake on all reproduction |
| lust `detection_threshold` | acuity gate | 880× below any reading, never fired |
| `thirst_weight` founding range | quiet drive | 100% of every well-fed decision |
| `intake_rate` (caught pre-merge) | handling time | whole population starved on full ground |

The common shape: a plausible number in a config, a name that reads correctly, and no assertion that
it lands anywhere near the distribution it is compared against. The guard above is the general
remedy — assert the *shape of the outcome*, not the value of the input.
