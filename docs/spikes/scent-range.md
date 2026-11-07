# How far can a lone animal smell a mate?

> **Two measurements live here.** #251's, under the three-pass box blur, which chose the range and
> lust's threshold; and #262's (the last section), which replaced the box with an exponential kernel
> and re-derived the threshold by the same method. **The demo world now ships #262's numbers:
> range 10, lust threshold 1e-4.** #251's tables are kept as the record of why, and reproduce only
> from 1b57c0e.

Measured for #251, under the decision recorded there on 2026-11-06: **a longer scent range**, with
the implementation left to settle by measurement whether "longer" means a wider
`CueFieldConfig.diffusion_range`, the acuity gene acting as range through a lower detection
threshold, or both. Bench: `docs/spikes/scent_range.py`. `master` at d16c681, which includes #250
(weather), #252 and #255 (the field-reading drive extraction, #253).

## The finding that decides it: one mate was never perceptible at all

The cue blur conserves mass. So one animal's plume reads lower the wider it spreads, as 1/R², and
the blur (three box passes) has **compact support**: it ends exactly 3R from the source. One typical
mate's lust reading (emission 1, own-signature dot 2, acuity 1), along an axis:

| R | d = 0 | 5 | 10 | 15 | 20 | 30 | 40 | 50 | support |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 2.3e-2 | 9.4e-3 | 0 | 0 | 0 | 0 | 0 | 0 | 9 |
| 6 | 6.7e-3 | 5.4e-3 | 2.4e-3 | 5.3e-4 | 0 | 0 | 0 | 0 | 18 |
| 10 | 2.6e-3 | 2.4e-3 | 1.8e-3 | 1.1e-3 | 5.1e-4 | 7.7e-6 | 0 | 0 | 30 |
| 15 | 1.2e-3 | 1.1e-3 | 1.0e-3 | 8.1e-4 | 5.7e-4 | 2.2e-4 | 3.4e-5 | 0 | 45 |
| 20 | 6.7e-4 | 6.6e-4 | 6.2e-4 | 5.5e-4 | 4.6e-4 | 2.6e-4 | 1.2e-4 | 3.5e-5 | 60 |

Lust's threshold was **0.05**. At the shipped range of 3, one mate read 0.023 *standing next to it*.
**A lone mate was never detected at any distance.** Lust only ever followed crowds, and #251's pairs
met by chance or not at all. Two consequences settle the options:

- **A wider range alone cannot help.** It lowers every single-animal reading further below 0.05.
- **A lower threshold alone cannot reach past 3R.** At R = 3 nothing exists beyond 9 units to detect,
  and #251's pairs sat 15–57 apart.

So the decision's third reading, **both**, is the only one that can work. The bench confirms it.

## Pair meeting: established worlds knocked down to a random pair

Source worlds grown 1,500 ticks under the pre-#251 config (seeds 1–3), each knocked down to two
random born animals, followed 3,000 ticks. "Died" means one of the pair died before conceiving;
"apart" means both were alive at the horizon without having conceived.

| config | pairs | conceived | died | apart | median conception tick |
|---|---:|---:|---:|---:|---:|
| R3, t 0.05 (before) | 30 | 5 | 7 | 18 | 450 |
| R6, t 1e-5 | 30 | 16 | 4 | 10 | 265 |
| R6, t 5e-5 | 30 | 16 | 4 | 10 | 290 |
| R10, t 1e-5 | 30 | 18 | 4 | 8 | 410 |
| **R10, t 5e-5 (shipped)** | 30 | **17** | 4 | 9 | 400 |
| R15, t 1e-5 | 30 | 20 | 7 | 3 | 580 |
| R15, t 2e-5 | 30 | 19 | 7 | 4 | 540 |

A first, 15-pair pass also ran the two single-knob options, and they behave as the profile predicts:

| config | pairs | conceived | died | apart |
|---|---:|---:|---:|---:|
| R3, t 0.05 (before) | 15 | 2 | 4 | 9 |
| R20, t 0.05 (wider only) | 15 | 2 | 4 | 9 |
| R3, t 1e-4 (threshold only) | 15 | 5 | 3 | 7 |
| R20, t 5e-5 | 15 | 9 | 5 | 1 |

**Wider-only is identical to doing nothing**, outcome for outcome. Threshold-only helps pairs that
happen to start within 9 units.

## Pair meeting: two fresh founders 30 apart

The regression test's setup (`tests/core/ecology/test_mates_find_each_other.py`): the demo world at
two founders, placed 30 units apart, aged past maturity and topped up to lust's saturation energy,
16 seeds, 2,000 ticks.

| config | conceived | died first | all conceptions by tick |
|---|---:|---:|---:|
| R3, t 0.05 (before) | 0 | 16 | – |
| R6, t 5e-5 | 4 | 12 | 370 |
| **R10, t 5e-5 (shipped)** | **10** | 6 | 420 |

**This is what separates R6 from R10.** In the bottleneck pairs above the two are indistinguishable,
because an established animal lives long enough to drift into R6's 18-unit reach. A naive founder
mostly doesn't: at 30 apart R6 cannot perceive its mate, and 12 of 16 pairs died first. R10 perceives
a mate out to 26–28 units (the founder acuity range 0.5–1.5) and breeds 10 of 16, every one by tick
420. **R10 is the narrowest range that perceives mates at the distances #251 found them**, rather
than waiting for them to wander into range.

## Chosen: diffusion range 10, lust threshold 5e-5

- `CueFieldConfig.diffusion_range`: 3.0 → **10.0**.
- `LustConfig.detection_threshold`: 0.05 → **5e-5**. It is set against one mate's plume, not a
  crowd's. 1e-5 and 5e-5 are indistinguishable in every table above. The higher one is kept because
  it is the stricter gate.
- Fear's and hunger's thresholds are unchanged.

**What acuity did then, honestly** (fixed by #262, below). Across the founder acuity range the
reach only moved from 26 to 28 units, because the blur's tail falls steeply near its support edge. Once something is perceived,
`detected` normalises acuity out of the appeal. So for lust, acuity is nearly inert. The gene keeps
its teeth in fear, which reads the same nose against absolute thresholds. If a lineage should be able
to evolve a longer nose, the blur needs a tail that does not end (#139's terrain-aware diffusion is
the place to decide that), not a different threshold.

## What it costs: nothing in compute, some locality in fear

**Compute is flat in the range.** Each box pass is a pair of cumulative-sum differences, O(cells)
whatever its width. Median of 15, one cue rebuild plus an (n, 9) candidate sample:

| grid, n | R3 | R6 | R10 | R15 | R20 | R25 |
|---|---:|---:|---:|---:|---:|---:|
| 80×80, 2,000 | 8.6 ms | 8.4 | 7.8 | 8.1 | 7.7 | 6.7 |
| 256×256, 20,000 | 96.7 ms | 96.6 | 101.9 | 100.2 | 96.1 | 98.7 |

**The field is shared, so fear sees the wider blur too.** On the three established source worlds
(about 965 born animals each), at their own cell:

| config | mean fear detection | ρ(danger, neighbours < 5 units) | ρ(danger, neighbours < 20) |
|---|---:|---:|---:|
| R3 (before) | 0.102 | +0.24 | +0.16 |
| R6 | 0.092 | +0.15 | +0.16 |
| **R10 (shipped)** | **0.084** | **+0.09** | +0.16 |
| R15 | 0.078 | +0.07 | +0.14 |
| R20 | 0.072 | +0.04 | +0.12 |

Fear still fires at about the same rate, and its correlation with *regional* density is unchanged.
What it loses is locality: "something near me" becomes partly "something in this region". Fear has
no direction today (its appeal is flat), so this changes how often it is urgent, not where an animal
goes. This is the price the decision anticipated. It is why R10 and not R15. A per-reader range
(a separate field per modality) was not among the decided options and was not built.

## Demo-world viability

`build_demo_world`, 200 founders, 4,000 ticks, seeds 1–5, alive at the end:

| | per seed | mean |
|---|---|---:|
| before (R3, t 0.05) | 3,350 · 2,832 · 3,558 · 3,810 · 4,085 | 3,527 |
| after (R10, t 5e-5) | 3,523 · 4,261 · 3,495 · 3,369 · 3,800 | 3,690 |

Within noise. Every seed grows.

## What it does to #29's viability threshold

Re-measured in `docs/spikes/viable-population.md` (2026-11-06, after #251), with the bench unchanged.
A bottlenecked pair now recovers 7/20 within 10 generations and 15/20 within 50, against 0/40 and
0/20 before. That puts #29's threshold back at two, the biological minimum, and #28's ten-generation
cull guard at 6, against 12.

## Not measured

- **Predator-holding worlds.** Hunger's prey branch reads the same field. On the demo world grass
  dominates hunger (detection 1.00 in every config), so the prey branch's sensitivity to the range
  went unobserved. #179's worlds should re-check it.
- **Grids other than 80×80** for meeting. On a larger map a fixed reach of ~27 units covers a smaller
  share of it, so the same chance-encounter problem returns at lower density.

## #262: an exponential kernel, and what it re-tuned

Measured for #262, under the owner's decision recorded there on 2026-11-07: replace the three-pass
box blur with a **separable two-sided exponential**, the steady state of scent that decays while it
spreads, so that a keener nose detects from further away. `master` at 20a2df4, which includes #249
(thermal drive) and #254 (thirst). Every "box" row below was re-run on that commit, not copied from
the tables above, because the world has moved since d16c681.

### The kernel

`diffusion_range` is now the plume's e-folding distance λ: along a grid axis a reading falls by a
factor `a = exp(−cell_size / λ)` every cell, and never reaches zero. Each axis is one forward and
one backward first-order recursion, so a rebuild is O(cells) whatever λ is. The 2-D kernel is the
product of the two axes, which is what keeps self-exclusion exact (the operator's diagonal and
every off-diagonal entry factorise per axis), at the price of diamond-shaped contours.

**The rim reflects.** Each recursion starts from the mirror image of the whole axis across its end,
and the result is scaled by `(1 − a) / (1 + a)`, the kernel's sum over an infinite line. So a
deposit's plume sums to the deposit wherever it stands, and the operator is exactly symmetric:
what one animal reads of another is what the other reads of it. The box blur claimed a reflecting
rim but, normalised by in-world cells, did not have one: a corner broadcaster kept **51%** of its
scent on the map (a 21-cell grid at range 2) (`TestEdges.test_the_rim_reflects_rather_than_absorbs` fails on it). Only one
image per end is kept; the rest sit an axis length away and are worth `a^length`, 3e-4 of a
deposit at range 10 on the demo world's 80 cells.

### Acuity is range again

One typical mate's lust reading (emission 1, own-signature dot 2, acuity 1) along an axis, under the
exponential (`scent_range.py profile`):

| λ | d = 0 | 5 | 10 | 20 | 30 | 40 | 50 | 60 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 2.0e-2 | 7.3e-3 | 2.7e-3 | 3.6e-4 | 4.9e-5 | 6.7e-6 | 9.0e-7 | 1.2e-7 |
| 7 | 1.0e-2 | 5.0e-3 | 2.4e-3 | 5.8e-4 | 1.4e-4 | 3.4e-5 | 8.0e-6 | 1.9e-6 |
| 10 | 5.0e-3 | 3.0e-3 | 1.8e-3 | 6.8e-4 | 2.5e-4 | 9.1e-5 | 3.4e-5 | 1.2e-5 |
| 15 | 2.2e-3 | 1.6e-3 | 1.1e-3 | 5.9e-4 | 3.0e-4 | 1.5e-4 | 7.9e-5 | 4.1e-5 |

Reach, in world units, of a lone mate at each acuity. The box row is #251's shipped config on its
own kernel; every other row is the exponential:

| kernel, range, lust threshold | acuity 0.5 | 1.0 | 1.5 | 3.0 |
|---|---:|---:|---:|---:|
| box, 10, 5e-5 (before) | 26 | 27 | 28 | 29 |
| exp, 10, 5e-5 | 39 | 46 | 50 | 57 |
| **exp, 10, 1e-4 (shipped)** | **32** | **39** | **43** | **50** |
| exp, 10, 3e-4 | 21 | 28 | 32 | 39 |
| exp, 7, 5e-5 | 32 | 37 | 40 | 44 |
| exp, 7, 2e-4 | 22 | 27 | 30 | 35 |

Tripling acuity now buys `λ · ln 3`, 11 units at range 10, against 2 under the box. The founder
acuity range 0.5–1.5 spans 11 units of reach rather than 2.

### Re-deriving lust's threshold, by #251's method

Range 7 is included because its per-axis spread (√2 · 7 ≈ 9.9) matches the box's at range 10
(√110 ≈ 10.5), so it is the exponential that disturbs fear's locality least. The thresholds put a
lone mate at the reach #251 validated (27 units at acuity 1: t3e-4, t2e-4) or further.

**Two fresh founders 30 apart** (the regression test's setup, 2,000 ticks; 32 seeds rather than
#251's 16, since the bench is cheap):

| config | conceived | died first | median conception tick |
|---|---:|---:|---:|
| box, 10, 5e-5 (before) | 18 | 14 | 135 |
| exp, 10, 5e-5 | 24 | 8 | 80 |
| **exp, 10, 1e-4 (shipped)** | **23** | 9 | 90 |
| exp, 10, 3e-4 | 17 | 15 | 100 |
| exp, 7, 5e-5 | 26 | 6 | 80 |
| exp, 7, 2e-4 | 20 | 12 | 85 |

**Bottlenecked pairs** (source worlds grown 1,500 ticks under 20a2df4's demo config, seeds 1–3,
each knocked to a random pair, 10 reps, 3,000 ticks). The same source worlds served both kernels:

| config | pairs | conceived | died | apart | median conception tick |
|---|---:|---:|---:|---:|---:|
| box, 10, 5e-5 (before) | 30 | 22 | 8 | 0 | 350 |
| exp, 10, 5e-5 | 30 | 23 | 7 | 0 | 240 |
| **exp, 10, 1e-4 (shipped)** | 30 | **23** | 7 | 0 | 240 |
| exp, 10, 3e-4 | 30 | 22 | 8 | 0 | 285 |
| exp, 7, 5e-5 | 30 | 23 | 7 | 0 | 220 |
| exp, 7, 2e-4 | 30 | 21 | 9 | 0 | 190 |

The bottleneck bench cannot tell any of them apart (an established animal lives long enough to
drift into range under all of them), so, as in #251, the founders decide. Holding #251's 27-unit
reach (t3e-4) breeds exactly what the box did; a reach covering 30 units at every founder acuity
breeds a third more and sooner. That is **1e-4 or 5e-5, indistinguishable**, and #251's rule keeps
the stricter: **1e-4**. It is also, to a round number, the strictest threshold at which the dullest
founder (acuity 0.5) still perceives a lone mate 30 units off: `0.5 × 5.0e-3 × e^−3 = 1.2e-4`.

**Range stays 10.** Range 7 did as well or better on both benches, but nothing measured separates
them, and 10 is what the owner's decision was argued at (11 units of reach per tripling of acuity,
against 7.7 at range 7). Choosing range on a difference the benches cannot see would be tuning to
noise.

### Fear

Fear reads the same field at the animal's own cell against absolute thresholds
(`detection_threshold` 0.05, `saturation` 1.0). Both kernels read a uniform crowd at the same
level; what changes is how one neighbour's reading is spread with distance.

On the bottleneck bench's source worlds (sparse, 42–580 born; `scent_range.py readers`):

| config | mean fear detection | ρ(danger, neighbours < 5) | ρ(danger, neighbours < 20) |
|---|---:|---:|---:|
| box, 10 (before) | 0.012 | +0.12 | +0.12 |
| exp, 10 | 0.012 | +0.13 | +0.11 |
| exp, 7 | 0.015 | +0.14 | +0.12 |

That bench is too sparse for fear to fire much, so it was also read on dense worlds: the demo
world grown 4,000 ticks (seeds 6–10, 1,000–4,800 alive), its fear read through its own kernel and
then, **on the identical end state**, through the other one. This isolates the kernel from the
run-to-run differences in evolved aversion, which dominate otherwise (mean fear across seeds runs
from 0.002 to 0.56 under either kernel).

| grown under | seed | mean fear detection, own kernel | same world, other kernel |
|---|---:|---:|---:|
| box | 6 | 0.031 | 0.030 |
| box | 7 | 0.118 | 0.119 |
| box | 8 | 0.323 | 0.327 |
| box | 9 | 0.107 | 0.105 |
| box | 10 | 0.146 | 0.144 |
| exp, 1e-4 | 6 | 0.561 | 0.557 |
| exp, 1e-4 | 7 | 0.545 | 0.542 |
| exp, 1e-4 | 8 | 0.063 | 0.063 |
| exp, 1e-4 | 9 | 0.095 | 0.096 |
| exp, 1e-4 | 10 | 0.103 | 0.103 |

The kernel moves mean fear by at most 0.004 on the same world, and the share of animals afraid by
at most 0.01. On a settled population fear integrates many neighbours, and both kernels conserve
the mass it integrates; the tail shape matters for a lone neighbour, which fear's 0.05 threshold
does not notice under either kernel (one animal's plume peaks at 0.0025 per unit of signature
overlap). The correlation of danger with neighbours within 5 units, #251's locality measure,
averaged +0.04 on the box's dense worlds and +0.03 on the exponential's (seeds 1–5).

**Fear's thresholds are unchanged.**

### Demo-world viability

`build_demo_world`, 200 founders, 4,000 ticks, alive at the end. Ten seeds rather than #251's
five, because five on the exponential came out 600 lower and that needed checking:

| | seeds 1–5 | seeds 6–10 | mean of 10 |
|---|---|---|---:|
| box, 10, 5e-5 (before) | 3,373 · 4,004 · 3,139 · 4,496 · 3,949 | 1,063 · 3,469 · 4,374 · 4,332 · 1,369 | 3,357 |
| exp, 10, 1e-4 (shipped) | 3,076 · 3,264 · 2,024 · 3,854 · 3,660 | 4,126 · 4,781 · 3,443 · 4,612 · 2,045 | 3,489 |

Within noise; the first five's gap reversed in the next five. Every seed survives under both.

### Cost

The rebuild got cheaper, not dearer. Median of 30 rebuilds of an 8-channel field (deposit and
diffusion; the box figures are 20a2df4's):

| grid, broadcasters | box | exponential |
|---|---:|---:|
| 80×80, 2,000 | 3.3 ms | 1.7 ms |
| 256×256, 20,000 | 39.8 ms | 15.2 ms |

The diffusion alone, on a dense random 8-channel deposit: 3.8 → 1.6 ms at 80×80, 37.6 → 12.7 ms at
256×256. The recursion is a Python loop over each axis's length, 4 × 256 steps at 256×256, each a
vectorised operation over the other axis and every channel at once. It is cheaper than the box's
three passes of cumulative sums, gathers and a normalising divide, and like them it does not depend
on the range. Construction builds each axis's full operator by running the same recursion over an
identity matrix, once, so the subtraction can never drift from the spreading.

### For #139

The terrain-aware walk that replaces this kernel must keep **a loss per pass**, so the tail stays
exponential. A loss-free spread of any shape gives a reach that grows only with the square root of
a logarithm of acuity, and acuity goes back to being nearly inert, which is exactly what the box
measured above.

## Reproducing

#262's tables, from this commit's `scent_range.py`. The "box" rows came from the same commands
run at 20a2df4 with `CONFIGS` cut to `R10_t5e-5`, against the **same** source worlds. A snapshot
only loads into the config it was saved under, and #262 changed lust's threshold, so the sources
are established and read with 20a2df4's demo config checked out; the configs under test are then
applied by `retune`, as before:

```
PYTHONPATH=. python docs/spikes/scent_range.py profile
PYTHONPATH=. python docs/spikes/scent_range.py founders --seeds 32 --workers 2
git checkout 20a2df4 -- clients/viewer/demo_world.py
PYTHONPATH=. python docs/spikes/viable_population.py establish --out <dir>
PYTHONPATH=. python docs/spikes/scent_range.py readers --sources <dir>
PYTHONPATH=. python docs/spikes/scent_range.py pairs --sources <dir> --reps 10 --workers 2
git checkout HEAD -- clients/viewer/demo_world.py
```

The rebuild timings and the dense-world viability and fear readings were one-off scratch scripts:
30 `CueField.rebuild` calls per grid on uniformly scattered broadcasters, and `build_demo_world`
advanced 4,000 ticks then read with `Fear._channels` exactly as `readers` does, before and after
swapping `world.scent.field` for the other kernel's `CueField`.

#251's tables, from 1b57c0e:

```
PYTHONPATH=. python docs/spikes/scent_range.py profile
PYTHONPATH=. python docs/spikes/scent_range.py cost
# The tables were measured on source worlds grown under the pre-#251 demo config, and a snapshot
# only loads into the config it was saved under, so run these with that config checked out:
git checkout d16c681 -- clients/viewer/demo_world.py
PYTHONPATH=. python docs/spikes/viable_population.py establish --out <dir>
PYTHONPATH=. python docs/spikes/scent_range.py readers --sources <dir>
PYTHONPATH=. python docs/spikes/scent_range.py pairs --sources <dir> --reps 10
PYTHONPATH=. python docs/spikes/scent_range.py founders --seeds 16
git checkout HEAD -- clients/viewer/demo_world.py
```

The bench's `CONFIGS` holds every row above, and every config (the "before" one included) is applied
by swapping the two knobs on a live world after it is built or loaded. That is safe because the cue
field is rebuilt every tick and carries no state, so which demo config is checked out only decides
what the source worlds grew under.
