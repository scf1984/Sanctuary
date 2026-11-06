# How far can a lone animal smell a mate?

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

**What acuity does now, honestly.** Across the founder acuity range the reach only moves from 26 to
28 units, because the blur's tail falls steeply near its support edge. Once something is perceived,
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

## Reproducing

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
