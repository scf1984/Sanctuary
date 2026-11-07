# Fear with a direction: does it steer, and what does it cost? (#265)

Until #265, `Fear.appeal` returned flat ones. Fear added the same number to every option, so a
frightened animal never moved away from anything. #265 gives it **relative safety**, the option the
user chose on the issue. Danger at each candidate is read the way urgency reads it where the animal
stands: acuity × each aversion direction · the self-excluded air, gated by `detection_threshold`,
summed over directions. Then `appeal = 1 − danger / max danger over the options`, or all zeros when
nothing is detected anywhere.

This records what that does on the demo world. In short: **the mechanism works, but on the demo
world fear almost never decides anything.** Its urgency is too faint there. Nothing was tuned
around that.

Measured on `feat/265-fear-direction` and on `master` at `4ca5978`, each in its own checkout, with
200 naive founders, 4,000 ticks and seeds 1–5, using `docs/spikes/fear_direction.py`. The spike
reads nothing #265 added, so the same file measures both. Samples are taken every 20 ticks. "Late"
means ticks 2,000–4,000.

## 1. Viability: unchanged within noise

| seed | master final | branch final | master minimum (tick) | branch minimum (tick) |
|---:|---:|---:|---:|---:|
| 1 | 2,747 | 3,406 | 73 (400) | 82 (280) |
| 2 | 2,820 | 2,052 | 39 (500) | 39 (620) |
| 3 | 1,599 | 2,249 | 32 (1,060) | 36 (500) |
| 4 | 3,512 | 3,771 | 40 (620) | 43 (480) |
| 5 | 3,403 | 3,125 | 52 (500) | 55 (360) |
| **mean** | **2,816** | **2,921** | **47** | **51** |

Every world survives the founder bottleneck and grows. Seed-to-seed spread is larger than the
difference between the two conditions in either direction.

## 2. How often fear fires: often, once the world is crowded

The share of the living with fear urgency above zero:

| seed | master, whole run | branch, whole run | master, late | branch, late |
|---:|---:|---:|---:|---:|
| 1 | 0.29 | 0.46 | 0.41 | 0.71 |
| 2 | 0.22 | 0.29 | 0.36 | 0.52 |
| 3 | 0.01 | 0.13 | 0.02 | 0.23 |
| 4 | 0.28 | 0.36 | 0.45 | 0.68 |
| 5 | 0.26 | 0.24 | 0.34 | 0.37 |

**What it fires on is the herd itself.** The demo world has one species and authors no predator.
Founders draw aversion uniformly over [−1, 1]⁸ and signature over [0, 1]⁸, so an animal whose
aversion happens to point along the population's mean signature fears its own kind. Firing
therefore grows with density: a few percent at tick 150, often more than half once thousands are
alive. Read through the noisy-OR, that is a weak fear: mean urgency among the frightened is 0.04 to
0.20.

Late in the run fear fires more on the branch in every seed, by 0.03 to 0.30. That has not been
explained, and five seeds cannot tell it apart from noise; seed 3 on master barely fired at all. If it is real,
the likely route is selection: aversion and fear weight are no longer neutral genes once fear can
move an animal. Measuring that needs aversion genes tracked over time, and is not done here.

## 3. Flight: not detectable on this world

For each animal frightened at a sample, the spike compares the danger it perceives where it stands
with the same animal's reading one tick (one decision) later:

| seed | master: danger fell | branch: danger fell | master: mean relative change | branch: mean relative change |
|---:|---:|---:|---:|---:|
| 1 | 0.50 | 0.51 | −0.007 | −0.009 |
| 2 | 0.52 | 0.52 | −0.018 | −0.025 |
| 3 | 0.45 | 0.56 | −0.015 | −0.016 |
| 4 | 0.51 | 0.46 | −0.008 | −0.020 |
| 5 | 0.52 | 0.48 | −0.007 | −0.006 |

In both conditions it is a coin flip whether a frightened animal's danger falls over the next tick.
Section 4 explains why. The mechanism itself is pinned by unit tests (`TestFearAppeal` and
`TestAllFiveDrivesCompeting` in `tests/core/behaviour/test_drives.py`). In those tests a decisive
animal facing a real plume turns away in 20 draws out of 20, and a hungry one swings wide of a pack
standing between it and its food.

## 4. The #207 spread question: fear almost never decides a choice

The spike scores a fresh decision twice with the same Gumbel draw, once with fear's contribution
and once without it. A choice counts as **decided by fear** when the two picks differ. With a flat
appeal this is zero by construction, which makes master the control.

| seed | decided, all animals | decided, frightened only | fear spread (frightened) | hunger spread (all) | urgency (frightened) | appeal spread (frightened) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.23% | 0.3% | 0.023 | 0.20 | 0.20 | 0.28 |
| 2 | 0.04% | 0.1% | 0.007 | 0.16 | 0.04 | 0.35 |
| 3 | 0.02% | 0.1% | 0.008 | 0.17 | 0.05 | 0.35 |
| 4 | 0.10% | 0.2% | 0.013 | 0.16 | 0.11 | 0.35 |
| 5 | 0.05% | 0.2% | 0.011 | 0.21 | 0.06 | 0.33 |

The late-run maximum is 0.4% of all choices (seed 1). Master is 0 throughout.

- **Appeal is not the bottleneck.** Relative safety does what it was chosen for. Among the
  frightened, the gap between the safest and the most dangerous option is 0.28 to 0.35 of the full
  [0, 1] range. That is comparable to fatigue's 0.25 travel effort.
- **Urgency is.** A frightened animal's urgency averages 0.04 to 0.20, because the herd's plume sits
  just above `detection_threshold = 0.05` and far below `saturation = 1.0`. Fear's spread
  (urgency × appeal spread) therefore comes to 0.007 to 0.023. Hunger's is 0.16 to 0.21, and the
  commitment band is about 0.29. The voice is real, but ten to twenty times too quiet to move a
  ranking.
- **The risk #265 named did not materialise here.** The worry was that a faint detection would
  steer hard because the scale is relative. On the demo world faint detections are the only kind
  there are, and they steer almost nothing, because urgency carries the magnitude, as designed.
  Whether a world with a real predator (#179) or a louder `fear` calibration would show the opposite
  is not measured here.

`tests/core/ecology/test_temperament_evolves.py` records the same finding mechanically: at tick
120, ten times the fear weight changes no heading in any of five seeds.

## 5. Cost: +210 ms per `choose()` at 100,000 animals

`choose()` from `docs/spikes/option_sampling_bench.py`'s `Bench` (100,000 founders, 256 × 256 grid,
N = 8, so 9 options) was timed with a median of 7, alternating branch and master runs three times
each:

| | run 1 | run 2 | run 3 |
|---|---:|---:|---:|
| master `choose()` | 889.0 ms | 878.5 ms | 878.7 ms |
| branch `choose()` | 1,088.4 ms | 1,093.1 ms | 1,094.4 ms |
| branch `Fear.appeal` alone | 215.2 ms | 216.5 ms | 204.6 ms |

So `choose()` costs **about 210 ms more per tick, +24%**, which takes it past §2.1's 1,000 ms budget
at 100,000 entities on this machine. The parts of `Fear.appeal` at that size:

| part | ms |
|---|---:|
| `Scent.perceived_at` over the (100,000 × 9) block | 110 |
| `expressed_columns` for acuity and 16 aversion genes | 29 |
| the aversion product, as a batched `matmul` | 25 |
| float64 casts, the gate, the sum and the scaling | ~50 |

The aversion product was first written as `np.einsum("nkc,ndc->nkd", …)`. That takes NumPy's
unblocked path and cost **133 ms** for the same result (maximum absolute difference 0.0). With it,
`choose()` was 1,179–1,205 ms. Writing it as a `matmul` is a choice of form for new code, not an
optimisation of anything existing, so it was kept.

Nothing else was optimised. `perceived_at` is now read twice per tick, once by lust and once by
fear, over the same candidates with the same self-exclusion. Sharing that read, and the phenotype
block both drives fetch, is #258's and #174's open question, and these numbers are recorded for it.

## Reproducing

```
PYTHONPATH=. python docs/spikes/fear_direction.py SEED 4000 branch   # in this checkout
PYTHONPATH=. python <path>/fear_direction.py SEED 4000 master       # from a master checkout
```

The summary line holds every column above except the late-run
figures, which come from the per-sample lines.
