# The thermal drive: does sensing temperature turn seasons into migration? (#249)

Before #249 nothing read the temperature field to decide where to go. Animals felt the climate only
as upkeep, so a bad season was differential mortality (range contraction), never movement. #249
adds a drive that reads it. This records what it does on the demo world, and what it does not.

Measured on `feat/249-thermal-drive` (master at `d16c681` plus this change), 200 naive founders,
4,000 ticks, seasons on (720-tick year), with `docs/spikes/thermal_drive.py`. "Off" is the same
world and vocabulary with the drive's detection threshold set out of reach, so it contributes
nothing to any option.

## 1. The two numbers, measured rather than chosen

Founders on the demo world over the first year (seeds 1–2, ticks 0–540), energy units per tick:

| quantity | p10 | median | p90 |
|---|---:|---:|---:|
| thermoregulation bill where the animal stands | 0.026–0.056 | 0.040–0.074 | 0.060–0.098 |
| total upkeep (founders, tick 0) | 0.19–0.20 | 0.22–0.23 | 0.24–0.26 |
| best saving among the candidates, one look-ahead away | 0.0007–0.0025 | 0.0022–0.0065 | 0.008–0.013 |

- **`saturation_upkeep = 0.15`**: a founder's upkeep *outside* thermoregulation (about 0.22 − 0.065).
  The weather may cost as much as everything else the body does before escaping it outranks every
  other drive. Typical urgency is therefore about 0.3–0.5 times the weight.
- **`detection_threshold = 0.002`**: between the 10th percentile and the median of the best saving
  at acuity 1.
  A founder of acuity 0.5 perceives roughly half the gradients it stands in, one of 1.5 nearly all,
  so the charged sense buys something across the founding band.

## 2. Range: animals sort toward cooler ground; they do not migrate with the seasons

The demo map is hot everywhere (26–31 against a neutral of 20), so a thermal drive can only ever
point toward the coolest ground, which is away from the equator. Mean distance from the equator of
the living, after the first year:

| seed | off: mean latitude | on: mean latitude | off: corr(latitude, season) | on: corr(latitude, season) | off: swing warm−cold | on: swing warm−cold |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 19.20 | 22.97 | −0.026 | +0.081 | −0.02 | +0.34 |
| 2 | 20.26 | 23.55 | −0.175 | −0.002 | −0.34 | −0.21 |
| 3 | 18.30 | 23.27 | +0.025 | +0.020 | +0.18 | −0.02 |
| 4 | 18.70 | 25.31 | −0.197 | −0.134 | −0.60 | −1.05 |
| 5 | 20.27 | 24.16 | +0.060 | +0.046 | +0.04 | +0.28 |
| **mean** | **19.35** | **23.85** | **−0.063** | **+0.002** | **−0.15** | **−0.13** |

- **A standing shift: yes.** With the drive the population sits about 4.5 world units further from
  the equator in every seed: a cooler range, chosen by the animals rather than left by the dead.
- **Seasonal migration: no.** Occupied latitude does not track the season in either condition.
  Correlations are within ±0.2 and of either sign, and the warm-minus-cold swing is a fraction of a
  unit with no consistent sign. The temperature the animals stand in tracks the season just as
  strongly with the drive as without (corr +0.74 to +0.87 on, +0.80 to +0.92 off), so they ride
  the seasons where they are rather than moving to escape them.

**Why, most likely.** Two reasons, and neither is a tuning problem. Neither was tuned around (the
brief was to report this honestly):

1. **There is nowhere better to go in summer.** The seasonal swing is at most half the latitude
   gradient (#159's inversion guard), and on this map the coolest ground in summer is the same
   coolest ground as in winter: the poles. A gradient that never reverses gives a local sense
   nothing seasonal to follow. Migration needs a world where the comfortable ground *moves*, which
   means a span that straddles neutral or terrain that does.
2. **The drive sees one look-ahead.** It follows the local gradient and has no memory of where it
   was comfortable, so even where a better region exists far away, nothing carries the animal
   across the uncomfortable ground in between.

## 3. Viability, and a deeper founder bottleneck

| seed | off: final | on: final | off: minimum (tick) | on: minimum (tick) | off: deaths warm/cold | on: deaths warm/cold |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3,637 | 2,971 | 109 (320) | 81 (300) | 10,971 / 9,064 | 13,163 / 11,009 |
| 2 | 3,465 | 3,224 | 90 (460) | 45 (620) | 11,936 / 10,663 | 11,410 / 9,251 |
| 3 | 2,842 | 3,570 | 79 (340) | 53 (580) | 9,576 / 8,275 | 10,509 / 8,657 |
| 4 | 3,406 | 2,334 | 93 (380) | 42 (620) | 12,027 / 10,334 | 3,948 / 2,509 |
| 5 | 3,463 | 3,726 | 78 (420) | 43 (420) | 14,771 / 13,128 | 10,467 / 8,443 |
| **mean** | **3,363** | **3,165** | **90** | **53** | warm/cold 1.15 | warm/cold 1.24 |

- **The world stays viable**: every seed grows from 200 to between 2,334 and 3,726 by tick 4,000.
  The final means are within noise of each other.
- **The founder bottleneck is deeper and later with the drive**: a mean minimum of 53 against 90,
  reached around tick 300–620 against 320–460. Seed 4 bottoms at 42 and recovers slowest (still
  2,334 at the end, and correspondingly fewer deaths in total).
- **Selection turns the drive down.** Mean expressed `thermal_weight` over the living (seeds 2, 4, 5):

  | tick | 0 | 1,000 | 2,000 | 3,000 |
  |---|---:|---:|---:|---:|
  | seed 2 | 1.00 | 0.78 | 0.79 | 0.74 |
  | seed 4 | 0.96 | 0.86 | 0.82 | 0.80 |
  | seed 5 | 1.03 | 0.95 | 0.46 | 0.53 |

  `hunger_weight` holds or rises over the same runs (0.98–1.35). Naive founders weight the new drive
  like every other (0.6–1.4, §2.5, #101). On a map whose every cell is too hot it pulls them toward
  the edge every tick, at the expense of food and water, and the lineages that weight it less
  survive the first year better. This is #23's mechanism doing its job, and the bottleneck is its
  price. It was not tuned away: founders are naive by design, and a founding range chosen to hide
  this would be authoring the outcome.
- **Seasonal mortality is not what the drive changes.** Deaths are 15–24% higher in the warm half of
  the year in both conditions, as expected on a map that is too hot everywhere.

## 4. What this means

The drive does what its tests say: animals choose cooler or warmer headings when that saves upkeep,
and a population sorts into the cooler band of a hot map. It does **not** produce migration or
hibernation here, and the reason is the world, not the drive: on a map entirely on one side of
neutral, with a seasonal swing that cannot reverse the gradient, there is nowhere seasonal to go.
The place to look for migration is a world whose thermal span straddles neutral (the split-map
setup in `thermal-divergence.md` does), or one where terrain makes the comfortable ground move. That
is the next measurement, not a reason to tune this one. Hibernation additionally needs a way to
lower upkeep in place, which nothing provides.
