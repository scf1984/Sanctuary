# The thermal drive: does sensing temperature turn seasons into migration? (#249)

Before #249 nothing read the temperature field to decide where to go. Animals felt the climate only
as upkeep, so a bad season was differential mortality (range contraction), never movement. #249
adds a drive that reads it. This records what it does on the demo world, and what it does not.
§§1–4 are the demo world, where it does not produce migration. §5 is the temperate world (#261),
where it does.

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

## 5. The temperate world: seasons do produce migration (#261)

§4 put the missing migration down to the map. #261 tests that with a second world,
`clients.viewer.demo_world.temperate_world_config`: the demo world in every respect but the
climate, whose range straddles `neutral_temperature`. Measured at `20a2df4` plus this change, 200
naive founders, 4,000 ticks (5.5 years of 720 ticks), seeds 1–5, drive on and off. The demo world was
re-run alongside it on the same code, so the two are compared like for like (§2's demo numbers
predate #254 and the scent-range change and are not reused here).

### The climate, fitted rather than chosen

With `docs/spikes/temperate_climate_fit.py`, over seeds 1–8 of the terrain, with `amplitude_fraction` left at the demo's 0.5 (#159's guard) and the
weather unchanged:

| gradients × demo | equator above neutral | share of ground-and-year above neutral | mean squared deviation from neutral | summer / winter ground above neutral |
|---:|---:|---:|---:|---:|
| 10 | 16 | 0.37 | 78.2 | 0.63 / 0.22 |
| 10 | **18** | **0.48** | **68.6** | **0.78 / 0.28** |
| 9 | 18 | 0.58 | 54.3 | 0.87 / 0.35 |
| 11 | 18 | 0.39 | 91.7 | 0.66 / 0.23 |
| demo (1, +10) | | 1.00 | 66.1 | 1.00 / 1.00 |

Two conditions fix the two numbers. **About half the ground-and-year on each side of neutral**, so the
comfortable band sits mid-map on average rather than hugging an edge. And **the same mean squared
deviation as the demo world**, because `thermoregulation_rate` was derived from that quantity (#176)
and is shared: the temperate climate moves where the bill falls, not how large it typically is. Both
gradients scale together, so a peak stays about as cold as a pole (#112's pair). The result: winter
poles are at most 8.6 against a neutral of 20, the summer equator's warmest cell reaches at least
31.7 in every seed, and the share of ground
above neutral swings from 28% in winter to 78% in summer.

### What the measurement adds: movement, not just range

Mean occupied latitude cannot tell migration from demography. A winter that kills at the poles and a
summer that breeds there shift the population's mean latitude with no animal moving, and that is
exactly the range contraction #249 set out to get past. So the spike now also records the mean
latitude change of **animals alive at both of two consecutive samples**. Summed over the run and
fitted for its annual cycle (`move_harmonic`), that is the range shift made of movement alone. The
standing crop's biomass-weighted latitude (`forage_harmonic`) is recorded too, because hunger can
follow food that the seasons move.

`lat_harmonic` and `move_harmonic` are amplitudes of the one-year cycle in world units, fitted on sin
and cos together so a lagging cycle counts at full size. A lag near 0 means the cycle peaks at the
summer solstice, and one near 360 means it peaks in midwinter, the wrong way round. Everything is
after the first year.

| world | drive | corr(latitude, season) | swing warm − cold | `lat_harmonic` (lag) | `move_harmonic` (lag) | `forage_harmonic` | \|T − neutral\| where they stand | final n (min) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| temperate | off | +0.72 (0.63 – 0.82) | +4.38 | 3.18 (72) | 1.19 (0.87 – 1.74) | 2.29 | 5.34 | 4,223 (131) |
| temperate | **on** | **+0.77** (0.66 – 0.88) | **+6.17** | **4.43** (67) | **3.10** (2.65 – 3.74) | 2.13 | **4.38** | 4,088 (104) |
| demo | off | −0.04 (−0.16 – +0.20) | −0.12 | 0.35 (483) | 0.39 (0.10 – 0.71) | 0.32 | 8.22 | 3,366 (106) |
| demo | on | −0.05 (−0.08 – −0.01) | −0.61 | 0.78 (494) | 0.99 (0.58 – 1.35) | 0.69 | 7.86 | 3,398 (60) |

Per seed, temperate world, `move_harmonic` off → on: 1.74 → 2.65, 0.99 → 3.03, 1.45 → 3.44,
0.90 → 2.66, 0.87 → 3.74. On is higher in every seed, and the lowest on value is above the highest off
value.

- **Migration: yes, on the temperate world, and the thermal drive is what makes it.** Survivors move
  poleward into summer and back toward the equator into winter by about 3.1 world units of annual
  amplitude with the drive, against 1.2 without it and ~0.4 on the demo world, which is the measure's
  noise floor. The movement peaks 60–100 ticks after the solstice, about a tenth of a year behind
  the temperature: animals follow a moving gradient one look-ahead at a time, so they lag it.
- **Most of the range shift is now movement.** With the drive, movement accounts for about 70% of
  the occupied-latitude cycle (3.10 of 4.43). Without it, under 40% (1.19 of 3.18). The rest is
  demography: in a straddling world, seasons move the range even when nobody chooses to.
- **The drive-off world is not still**, and that is worth knowing before reading the off column as a
  baseline of zero. The standing crop's centroid itself cycles by about 2.2 units, because plant growth
  reads temperature too. Survivors in the drive-off world still move about 40% as far as with the
  drive, most plausibly hunger following that crop. That would be migration by forage, produced by
  the same climate, though this spike does not separate it from the other drives.
- **They stand closer to neutral**: 4.4 temperature units from it with the drive against 5.3 without.
  On the demo world the drive buys 0.4 (8.2 → 7.9), because there is no comfortable ground to reach.
- **The demo world is unchanged in kind**: no seasonal latitude cycle in either condition, and a
  standing poleward shift with the drive (mean latitude 25.1 against 19.6), as §2 found.
- **Viability holds.** Every temperate run grows from 200 to 2,912–4,666. The founder bottleneck is a
  little deeper with the drive (minimum 104 against 131), the same direction as §3's but much milder
  than on the demo world (60 against 106), where the drive pulls every founder toward an edge.

### What this means

§4's diagnosis is confirmed. The drive was never what stopped migration. A map where the coolest ground
is the best ground in every season was. Give the comfortable band somewhere to move and a local sense
with no memory follows it: one look-ahead at a time, a tenth of a year behind, and with the
population's range shift now made mostly of animals moving rather than of where they die.
Hibernation is still absent, since nothing lowers upkeep in place.

Reproduce with `python docs/spikes/thermal_drive.py SEED 4000 on|off temperate|demo`.
