# What the scenario's progress reading shows (#317)

Decision 3 on #317 records what the speciation check measures and shows it as the scenario's
progress: `Speciation.divergence`, the centroid gap between the two groups `propose` finds, over
the gap a split needs. 1 is the gate firing. This spike asks what a player actually sees in that
number, fenced and not, on the demo world.

Script: [`scenario_progress.py`](scenario_progress.py). The demo world, 200 founders, read at every
speciation check (once a generation, 120 ticks).

- **Unfenced** (`open`): seeds 0–5, 6,000 ticks.
- **Fenced** (`fenced`): seeds 0–5. At tick 1,500 a 16-unit square is fenced on the densest part of
  the map, which is what a player is likely to draw. Run until the first split or 40,000 ticks. A
  fixed 24-unit square at the centre of the map was tried first and came out empty or on water in
  three seeds of four, so those runs were discarded; one (`centre_0`) is kept for comparison.

## 1. An unfenced species never reads zero

`propose` splits even one well-mixed population in two, and two halves of a herd have centroids some
way apart. Over ticks 2,000–6,000:

| seed | checks measured | mean | min | max |
|---|---|---|---|---|
| 0 | 33 | 0.30 | 0.24 | 0.37 |
| 1 | 34 | 0.44 | 0.36 | 0.57 |
| 2 | 34 | 0.33 | 0.27 | 0.39 |
| 3 | 32 | 0.44 | 0.31 | 0.60 |
| 4 | 33 | 0.36 | 0.30 | 0.45 |
| 5 | 34 | 0.33 | 0.25 | 0.38 |

So a herd nobody touched reads **a quarter to three fifths**, and it wanders by a fifth from one
generation to the next. Isolation by distance on an 80×80 map with ridges and lakes is real
substructure, and the clustering finds it.

The check skipped a species (two groups not both `min_founders` strong) in 0–2 of 34 unfenced checks
and up to 65 of 183 fenced ones (seed 3, whose pen spent a long stretch under 20). Read raw, a progress bar would drop to nothing for a generation at a
time, which is why the goal reads the last measured value instead.

## 2. A pen barely moves it until it splits

Fenced at tick 1,560 on the densest 16-unit square. "Penned" is the animals living inside a
generation later, and at the split. "Before the jump" is the reading over every measured check from
tick 2,000 until it first passed 0.75.

| seed | penned | split at tick | generations after the fence | at the split, penned | before the jump: mean, max | first past 0.75, generations before the split |
|---|---|---|---|---|---|---|
| 0 | 84 | 17,760 | 135 | 114 | 0.38, 0.72 | 3 |
| 1 | 83 | 22,920 | 178 | 30 | 0.43, 0.74 | 6 |
| 2 | 74 | 9,480 | 66 | 103 | 0.37, 0.56 | 0 (0.57, then 1.45) |
| 3 | 11 | 21,840 | 169 | 28 | 0.46, 0.74 | 32 |
| 4 | 62 | 14,640 | 109 | 86 | 0.34, 0.48 | 3 |
| 5 | 84 | 15,240 | 114 | 84 | 0.40, 0.64 | 0 (0.64, 0.89, then split) |
| 0, centre square | 48 | 19,320 | 148 | | | |

Two shapes, set by the pen's size against the check's `min_founders` (20):

- **A pen of 60 to 100 sits inside the unfenced range until the last few generations**, then jumps:
  seed 2 went from 0.57 to a split in one check, seed 5 from 0.64 through 0.89, seeds 0 and 4 from
  under 0.5 to near 0.9 three generations out. Most of the wait reads like a herd nobody fenced.
- **A pen that dwindles to 20–35 is picked out early and climbs**: seeds 1 and 3 read 0.6–0.8 for
  tens of generations before splitting. Seed 3's pen was 11 when fenced and grew; while it was under
  20 the check could not measure it, which is most of its 65 skipped checks.

**Time to a split**: 9,480 to 22,920 ticks, 66 to 178 generations after the fence, so **2.6 to 6.4
hours at the live rate and 40 to 95 minutes at the fastest rate the demo world offers (4 ticks a
second)**. #16's drift-only harness put a pen of 40 at a median of 100 generations, inside this
range.

## What this means

The reading is honest about what it is: the number the gate decides on. It is **not** a smooth
measure of a pen's drift. A pen of a few dozen to a hundred among thousands outside is not one of
the two groups the clustering proposes until it has drifted most of the way, so for most of the
wait the reading tracks the herd's own halves, inside the unfenced range above. A player told "it
climbs as your pen drifts" would watch it not climb.

So the briefing and the panel say what it does: it starts between a quarter and a half, wanders
there, and moves once the fenced group has drifted far enough to stand out.

A reading of the pen against everyone else would be smooth, and would need the sealed regions
(#169) that the split check deliberately does not read, so that terrain can isolate as well as
fences (#16). That is a different decision from decision 3, and it is recorded on #317 rather than
made here.
