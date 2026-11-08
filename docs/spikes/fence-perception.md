# Fence perception: what sealed regions do to drives, contact and scent (#169, #276, #139, #278, #286)

Measured with [`fence_perception.py`](fence_perception.py). Sections 1 and 2 compare
`origin/master` at `62b33d0` ("before") with `feat/169-region-gate` ("after"); sections 3-5 say
which trees they compare. The script's run modes use only what master already had, so the same file
measures every tree.

## 1. Against the wire

#27's fenced run: the demo world, 300 founders, grown 200 ticks, then a pen over the middle quarter
of the map (cells 20-59 on both axes of 80) and 400 more ticks. Every 10 ticks, the share of penned
animals standing in the ring of cells just inside the wire, divided by that ring's share of the pen
(156 of 1,600 cells). 1.0 is no preference. The outside column is the same for outsiders and the
ring just outside the wire.

| seed | inside, before | inside, after | outside, before | outside, after | alive in / out, before | alive in / out, after |
|---|---|---|---|---|---|---|
| 1 | 3.98 | 0.69 | 0.70 | 0.10 | 11 / 51 | 8 / 42 |
| 2 | 3.48 | 0.49 | 2.25 | 0.28 | 10 / 57 | 9 / 46 |
| 3 | 3.15 | 0.53 | 0.48 | 0.09 | 18 / 102 | 16 / 105 |

**Before**, penned animals crowded the wire at three to four times its area share: every drive
read the far side's field at the candidates beyond it, turned toward food, water or a mate
it could not reach, walked into the wire and was refused.

**After**, the wire band holds about half its share. The gate overshoots into a mild repulsion,
because an option worth nothing loses to any option worth something. The world edge, which has the
same problem, keeps its band at 1.01 of its share by clipping a candidate onto the rim rather than
discarding it ([`edge-candidates.md`](edge-candidates.md)); the equivalent at a fence is #278,
measured in section 6, and it does not hold the wire band at 1.0.

The pens are small (4-18 animals at the end), so these are noisy, but the direction is the same on
every seed. Viability is unchanged in kind: every pen and every outside population survives the
400 ticks on all three seeds.

Which drives carry the repulsion, from the same run with drives exempted from the gate one at a
time: exempting fatigue gives 0.49 / 0.79 / 0.89 inside; exempting fatigue and fear as well gives
0.49 / 0.78 / 0.89. Fear is almost never active in the demo world, and most of the repulsion is the
appetitive drives, where an option across the wire reads nothing by the decision itself.

## 2. What it costs

`Behaviour.choose()` at 100,000 animals, N = 8, on the 256 x 256 bench grid. Within each process
the unfenced and fenced timings are interleaved four times (median of 3 each), and the processes
alternate between the two trees:

| run | before, unfenced | before, four pens | after, unfenced | after, four pens |
|---|---|---|---|---|
| 1 | 958 ms | 895 ms | 885 ms | 924 ms |
| 2 | 908 ms | 928 ms | 914 ms | 955 ms |

The whole-call difference is inside the run-to-run noise. The gate's own work, timed directly on
the fenced world: **marking** which candidates are reachable (`Behaviour.candidates`, one region
gather per candidate) is 23-26 ms, and one masking pass (`Candidates.only_reachable`) 1.7 ms, of
which the six drives make nine. So a fenced world pays about 40 ms per tick at 100,000 animals, and
an unfenced one nothing, since it is one region and neither step runs.

Labelling is recomputed only when a fence changes: 6 ms on 256 x 256 with four pens.

## 3. Contact across the wire (#276)

The same fenced run, counting the touching pairs mating and predation are handed after the fence
goes up, and how many straddle it (`fence_perception.py contact`). Mating pairs are counted before
the compatibility draw, so they are pairs *eligible* to conceive.

| seed | master: mating across / all | master: strikes across / all | #169 only | #169 + #276 |
|---|---|---|---|---|
| 1 | 2 / 55 | 22 / 389 | 0 / 51, 0 / 371 | 0 / 51, 0 / 371 |
| 2 | 2 / 71 | 34 / 615 | 0 / 54, 0 / 624 | 0 / 54, 0 / 624 |
| 3 | 8 / 138 | 6 / 951 | 0 / 131, 0 / 1018 | 0 / 131, 0 / 1089 |

On master, genes and strikes crossed the wire that bodies could not. #169's gate alone brought
both to zero in these runs, because it keeps animals off the wire (section 1); that is a tendency,
not a rule, and #276 makes it one by bucketing contact per sealed region. The two differ on seed 3
because bucketing by region changes which animals pair in a fenced world, as a MAJOR change may.

## 4. Scent cost with k pens (#139)

On the 256 x 256 bench grid at 100,000 animals, before (#276's branch) and after one scent field per
sealed region (`fence_perception.py scent-cost`, median of 7, two alternated runs each):

| pens | rebuild, before | rebuild, after | candidate read, before | candidate read, after |
|---|---|---|---|---|
| 0 | 37.8 / 37.8 ms | 37.1 / 35.3 ms | 114 / 111 ms | 112 / 112 ms |
| 1 | 36.6 / 36.3 ms | 43.2 / 41.1 ms | 114 / 112 ms | 140 / 138 ms |
| 2 | 35.3 / 36.2 ms | 42.1 / 43.7 ms | 114 / 112 ms | 140 / 138 ms |
| 4 | 36.6 / 34.8 ms | 46.2 / 42.5 ms | 115 / 113 ms | 140 / 136 ms |
| 8 | 36.9 / 36.0 ms | 44.8 / 45.1 ms | 118 / 112 ms | 139 / 136 ms |

An unfenced world is unchanged, as it must be: it takes the original code path. With pens up the
rebuild costs 5-9 ms more, because the outside region is still spread over the whole grid and each
pen adds a spread over its own 40 x 40 box. The candidate read (`Scent.perceived_at`, shared by
hunger, lust and fear since #258) costs about 25 ms more, the price of one pass per region with
samplers in it rather than one over everybody.

## 5. Genetic distance between pen and outside (#139)

#27's fenced run carried on for 1,000 ticks after the fence (`fence_perception.py distance`),
before (#276's branch: no contact through the wire, but scent through it) and after (scent per
region). Distance between the pen's and the outside's mean heritable phenotype, over every gene and
over the cue signature genes alone, which cost nothing and drift neutrally:

| seed | signature at fence | signature after, before | signature after, after | all genes after, before | all genes after, after | alive in / out, before | alive in / out, after |
|---|---|---|---|---|---|---|---|
| 1 | 0.136 | 0.745 | 0.789 | 7.3 | 14.9 | 17 / 143 | 5 / 120 |
| 2 | 0.303 | 0.524 | 0.816 | 18.2 | 14.1 | 13 / 109 | 9 / 117 |
| 3 | 0.267 | 0.719 | 0.580 | 4.5 | 17.6 | 79 / 264 | 89 / 302 |
| 4 | 0.237 | 0.478 | 0.482 | 5.0 | 4.0 | 16 / 192 | 10 / 160 |

The pen and the outside diverge in every run, from 0.14-0.30 in signature at the fence to 0.48-0.82
a thousand ticks later. **Scent per region does not measurably change how fast**: the signature
distance averages 0.62 before and 0.67 after, well inside the spread between seeds. That is the
expected result once #276 is in, since what isolates genes is who can touch, and scent only steers
who walks toward whom; with contact already refused at the wire, a mate smelled through it was a
mate that could never be reached. What #139 changes is perception, measured directly by the tests:
an outsider no longer frightens a penned animal or draws it to the wire. The all-genes distance is
dominated by `maturity_age`, whose founding range is two orders of magnitude wider than most genes'
(#193), so it is reported but not read.

## 6. Clipping candidates onto the wire (#278)

#278 replaced #169's gate: a candidate whose straight path crosses a fence is read where the wire
would stop the animal, a thousandth of a cell short of the line, found by `Barriers.reach`, which
`Movement` now uses to stop a step too. Every candidate then lies in its animal's region, so the
gate had nothing left to remove and was deleted. "Before" is `origin/master` at `80bd7f3` (#169's
gate, plus #276 and #139); "after" is `fix/278-clip-to-wire`. Rebased onto `287e47a` (#174) the
seeds re-run give identical figures on both sides.

### The wire band in the live world

`fence_perception.py wire`, as section 1, six seeds, with the rim band for outsiders beside it:

| seed | inside ring, gate | inside ring, clip | rim band, gate | rim band, clip | alive in / out, gate | alive in / out, clip |
|---|---|---|---|---|---|---|
| 1 | 0.80 | 1.89 | 1.37 | 1.60 | 4 / 97 | 7 / 132 |
| 2 | 0.75 | 2.74 | 1.25 | 1.66 | 10 / 51 | 10 / 51 |
| 3 | 0.41 | 3.41 | 1.30 | 1.50 | 12 / 85 | 4 / 72 |
| 4 | 0.26 | 2.18 | 1.75 | 1.99 | 12 / 64 | 14 / 79 |
| 5 | 0.30 | 1.84 | 1.45 | 1.66 | 7 / 89 | 8 / 81 |
| 6 | 0.15 | 1.57 | 1.56 | 1.67 | 4 / 46 | 11 / 59 |
| mean | 0.45 | 2.27 | 1.45 | 1.68 | | |

**Clipping does not hold the band at 1.0; it turns the repulsion into a pull** of about the same
size the other way (0.45 against 2.27, or 1/2.2 against 2.3). That is not what #209's 1.01 at the
rim predicted, and the table says why the prediction did not carry: the rim band in this live world
is not 1.0 either, but 1.25-1.99. The 1.01 was an indifferent population. So the comparison that
matters is the wire against the rim in one run, and the wire still crowds more, by about 1.35x.

The outside ring (outsiders against the wire) moves the same way, from 0.02-0.49 to 0.22-2.60.
Every pen and outside population survives on every seed in both trees.

### The wire with no drives

`fence_perception.py indifferent`, four seeds. With no drives the gate has nothing to mask, so
both trees run the same choices and differ only in `Movement`'s crossing finder:

| seed | inside ring, gate | inside ring, clip | rim band, either |
|---|---|---|---|
| 1 | 1.210 | 1.207 | 0.847 |
| 2 | 1.292 | 1.292 | 0.807 / 0.805 |
| 3 | 1.448 | 1.451 | 0.967 / 0.964 |
| 4 | 1.250 | 1.250 | 0.903 / 0.905 |

**A wire is stickier than the rim before any drive reads anything**: 1.21-1.45 against
0.81-0.97, identically in both trees. The rim clips a landing per axis, so a step into it slides
along it; a fence stops the step dead on its own line (#27), so an animal that heads into it stays
where it is. That is `Movement`'s, not perception's, and it is the same on master.

### Pressed into the wire

`fence_perception.py pressed`, three seeds: the share of penned animal-ticks that chose a
travelling option yet moved less than 0.05 world units.

| seed | ring, gate | ring, clip | rest of pen, gate | rest of pen, clip |
|---|---|---|---|---|
| 1 | 0.100 | 0.149 | 0.029 | 0.032 |
| 2 | 0.070 | 0.174 | 0.009 | 0.009 |
| 3 | 0.074 | 0.143 | 0.005 | 0.007 |

Choosing to rest is the same everywhere (0.10-0.15). What differs at the wire is choosing to go
somewhere and going nowhere, and clipping doubles it. A candidate clipped along its own ray is
pulled back toward the animal, so at the wire every outward heading reads nearly the ground it
stands on: it is #209's folded option (an option proposing almost no displacement under a
travelling index), at a fence instead of a corner. Several near-copies of staying put, one real
one, and a wire that stops the step dead together make the wire a place animals stay. The forage
field is not the cause: fencing raises its reading in the ring against the pen's interior by
0-3% (0.781 to 0.767, 1.001 to 1.031, 1.033 to 1.040 on seeds 1-3).

What would match the rim is the rim's own rule, a projection rather than a truncation: slide the
step and the candidate along the wire. That is a decision rather than a fix, filed as #286.

### What it costs

`fence_perception.py choose`, 100,000 animals, N = 8, 256 x 256, four pens; alternating trees,
two runs each. The first pair was taken on a quiet machine before rebasing onto #174; the second
after, with another agent's benchmarks running, so it is slower across the board.

| | gate, run 1 | gate, run 2 | clip, run 1 | clip, run 2 |
|---|---|---|---|---|
| `choose()` unfenced, quiet | 954 ms | 988 ms | 951 ms | 973 ms |
| `choose()` four pens, quiet | 1,007 ms | 1,005 ms | 1,003 ms | 1,021 ms |
| placing candidates, unfenced / pens, quiet | 27 / 27 ms | 29 / 28 ms | 29 / 89 ms | 28 / 86 ms |
| gate marking, pens, quiet | 23 ms | 27 ms | - | - |
| `Movement.step` unfenced / pens, quiet | 161 / 218 ms | 173 / 210 ms | 177 / 172 ms | 165 / 168 ms |
| `choose()` unfenced / pens, loaded | 1,623 / 1,727 ms | 1,629 / 1,763 ms | 1,590 / 1,777 ms | 1,367 / 1,489 ms |
| `Movement.step` unfenced / pens, loaded | 260 / 371 ms | 247 / 356 ms | 288 / 300 ms | 181 / 186 ms |

Clipping costs about 60 ms per `choose` with four pens (placing 86-89 ms against 27-29 ms), and
the gate it replaces cost about 40 ms (23-27 ms marking plus nine 1.8-2 ms masks), so `choose`
is within noise of the gate. `Movement.step` gets cheaper with pens up, from 210-218 ms to
168-172 ms, because the fence check moved out of the walk (which visited a second lattice and
probed every pass) into one `reach` call over the steps that could meet a wire. An unfenced world
takes neither path.

Walking every candidate was what #169 measured at 2.7-7.6 s per tick. `reach` walks only rays
that two summed-area box queries cannot rule out: per animal (any wire within one `look_ahead`,
16% of animals with four pens) and then per ray (any wire in the box its two end cells span).

### Unfenced worlds

Same seed, identical output: `build_demo_world(seed, 200)`, 300 ticks, SHA-256 over every store
column on live rows and their ids. `287e47a`: seed 1 `2a847dec...59cca07`, seed 2
`727c9d1f...9618593c`; the branch gives the same two hashes, as did `80bd7f3` before the rebase.

## 7. Sliding along the wire instead of stopping (#286)

The owner's decision on #286 was to measure option B (slide) against A (#287's stop) on the same
runs before deciding. Three trees, all built on `fix/278-clip-to-wire` at `86005ad`:

- **A** is #287 as it stands. A step stops dead a margin short of the wire along its own ray, and
  a candidate is read there.
- **B** (`d30e56f`) adds `Barriers.slide`. A straight step is walked to its first blocked
  crossing. It bends a `WIRE_MARGIN` short of the wire, measured across the wire, drops its
  component normal to the wire and walks the rest along it, stopping a margin short of a corner.
  `Movement.step` walks and prices both legs, and writes velocity from the net displacement.
  `candidate_positions` reads each candidate at the slide's stop.
- **B+T** (`812293d`) is B, plus `chosen_target` slides the steering target the same way. The rim
  clips the target as well as the candidate (`chosen_target`'s `np.clip`), so the target is the
  third place the rim's rule acts. B leaves it out.

B+T was not in the brief. It is here because B alone did not move the population with no drives
(below), and the target was the reason. Steered at the raw point across the wire, an animal leans
on it and slides at only the tangential part of its speed, so it lingers. At the rim the clipped
target points along the wall, and the animal travels at its full pace.

All the modes below are `fence_perception.py`, seeds 1-6, the same runs as section 6. Tree A
reproduces section 6's figures exactly on every seed.

### The wire band in the live world

`wire`: penned animals in the ring just inside the wire, and outsiders in the band within one
world unit of the rim, each over its area share.

| seed | ring, A | ring, B | ring, B+T | rim, A | rim, B | rim, B+T |
|---|---|---|---|---|---|---|
| 1 | 1.89 | 2.72 | 2.05 | 1.60 | 1.60 | 1.76 |
| 2 | 2.74 | 2.83 | 0.74 | 1.66 | 1.23 | 1.70 |
| 3 | 3.41 | 3.41 | 2.11 | 1.50 | 1.18 | 1.63 |
| 4 | 2.18 | 2.83 | 1.96 | 1.99 | 2.38 | 2.25 |
| 5 | 1.84 | 1.49 | 1.06 | 1.66 | 1.60 | 1.75 |
| 6 | 1.57 | 2.15 | 1.94 | 1.67 | 1.58 | 1.56 |
| mean | 2.27 | 2.57 | 1.64 | 1.68 | 1.60 | 1.78 |
| ring / rim | 1.35 | 1.61 | 0.93 | | | |

B does not lower the ring. B+T brings it to the rim's level in the same runs. Outsiders against
the wire (the outside ring) read A 0.22-2.60 (mean 1.03), B 0.31-2.60 (1.06) and B+T 0.18-1.25
(0.65).

### The wire with no drives

`indifferent`: 2,000 founders, the pen over the middle quarter, every drive removed.

| seed | ring, A | ring, B | ring, B+T | rim, A | rim, B | rim, B+T |
|---|---|---|---|---|---|---|
| 1 | 1.207 | 1.277 | 0.818 | 0.847 | 0.847 | 0.865 |
| 2 | 1.292 | 1.301 | 0.941 | 0.807 | 0.812 | 0.813 |
| 3 | 1.451 | 1.507 | 0.973 | 0.964 | 0.949 | 0.966 |
| 4 | 1.250 | 1.372 | 0.892 | 0.905 | 0.926 | 0.927 |
| 5 | 1.250 | 1.220 | 0.816 | 0.911 | 0.918 | 0.922 |
| 6 | 1.285 | 1.327 | 0.932 | 0.832 | 0.849 | 0.861 |
| mean | 1.289 | 1.334 | 0.895 | 0.878 | 0.884 | 0.892 |

**Sliding the step alone is not the rim's rule.** B is as sticky as A, and slightly more so (1.33
against 1.29). With no drives a choice depends only on the heading, and the heading's steering
target still lies across the wire. So the animal is steered into the wire every tick, slides at
`speed x sin(angle)` and stays in the ring. B+T matches the rim, at 0.895 against 0.892.

This measure has two caveats. First, it is not 1.0 without a fence. On the demo terrain the same
ring reads 1.04 with no fence at all (seed 1). On flat, dry ground with energy topped up every tick
it reads 1.05 and 0.95 (seeds 1-2). So about 5% either way is terrain and noise. Second, the rim
itself reads 0.81-0.97 at this one-unit scale, which #209's 1.01 (measured over a band one
`look_ahead` wide) did not resolve. On that flat, dry ground the fenced ring reads A 1.29 / 1.38,
B 1.30 / 1.39 and B+T 0.84 / 0.93, with the rim at 0.85 throughout. With the terrain removed, the
picture is the same.

### Pressed into the wire, speed, and laps

`pressed`, on #27's fenced run, reports four things for penned animals, in the ring and in the rest
of the pen:

- the share of animal-ticks that chose a travelling option and moved less than 0.05 world units;
- the mean speed of travelling animals, in world units per tick;
- per visit to the ring (consecutive ticks there, by id), how long the visit lasts;
- the longest distance covered in one visit, over the ring's perimeter (1.0 would be a full lap).

| seed | pressed ring A / B / B+T | pressed interior A / B / B+T | speed ring A / B / B+T | speed interior A / B / B+T |
|---|---|---|---|---|
| 1 | 0.149 / 0.025 / 0.015 | 0.032 / 0.026 / 0.026 | 0.37 / 0.44 / 0.53 | 0.49 / 0.50 / 0.50 |
| 2 | 0.174 / 0.024 / 0.006 | 0.009 / 0.011 / 0.021 | 0.38 / 0.48 / 0.52 | 0.50 / 0.52 / 0.49 |
| 3 | 0.143 / 0.028 / 0.017 | 0.007 / 0.009 / 0.008 | 0.38 / 0.45 / 0.55 | 0.52 / 0.51 / 0.54 |
| 4 | 0.144 / 0.024 / 0.014 | 0.010 / 0.010 / 0.012 | 0.33 / 0.40 / 0.45 | 0.42 / 0.43 / 0.42 |
| 5 | 0.162 / 0.022 / 0.023 | 0.009 / 0.007 / 0.003 | 0.39 / 0.48 / 0.56 | 0.52 / 0.51 / 0.54 |
| 6 | 0.136 / 0.029 / 0.027 | 0.015 / 0.012 / 0.009 | 0.34 / 0.35 / 0.46 | 0.50 / 0.49 / 0.50 |

| | A | B | B+T |
|---|---|---|---|
| chose rest, ring | 0.10-0.12 | 0.10-0.14 | 0.09-0.13 |
| ring visit, mean ticks | 9.0-12.1 | 9.0-12.9 | 5.5-9.6 |
| ring visit, 95th percentile ticks | 25-41 | 25-38 | 14-30 |
| longest visit / perimeter | 0.07-0.16 | 0.12-0.26 | 0.07-0.19 |
| 95th-percentile speed, ring | 0.71-0.85 | 0.61-0.93 | 0.89-1.10 |
| 95th-percentile speed, interior | 0.83-0.95 | 0.85-0.94 | 0.82-1.03 |

Both B and B+T remove the pressed animals. Choosing to travel and going nowhere falls from 14-17% in
the ring to 1-3%, the interior's level. Under B the ring is still slower than the interior, and
visits last as long as under A, which is why its density does not fall. Under B+T the ring moves at
the interior's speed (2-9% faster on five seeds of six), and visits are shorter.

**No laps.** The longest single visit to the ring covered at most a quarter of its perimeter under
B and a fifth under B+T (a sixth under A). Ring speeds stay within the interior's range at the 95th
percentile. A fence has become a wall an animal can travel along, which the rim already is, and
nothing in these runs circles it.

### Viability

Alive in / out at the end of `wire`:

| seed | A | B | B+T |
|---|---|---|---|
| 1 | 7 / 132 | 7 / 73 | 8 / 75 |
| 2 | 10 / 51 | 8 / 38 | 4 / 33 |
| 3 | 4 / 72 | 6 / 92 | 7 / 89 |
| 4 | 14 / 79 | 15 / 60 | 12 / 64 |
| 5 | 8 / 81 | 6 / 71 | 8 / 78 |
| 6 | 11 / 59 | 5 / 56 | 9 / 62 |

Every pen and every outside population survives on every seed in all three trees. The outside
totals differ (means 79, 65 and 67), mostly through A's seed 1. Six seeds of a 300-founder world do
not separate them, and this run was not designed to.

### What it costs

`fence_perception.py choose`: 100,000 animals, N = 8, 256 x 256, four pens. The trees alternated
A, B, B+T and then again, on a quiet machine. Each figure is the median of each run; two runs per
tree.

| | A | B | B+T |
|---|---|---|---|
| `choose()` unfenced | 827 / 822 ms | 820 / 821 ms | 822 / 824 ms |
| `choose()` four pens | 880 / 881 ms | 908 / 900 ms | 899 / 900 ms |
| placing candidates, four pens | 85 / 86 ms | 112 / 110 ms | 111 / 109 ms |
| `Movement.step` unfenced | 120 / 120 ms | 120 / 113 ms | 122 / 111 ms |
| `Movement.step` four pens | 121 / 126 ms | 141 / 130 ms | 130 / 126 ms |
| `chosen_target` four pens | 2.4 / 2.3 ms | 2.3 / 2.3 ms | 17.3 / 16.5 ms |

The slide's second pass costs about 25 ms in placing candidates, because each blocked candidate is
walked again along the wire. `chosen_target` costs about 15 ms more under B+T, because it is one
slide per animal near a wire. `Movement.step` costs 0-15 ms more, though this bench steps founders
from rest, so few steps reach a wire. In all, B+T costs about 45 ms a tick over A at 100,000
animals with four pens, about 5% of §2.1's 1 s tick. Unfenced worlds take none of these paths.

### Unfenced worlds

All three trees give the same output for the same seed: `build_demo_world(seed, 200)`, 300 ticks,
SHA-256 over every per-row store array on live rows plus their ids. That is a slightly different
column set from section 6's, so the digests differ from it. Seed 1 gives `95c0851f...7248583` and
seed 2 gives `a4adac3e...6041dab` on A, B and B+T alike.

### Reading

B as specified fixes the pressed animals and nothing else: the ring stays at A's density, both in
the live world and with no drives. The rim's rule acts in three places (the landing, the
candidate and the steering target), and B carries two of them. B+T carries all three. It is the
only variant that makes the wire read like the rim: 0.895 against the rim's 0.892 with no drives,
and 0.93x the rim band in the live world against A's 1.35x.
