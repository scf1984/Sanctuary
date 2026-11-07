# Fence perception: what sealed regions do to drives, contact and scent (#169, #276, #139)

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
not built here.

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
