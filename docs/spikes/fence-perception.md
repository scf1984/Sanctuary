# Fence perception: what a sealed region does to where animals stand (#169)

Measured with [`fence_perception.py`](fence_perception.py) against `origin/master` at `62b33d0`
("before") and `feat/169-region-gate` ("after"). The same script runs on both, since it uses only
what master already had.

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
