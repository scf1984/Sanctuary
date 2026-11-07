# Water drag: pricing a step by the depth it wades (#143)

Measured with [`water_drag.py`](water_drag.py) against `origin/master` at `62b33d0`.

## The rule

```
cost = size × (transport_cost × (distance + drag × ∫ depth ds) × (1 + exertion_premium × pace)
               + climb_cost × gain)
```

Depth is `Water.depth`, bilinear on the elevation's node lattice. Along a straight line inside one
node cell it is a quadratic, so the walk integrates it with Simpson's rule on the segments it
already visits (#113), which is exact. One extra depth sample per pass, no extra passes. Animals
still walk the lake bed, so the climb out of a basin is charged as before. Drag multiplies the haul
only.

Perception reads the same price. `CostAwareDiffusion` divides each edge's conductance by
`1 + drag × mean depth` of its two ends, which is exactly what the walk charges along a grid edge.
The forage field, the carrion smell and the reachable-water field all get it.

## Drag 0 changes nothing

`water_drag.py hash` fingerprints every store column plus biomass, soil nutrients and carrion
after 300 ticks of the demo world at drag 0. It ran unchanged on master and on the branch:

| seed | living | master | branch |
|---:|---:|---|---|
| 1 | 92 | `2950fb68b7328cdf` | `2950fb68b7328cdf` |
| 2 | 62 | `dcc2b782b107ddc3` | `dcc2b782b107ddc3` |

Bit-identical, so the new plumbing is neutral until the coefficient bites (§2.8 tier 1).

## The demo map's water

| map | wet cells | mean depth | median | p90 | max |
|---:|---:|---:|---:|---:|---:|
| 1 | 28.8% | 0.46 | 0.37 | 1.05 | 1.76 |
| 2 | 10.4% | 0.21 | 0.13 | 0.49 | 1.37 |
| 3 | 19.1% | 0.27 | 0.17 | 0.71 | 1.26 |

Relief is 7.9. The water is shallow.

## Choosing drag

`water_drag.py run MAP DRAG 900 REP`: 3 maps × 8 replicates × 6 drags, 900 ticks each, 200
founders. The replicate seeds the world's generator apart from its terrain. One run per map
measures nothing here, because a single map's final population ranges from 49 to 615 between runs
at drag 0.

Terms used below:
- A **crossing** is a spell in water (depth > 0) that ends on dry ground more than 4 world units
  from where it began.
- A **deep crossing** is a crossing whose spell reached depth 0.5, the demo world's
  `max_rooting_depth`.
- A **dip** is a spell that ends within 4 units of where it began, which is mostly drinking.

Rates are per 1,000 animal-ticks. Populations are the median and range over all 24 runs.

| drag | final | min | dehydration | wet % | deep % | crossings | deep crossings | dips |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 0 | 102 [49–615] | 66 [38–86] | 0.260 | 18.4 | 0.94 | 2.51 | 0.599 | 22.4 |
| 1 | 120 [59–560] | 64 [39–77] | 0.260 | 17.5 | 0.87 | 2.33 | 0.537 | 21.8 |
| **2** | **110 [40–721]** | **61 [39–84]** | **0.269** | **18.1** | **0.86** | **2.34** | **0.510** | **22.1** |
| 4 | 120 [30–817] | 56 [30–84] | 0.293 | 17.5 | 0.74 | 2.17 | 0.449 | 21.5 |
| 8 | 71 [27–200] | 50 [26–73] | 0.297 | 18.1 | 0.78 | 2.13 | 0.383 | 21.7 |
| 16 | 52 [25–275] | 44 [23–61] | 0.306 | 18.1 | 0.62 | 1.80 | 0.259 | 20.6 |

The next table gives the probability that a random run at this drag scores higher than a random
run at drag 0. A value of 0.5 means the two cannot be told apart.

| drag | final | min | dehydration | deep crossings |
|---:|---:|---:|---:|---:|
| 1 | 0.54 | 0.47 | 0.47 | 0.42 |
| 2 | 0.49 | 0.46 | 0.53 | 0.37 |
| 4 | 0.52 | 0.38 | 0.72 | 0.31 |
| 8 | 0.24 | 0.23 | 0.70 | 0.24 |
| 16 | 0.21 | 0.14 | 0.73 | 0.10 |

**The rule, stated before reading the table:** take the largest drag whose population, both final
and minimum, and whose dehydration stay within 0.4–0.6 of drag 0. That gives **2**. At 4, the
minimum population (0.38) and dehydration (0.72) have started to move. At 8, the final population
is a third lower.

At drag 2:
- The deepest water on the map hauls at 4.5× the dry rate.
- A median lake hauls at 1.3–1.7×.
- A drinking margin under 0.1 deep hauls at under 1.2×.
- Deep crossings fall by 15% and dips do not change, so drinking stays cheap.

## What it does not buy

**On this map, lakes stay weak barriers at any drag the world can afford.** Making them strong costs
viability before it costs crossings. Even at 16, crossings fall only 28% while the population
halves. Two things on the demo map drive this:
- Its water is shallow.
- Every drink is a wet step, so a high drag taxes the thirsty animal as much as the traveller.

Drag gives #16 a graded barrier wherever water is deep. It does not turn this map's lakes into
isolation.

## Throughput

`water_drag.py bench`: one `Movement.step` over 100,000 demo animals, with state restored between
repeats so every repeat walks the same paths. Median of 9 repeats, alternating master and branch,
on a machine shared with other work:

| round | master | branch |
|---:|---:|---:|
| 1 | 133.6 ms | 188.9 ms |
| 2 | 149.1 ms | 214.1 ms |
| 3 | 151.8 ms | 214.0 ms |

The step is about **+42% (+61 ms)**. The cost is two more bilinear samples per pass, the midpoint
and the endpoint depth, against one for elevation before. Against §2.1's 1,000 ms tick, that is 6%
of the budget. The obvious lever, if one is needed, is to sample elevation and depth at a point in
one pass, since they share the grid and the index arithmetic. It is not done here, because nothing
yet shows the tick over budget (§8.5).
