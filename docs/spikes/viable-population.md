# How small can a population get and still come back?

Measured for #29, under the decision recorded there on 2026-11-06: **a viable breeding population is
calibrated offline by simulation**, and the measured number ships with the rule version. #28's cull
guard reads the same number.

The headline result is that **under today's rules there is no such number above two.** An
established population knocked down to any size of two or more never died out in 620 runs. Small
bottlenecks are slow, not doomed, and the reason is two missing mechanics rather than anything about
small populations. What *does* have a threshold is a population of naive founders, and that is a
different question.

## What exists today, and what this was measured under

`master` at 5fe0c75, `build_demo_world` (80×80 grid, cell size 1). Windows 10, Python 3.13.5,
NumPy 2.5.4, 14 worker processes.

Reproduction as it stands (#20 is still open):

- **No sexes.** Any two animals can conceive. Sex allocation is #99 and doesn't exist, so a
  sex-ratio sensitivity check is not possible and was not run.
- Two animals conceive when both are mature (`age >= maturity_age`, a gene, founded 40–120 ticks),
  both hold more than `breeding_energy`, they are within `contact_range` (2.0 world units), and the
  genetic-distance gate passes (`speciation_threshold` 300, which in one population is effectively
  always).
- **One young per couple per conception**, endowed with 60 energy taken equally from both parents and
  gestating as a negative age for `gestation_length` ticks (a gene, founded 20–60).
- Finding a mate is the lust drive, steered by the scent cue field (#188). There is no other search.

Two things are **absent**, and they shape every result below:

- **Senescence** (#239). Designed in §2.5 and #21, but not built. Nothing dies of age, so an animal
  that can feed lives forever.
- **Predators.** The demo world holds one interbreeding species founded undecided about diet. #179
  measured that it does not hold both grazers and carnivores.

So the only thing that kills an animal here is starvation or thirst.

**Re-measure this when any of these land:** #239 (senescence), #20's remaining scope, #240 (the
crash below), a predator-holding world, or any MAJOR (§2.8). Every one of them changes the result
directly.

## Definitions

Each run ends in one of four outcomes:

| outcome | meaning |
|---|---|
| **recovered** | the born population reached **200** |
| **extinct** | no living rows, unborn included |
| **crashed** | the tick loop raised #240's ledger error (see below) |
| **stalled** | none of those within the horizon |

- **200** is the demo world's own founding size. It is checked rather than assumed: every
  bottleneck *to* 200 survived (40/40), as did every one to 100. A population that reaches it is past
  the risk this spike is measuring.
- **Horizon: 10 generations = 1,130 ticks.** A generation is taken as mean expressed
  `maturity_age + gestation_length` across the living animals of the three source worlds (118.9,
  110.1 and 108.7 ticks, mean 113). That is the *shortest* a lineage can turn over, so the horizon
  is conservative. §2.1's ratio table puts generation time at ~2 sim-months. The demo world runs far
  faster than that, which is why the horizon is set in its own generations rather than in §2.1's
  units.
- A follow-up re-ran the small sizes at **50 generations** (5,630 ticks) to see what "stalled"
  becomes.

## Bottleneck: an established population knocked down to n

Three source worlds (seeds 1, 2, 3) were founded at 200 and grown for 1,500 ticks, reaching 1,487,
2,248 and 1,070 born animals. Each replicate restores a snapshot, reseeds the world's generator, and
keeps `n` born animals chosen at random. Everything else alive, unborn included, is released with
its body returned to the ground it lay on, exactly as `Cull.apply` does. 40 replicates per n,
rotating across the three sources.

| n | runs | recovered | extinct | stalled (bred) | P(recover) | 95% CI | median ticks |
|---:|---:|---:|---:|---:|---:|---|---:|
| 2 | 40 | 5 | 0 | 35 (4) | 0.12 | [0.05, 0.26] | — |
| 3 | 40 | 9 | 0 | 31 (9) | 0.23 | [0.12, 0.38] | — |
| 4 | 40 | 14 | 0 | 26 (7) | 0.35 | [0.22, 0.50] | — |
| 6 | 40 | 30 | 0 | 10 (5) | 0.75 | [0.60, 0.86] | 895 |
| 8 | 40 | 35 | 0 | 5 (5) | 0.88 | [0.74, 0.95] | 675 |
| 12 | 40 | 40 | 0 | 0 | 1.00 | [0.91, 1.00] | 560 |
| 16–200 | 40 each | 40 | 0 | 0 | 1.00 | [0.91, 1.00] | 500 → 40 |

Intervals are Wilson 95%. "Median ticks" is time to recovery, shown where most runs recovered.

**Within 10 generations, P(recovery) crosses 0.5 between n = 4 and n = 6, and 0.95 between n = 8 and
n = 12.**

And over 50 generations:

| n | runs | recovered | extinct | stalled | P(recover) | 95% CI | median ticks |
|---:|---:|---:|---:|---:|---:|---|---:|
| 2 | 20 | 11 | 0 | 9 | 0.55 | [0.34, 0.74] | 4,865 |
| 3 | 20 | 19 | 0 | 1 | 0.95 | [0.76, 0.99] | 2,220 |
| 4 | 20 | 19 | 0 | 1 | 0.95 | [0.76, 0.99] | 2,060 |
| 6 | 20 | 20 | 0 | 0 | 1.00 | [0.84, 1.00] | 815 |
| 8 | 20 | 20 | 0 | 0 | 1.00 | [0.84, 1.00] | 645 |

**Zero extinctions in all 620 bottleneck runs.** A small bottleneck isn't doomed; it is slow. The
lag is mate-finding, a real Allee effect: a pair scattered across a 79×79 world has to find each
other by scent within a contact range of 2. With nothing to kill it while it searches, and a field
made rich by the returned bodies and the absence of other grazers, it eventually does.

So the 10-generation curve is a measurement of **waiting time, not of doom**. It moves with the
horizon (n = 3 goes from 0.23 to 0.95 between 10 and 50 generations), which is the signature of a
quantity that isn't a threshold.

## Naive founders: n animals drawn from the founder ranges

The same sizes, founded fresh with `demo_world_config(n, seed)`, 40 replicates each. Founders are
unselected: the full 200-founder world loses ~40% of them before tick 300 and only then grows.

| n | runs | recovered | extinct | crashed | stalled (bred) | P(recover) | 95% CI |
|---:|---:|---:|---:|---:|---:|---:|---|
| 2 | 40 | 0 | 7 | 6 | 27 (2) | 0.00 | [0.00, 0.09] |
| 3 | 40 | 0 | 7 | 3 | 30 (3) | 0.00 | [0.00, 0.09] |
| 4 | 40 | 0 | 6 | 5 | 29 (6) | 0.00 | [0.00, 0.09] |
| 6 | 40 | 0 | 1 | 1 | 38 (19) | 0.00 | [0.00, 0.09] |
| 8 | 40 | 0 | 0 | 0 | 40 (20) | 0.00 | [0.00, 0.09] |
| 12 | 40 | 1 | 0 | 0 | 39 (32) | 0.03 | [0.00, 0.13] |
| 16 | 40 | 1 | 0 | 0 | 39 (29) | 0.03 | [0.00, 0.13] |
| 24 | 40 | 4 | 0 | 0 | 36 (35) | 0.10 | [0.04, 0.23] |
| 32 | 40 | 18 | 0 | 0 | 22 (22) | 0.45 | [0.31, 0.60] |
| 48 | 40 | 24 | 0 | 0 | 16 (16) | 0.60 | [0.45, 0.74] |
| 64 | 40 | 35 | 0 | 0 | 5 (5) | 0.88 | [0.74, 0.95] |
| 100 | 40 | 37 | 0 | 0 | 3 (3) | 0.93 | [0.80, 0.97] |

The naive n = 200 row is omitted because it starts at the recovery size and so says nothing.

**Naive founders cross P = 0.5 at about n = 32–48, and approach 0.95 only at about n = 100.** That is
roughly an order of magnitude above the bottleneck curve, and the gap is founder maladaptation, not
demography. Here, and only here, populations actually die out (10 to 13 of 40 at each n ≤ 4,
counting crashes).

### Density: the same founders in a world four times the area (grid 160)

20 replicates per n:

| n | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|
| P(recover) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.50 [0.30, 0.70] |

Against 0.45 at n = 32 on the 80-grid, the crossing moves to about n = 64. **Four times the area
roughly doubles the founding size needed**, so the threshold tracks density as well as count. It
grows sub-linearly with area, consistent with mate-finding over a scent range rather than with a
fixed density requirement. This is only measured for naive founders, and with 20 replicates per
point.

## What this means for #29 and #28

**Collapse detection (#29): the measured threshold is two.** Under today's rules, a population with
two or more animals, unborn young counted, is never doomed in the sense #29 means. It recovers
eventually in every replicate measured. A population that cannot ever produce two breeders is
already doomed by physics, since conception needs two. So #29's "no viable breeding population"
check reduces, for now, to *fewer than two living animals, counting gestating young*. That is one
step earlier than #226's "none left". Shipping a larger number would contradict the measurement.

**Cull guard (#28): the honest number depends on the horizon, so it is a design choice, not a
measurement.** If the guard is meant to stop a cull that leaves the species *at risk of failing to
regrow within ten generations*, the data say **12** (40/40 recovered, CI [0.91, 1.00]; 8 still
failed 5 times in 40). If it's meant to stop only *doom*, the answer is 2, as above. These are
different guarantees, and #28 should say which one it gives.

**Neither number survives #239.** Once senescence exists, a stalled pair ages out instead of
searching forever, and the bottleneck curve will grow an extinction column. That is the point at
which a threshold above two can appear. This spike should be re-run then. Until then, any threshold
above two would be a number the world doesn't produce.

## A bug found along the way: #240

The naive sweeps first aborted with `ValueError: cannot return more nutrients than have left the
field: returning 0.418 against 0.416 outstanding`, raised from `Ecology.spend` → `Plants.return_nutrients`
inside the tick loop. As a population starves out, the ledger shrinks to the last animals' energy
while the float32 rounding accumulated in the `energy` column doesn't. So **a world that starves
to extinction raises instead of collapsing.** It is filed as #240. The bench records these runs as
`crashed`: 15 of the 160 naive runs at n ≤ 6 on the 80-grid, and 5 of 40 on the 160-grid.

## Compute

| step | runs | wall time |
|---|---:|---:|
| establish 3 source worlds (1,500 ticks each) | 3 | 63 s |
| bottleneck, 10 generations | 520 | 397 s |
| bottleneck, 50 generations | 100 | 216 s |
| naive, grid 80 | 520 | 745 s |
| naive, grid 160 | 140 | 527 s |
| **total** | **1,283** | **~32 min** |

A tick at small population costs about 13 ms on the 80-grid. That cost is fixed per grid (plants,
cue field, forage), so small worlds are not proportionally cheaper.

## Not measured

- **Sex ratio.** There are no sexes (#99).
- **Bottleneck density sensitivity.** Only naive founders were run on the larger grid. Growing an
  established 160-grid world costs about 4× the 80-grid establishment.
- **Spatially clustered survivors.** Bottleneck survivors are chosen uniformly from the living, so
  they are as scattered as the herd was. A cull that leaves one tight group would recover faster.
- **More than three source worlds**, and any world other than the demo world.

## Reproducing

```
PYTHONPATH=. python docs/spikes/viable_population.py establish --out <dir>
PYTHONPATH=. python docs/spikes/viable_population.py sweep --out <dir> --mode bottleneck --reps 40
PYTHONPATH=. python docs/spikes/viable_population.py sweep --out <dir> --mode bottleneck --reps 20 --ns 2,3,4,6,8 --generations 50
PYTHONPATH=. python docs/spikes/viable_population.py sweep --out <dir> --mode naive --reps 40
PYTHONPATH=. python docs/spikes/viable_population.py sweep --out <dir> --mode naive --grid 160 --reps 20 --ns 2,4,8,16,32,64,200
PYTHONPATH=. python docs/spikes/viable_population.py summarise --out <dir>
```

The simulation is non-deterministic (§2.2), so expect the counts to move within the intervals
above, not to repeat.
