# How small can a population get and still come back?

Measured for #29, under the decision recorded there on 2026-11-06: **a viable breeding population is
calibrated offline by simulation**, and the measured number ships with the rule version. #28's cull
guard reads the same number.

> **Superseded in part, twice, on 2026-11-06.** The newest section, directly below, re-measures after
> #251 let a lone animal smell a mate. It is the one #29 and #28 should read. The senescence-era
> re-measurement and the original follow it unchanged, because each comparison is the evidence for
> what one change did.

## Re-measured 2026-11-06, after #251: a lone mate can be smelled

#251 widened the demo world's cue field from 3 to 10 and dropped lust's detection threshold from
0.05 to 5e-5, so one mate is perceived out to 26–28 units instead of not at all
(`docs/spikes/scent-range.md`). The section below predicted that "the small-n end of this curve
measures that sensing range more than it measures demography". This tests that prediction.

Same bench and definitions. Source worlds re-grown under the new config (snapshots refuse a foreign
config): 3,865, 1,448 and 2,530 born animals in 1,500 ticks, generation time 96–113 ticks.
**Compute was cut to fit the budget**: 20 replicates per n rather than 40, n ≤ 24 only (n ≥ 12 was
already 40/40), and the 50-generation follow-up for n = 2 and 3 only. The naive sweep was not re-run.

### Bottleneck, 10 generations

| n | runs | recovered | extinct | stalled | P(recover) | 95% CI | senescence-era (40 runs) |
|---:|---:|---:|---:|---:|---:|---|---:|
| 2 | 20 | 7 | 0 | 13 | 0.35 | [0.18, 0.57] | 0.00 |
| 3 | 20 | 18 | 0 | 2 | 0.90 | [0.70, 0.97] | 0.10 |
| 4 | 20 | 18 | 0 | 2 | 0.90 | [0.70, 0.97] | 0.17 |
| 6 | 20 | 20 | 0 | 0 | 1.00 | [0.84, 1.00] | 0.55 |
| 8 | 20 | 20 | 0 | 0 | 1.00 | [0.84, 1.00] | 0.78 |
| 12–24 | 20 each | 20 | 0 | 0 | 1.00 | [0.84, 1.00] | 1.00 |

### Bottleneck, 50 generations

| n | runs | recovered | extinct | stalled | P(recover) | 95% CI | senescence-era (20 runs) |
|---:|---:|---:|---:|---:|---:|---|---:|
| 2 | 20 | 15 | 2 | 3 | 0.75 | [0.53, 0.89] | 0.00 |
| 3 | 20 | 20 | 0 | 0 | 1.00 | [0.84, 1.00] | 0.25 |

**The prediction holds.** The small-n curve moved with the sensing range, and by a lot: a pair went
from never recovering in 60 runs to recovering three times in four within 50 generations.

### What this means for #29 and #28, now

**Collapse detection (#29): the measured threshold falls back to two born animals**, the biological
minimum for a sexual species. A pair recovers 15/20 within 50 generations, so it is not doomed. Below
two nothing can breed. As before, unborn young were not measured: each bottleneck released
pregnancies with the bystanders.

**Cull guard (#28): 6 for "regrows within ten generations"** (20/20, CI [0.84, 1.00]), down from 12.
Three and four recover 18/20. The CI is wider than the senescence-era 12's [0.91, 1.00] because of the
halved replicate count. If #28 wants the same confidence, re-run n = 4–8 at 40. The "only stop doom"
guarantee is 2.

**Re-measure when** #20's remaining scope (mate selection) or any MAJOR lands, in a predator-holding
world, or on a larger grid. A fixed ~27-unit reach covers a smaller share of a bigger map, so the
chance-encounter problem returns at lower density.

| step | runs | wall time |
|---|---:|---:|
| establish 3 source worlds | 3 | 170 s |
| bottleneck, 10 generations | 160 | 199 s |
| bottleneck, 50 generations (n = 2, 3) | 40 | 84 s |

## Re-measured 2026-11-06: with senescence and float64 energy

`master` at ab9df5e. That includes #239 (speed, sight and scent decay with age), #244 (digestion yield
decays with age), #240 (float64 energy), #176 (heat and cold tolerance), and #238 (temperature in
world units). Same machine, bench, definitions, n values and replicate counts as the original. 15
worker processes.

**Definitions are unchanged, and checked rather than assumed.** The original expected senescence
to turn "stalled" into "extinct". Within 10 generations it doesn't, for a measured reason: an old
animal alone on a field nobody else grazes banks enormous reserves, so it outlives the horizon. In a
probe of three n = 2 bottlenecks, the survivors held 500–1,200 energy at age 1,000–2,000, and every
one died of age between about 2,700 and 5,000 ticks old, with vigour down to 0.02–0.06. That is
5–9× the dense-world lifespan #244 measured (90th percentile of age at death about 571), because
nothing competes for the food. So "stalled" is kept, and the 50-generation follow-up is what
resolves it.

Source worlds grew to 1,391, 390 and 4,342 born animals in 1,500 ticks. Generation time is 114
ticks, so the horizon is 1,140 ticks.

### Bottleneck, 10 generations

| n | runs | recovered | extinct | stalled (bred) | P(recover) | 95% CI | before |
|---:|---:|---:|---:|---:|---:|---|---:|
| 2 | 40 | 0 | 0 | 40 (3) | 0.00 | [0.00, 0.09] | 0.12 |
| 3 | 40 | 4 | 0 | 36 (3) | 0.10 | [0.04, 0.23] | 0.23 |
| 4 | 40 | 7 | 0 | 33 (12) | 0.17 | [0.09, 0.32] | 0.35 |
| 6 | 40 | 22 | 0 | 18 (11) | 0.55 | [0.40, 0.69] | 0.75 |
| 8 | 40 | 31 | 0 | 9 (8) | 0.78 | [0.62, 0.88] | 0.88 |
| 12 | 40 | 40 | 0 | 0 | 1.00 | [0.91, 1.00] | 1.00 |
| 16–200 | 40 each | 40 | 0 | 0 | 1.00 | [0.91, 1.00] | 1.00 |

### Bottleneck, 50 generations (5,680 ticks)

| n | runs | recovered | extinct | stalled | P(recover) | 95% CI | before |
|---:|---:|---:|---:|---:|---:|---|---:|
| 2 | 20 | 0 | 6 | 14 | 0.00 | [0.00, 0.16] | 0.55 |
| 3 | 20 | 5 | 5 | 10 | 0.25 | [0.11, 0.47] | 0.95 |
| 4 | 20 | 14 | 2 | 4 | 0.70 | [0.48, 0.85] | 0.95 |
| 6 | 20 | 18 | 1 | 1 | 0.90 | [0.70, 0.97] | 1.00 |
| 8 | 20 | 19 | 0 | 1 | 0.95 | [0.76, 0.99] | 1.00 |

**Extinctions now happen**: 14 of 100 long-horizon runs, against 0 of 620 before. The n = 2 runs
still "stalled" at 5,680 ticks had not bred once (none exceeded their starting size). They are pairs
ageing out on a full field, and by the probe above they die within a few thousand more ticks. So
**n = 2 never recovered in 60 runs (0/40 and 0/20)**. Before senescence it recovered 11 times in 20
over the same 50 generations, by outliving the wait for a meeting.

**Why the small-n curve is so flat: the pair never meets.** In the probe, both animals were mature
and well fed, and stayed 15–57 units apart for thousands of ticks without coming within the contact
range of 2. Mate-finding steers only by the scent cue field, whose demo-world diffusion range is
3.0. Beyond that nothing moves a lone animal toward others. This is filed as #251. **The small-n
end of this curve measures that sensing range more than it measures demography**, so it moves if
#251 is resolved.

### Naive founders, 10 generations

| n | runs | recovered | extinct | crashed | stalled (bred) | P(recover) | 95% CI | before |
|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 2 | 40 | 0 | 22 | 0 | 18 (2) | 0.00 | [0.00, 0.09] | 0.00 |
| 3 | 40 | 0 | 15 | 0 | 25 (0) | 0.00 | [0.00, 0.09] | 0.00 |
| 4 | 40 | 0 | 6 | 0 | 34 (6) | 0.00 | [0.00, 0.09] | 0.00 |
| 6 | 40 | 0 | 10 | 0 | 30 (4) | 0.00 | [0.00, 0.09] | 0.00 |
| 8 | 40 | 0 | 1 | 0 | 39 (14) | 0.00 | [0.00, 0.09] | 0.00 |
| 12 | 40 | 0 | 1 | 0 | 39 (26) | 0.00 | [0.00, 0.09] | 0.03 |
| 16 | 40 | 1 | 0 | 0 | 39 (28) | 0.03 | [0.00, 0.13] | 0.03 |
| 24 | 40 | 3 | 0 | 0 | 37 (36) | 0.07 | [0.03, 0.20] | 0.10 |
| 32 | 40 | 5 | 0 | 0 | 35 (32) | 0.12 | [0.05, 0.26] | 0.45 |
| 48 | 40 | 7 | 0 | 0 | 33 (32) | 0.17 | [0.09, 0.32] | 0.60 |
| 64 | 40 | 13 | 0 | 0 | 27 (26) | 0.33 | [0.20, 0.48] | 0.88 |
| 100 | 40 | 27 | 0 | 0 | 13 (11) | 0.68 | [0.52, 0.80] | 0.93 |

Naive founders now cross P = 0.5 at **about n = 64–100**, against 32–48 before. Founders age from
the moment they are placed and now take in less as they do, so an unselected founding cohort has
less time to sort itself out before it breeds. **#240's crash outcome is gone: 0 crashes in 520
naive runs**, against 15 in 160 at n ≤ 6 before. The worlds that used to raise now go extinct
cleanly, which is part of why the extinct column grew (22 at n = 2, against 7 extinct plus 6 crashed).

The grid-160 density check was not re-run (compute budget, below).

### What this means for #29 and #28, now

**Collapse detection (#29): the measured threshold rises from two to three born animals.** A
population of two never recovered in 60 runs (upper 95% bound 0.16 at 50 generations, 0.09 at 10).
Three recovered 5 times in 20 within 50 generations, so three is not doomed. Two caveats belong
with the number:
- **Unborn young were not measured.** Every bottleneck released pregnancies with the bystanders, so
  the number counts *born* animals only. Whether a gestating young should count toward the three is
  #29's call. The original section's "counting unborn young" was a reasoned extension of "two can
  breed", not a measurement.
- **It is a sensing-range artefact as much as a demographic one** (#251). Re-measure when #251
  lands.

**Cull guard (#28): unchanged at 12** for the "regrows within ten generations" guarantee (40/40, CI
[0.91, 1.00]). Below that, n = 8 now fails 9 times in 40, against 5 before. The "only stop doom"
guarantee moves with #29's number, from 2 to 3. As before, #28 should say which guarantee it gives.

**Re-measure when** #251 lands, when #20's remaining scope (mate selection) or any MAJOR (§2.8)
lands, or in a predator-holding world.

### Compute (re-measurement)

| step | runs | wall time |
|---|---:|---:|
| establish 3 source worlds | 3 | 72 s |
| bottleneck, 10 generations | 520 | 365 s |
| bottleneck, 50 generations | 100 | 554 s |
| naive, grid 80 | 520 | 678 s |
| **total** | **1,143** | **~28 min** |

The 50-generation sweep cost 2.6× the original's, because pairs that once stalled cheaply now live
long, banked lives and are simulated to the horizon.

---

# Original measurement (5fe0c75, before senescence)

The headline result of the original measurement was that **under the rules at the time there was no
such number above two.** An established population knocked down to any size of two or more never
died out in 620 runs. Small bottlenecks were slow, not doomed, and the reason was two missing
mechanics rather than anything about small populations. What *did* have a threshold was a
population of naive founders, and that is a different question.

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
