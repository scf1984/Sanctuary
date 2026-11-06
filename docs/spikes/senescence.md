# Senescence on the demo world: what it shortens, and what it cannot kill

Measured while implementing #239. Bench: [`senescence_lifespan.py`](senescence_lifespan.py), demo
world (`demo_world_config(200, seed)`), seeds 1–5, 4,000 ticks each. The baseline is `master` at
6f6e978 (after #176's thermal genes, before senescence), where nothing ages.

Deaths are read from the store: a row whose id changed or emptied across a tick held an animal that
died in it, at the age it had reached. Unborn rows (negative age, #20) are excluded.

## Results

Means over the five seeds, with each seed's range for the noisy columns:

| `base_decay` | median age at death | share of deaths past age 1,000 | oldest alive at tick 4,000 | population at tick 4,000 |
|---|---:|---:|---:|---:|
| none (baseline) | 319 | 4.6% (2.3–6.6%) | 2,279–2,938 | 3,518–4,452 |
| 0.001 | 305 | 3.7% (0.3–6.7%) | 1,487–3,433 | 3,849–4,397 |
| **0.002 (shipped)** | 288 | **1.9%** (0.6–3.5%) | 2,016–3,849 | 3,742–4,910 |
| 0.004 | 264 | 1.1% (0.3–2.7%) | 780–**4,000** | **2,507**–4,747 |

At a founder's typical resistance (0.25, mid-band), `base_decay = 0.002` leaves an animal ~20% of
its speed and senses at age 1,000 — §2.1's lifespan row is ~6 generations, and a demo generation is
~113 ticks (`docs/spikes/viable-population.md`).

**What it does.** Most deaths are not old age on this world: the median animal dies of other causes
(starvation, thirst, competition) at ~300 ticks with or without senescence, so the median barely
moves. Senescence acts on the tail, as it should: 0.002 cuts the share of deaths past age 1,000 by more than half, and viability
is unchanged (every seed ends between 3,742 and 4,910 animals, against 3,518–4,452 before). 0.004
cuts the tail further, to 1.1%, but dropped one seed to 2,507 animals, so it was not taken.

## What it cannot kill: an animal that never needs to move

The "oldest alive" column does **not** fall with the decay rate, and at 0.004 seed 1 ended with a
founder that had lived the entire run. Inspected at tick 4,000 (seed 1, `base_decay = 0.004`), the
59 animals aged 1,500 or more look like this:

```
row  58  age 4000  decay 1.2e-07  energy  52.1  velocity 0.0000  speed now 2.5e-07  dehydration 0.006
row 146  age 3784  decay 5.1e-06  energy 115.0  velocity 0.0000  speed now 1.5e-05  dehydration 0.006
row 230  age 1780  decay 6.6e-03  energy 110.3  velocity 0.0000  speed now 1.4e-02  dehydration 0.006
row 444  age 2604  decay 6.3e-04  energy  42.4  velocity 0.0005  speed now 1.3e-03  dehydration 0.006
```

Motionless, fed and watered. §2.5's premise is that an old animal "eventually cannot cover its own
upkeep", and death then falls out of starvation — but that only holds for an animal whose intake
depends on a decaying trait. One that came to rest on a cell with grass and water at hand needs no
speed and no acuity to keep eating, and its atrophied traits make its upkeep *cheaper* (§2.5's
"consistent consequence"). Senescence slows it to nothing and it lives on.

This is not tuned around. Which traits decay, and whether something other than performance should
degrade, is the open half of #21's design note, and the finding is filed for that decision.

## Not locked in by a statistical test

§8.1 asks for tuning to be locked in before merging. The mechanism is (unit tests: the curve,
where it applies, the invariant, an assembled world's top speed falling with age). The tail
reduction is not: it needs ~4,000 ticks at ~4,000 animals per seed (6–12 minutes each) to measure,
the per-seed spread overlaps the baseline's, and the effect it would pin is about to be revisited
by the filed question. A test asserting it now would be slow and would pin a number expected to
change.
