# Pace and momentum: what an animal spends to hurry, and what it costs to turn

Issues #203 and #204. Everything below is measured on `feat/203-pace-and-momentum` against
`28a52ee` (master), with `clients/viewer/demo_world.py` unless stated.

---

## 1. The defect, in one number

`Movement.step` took a **scalar** pace, and the only production caller passed
`config.movement.walking_pace`. So `haul_rate = transport_cost × (1 + exertion_premium × pace)`
had a constant second factor, and the walk/sprint ratio §2.5 describes was machinery nothing could
exercise. No animal had ever hurried; `exertion_premium` could have held any value without a world
behaving differently.

Momentum was absent in the same way but louder: there was no velocity column at all, so an animal
at full speed could reverse for free and a pursuit was an arrival.

---

## 2. What an animal actually wants: the urge distribution

`choice_urge` is the summed drive advantage of the chosen option over standing still, which
`Movement.pace` converts into a fraction of top speed. Measured over the living population of
`build_demo_world(seed=1, n_entities=200)`:

| tick | population | urge p50 | p90 | p99 | p99.9 | max | share with urge > 0 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 207 | −0.082 | +0.004 | +0.477 | +0.658 | +0.673 | **10.6%** |
| 300 | 1,086 | −0.745 | 0.000 | +0.074 | +0.150 | +0.173 | 2.6% |
| 500 | 3,664 | −0.842 | 0.000 | 0.000 | +0.158 | +0.238 | 0.8% |
| 700 | 5,665 | −0.860 | 0.000 | 0.000 | +0.101 | +0.117 | **0.5%** |

Broken down per drive at tick 60, as the advantage of the **argmax** option over resting:

| drive | mean advantage | p90 |
|---|---:|---:|
| hunger | +0.0125 | +0.0395 |
| lust | +0.0203 | +0.0844 |
| thirst | 0.0000 | 0.0000 |
| fear | 0.0000 | 0.0000 |
| fatigue | **−0.0601** | 0.0000 |

That reads correctly and is worth stating plainly:

- **Only hunger and lust have a direction.** Thirst and fear are flat by construction (§2.5 records
  why: nothing drinks, and flight is #24's), so they contribute exactly zero advantage — a flat
  drive shifts every option equally and cancels in the difference.
- **Fatigue is the cost of moving**, and it is the only systematically negative term. An animal
  that moves against it wanted the move *less* than one with no such reason to stay. That is the
  right reading, and it is why the urge is net rather than best-of.

---

## 3. The mechanism is correct and nearly inert, and the reason is #205

The urge collapses as the world fills. That is not the pace rule failing; it is the drive contest
being **barely better than a random walk at the founding `choice_temperature`** — animals take
their best option 13–27% of the time against 11% by chance, so the option they *did* take is
mostly noise and its advantage over resting is negative for the median animal.

Filed as **#205** with its own measurements. It suppresses this mechanism, and it will suppress
#24's flight and #179's chase in exactly the same way, so it is worth settling before either.

The consequence here: pace varies meaningfully in a **sparse** world (10.6% of animals hurrying at
tick 100, paces up to 0.94) and hardly at all in a settled one. Since #101 evolves starting states
from small founder populations, and a player's world is sparse before it is crowded, the mechanism
is not dead — but it is much quieter than the design intends.

| tick | pace p50 | p90 | p99 | p99.9 | max |
|---:|---:|---:|---:|---:|---:|
| 100 | 0.400 | 0.405 | 0.816 | 0.919 | **0.940** |
| 300 | 0.400 | 0.400 | 0.462 | 0.615 | 0.697 |
| 700 | 0.400 | 0.400 | 0.400 | 0.492 | 0.580 |

`walking_pace = 0.4` is the floor, reached by any animal whose chosen option was worth no more than
standing still.

---

## 4. Choosing the `haste` founding range

`pace = 1 − (1 − walking_pace) × exp(−haste × urge)`, and `haste` is read through `exp`, so the
stored gene is a log-scale. Picked against §2 rather than by feel: at a p99 urge of ~0.2,

| haste | pace at urge 0.2 | pace at urge 0.5 |
|---:|---:|---:|
| 1 | 0.44 | 0.53 |
| 2 | 0.50 | 0.62 |
| 4 | 0.73 | 0.92 |

Founded at **(0.0, 1.4)** — haste 1 to about 4 — so founders span "barely hurries" to "hurries
readily" and selection has a gradient to work on from the first generation. A narrower range
around 1 would leave the gene doing nothing at the urges this world produces, which is the
neutral-gene trap §2.5 warns about.

---

## 5. What momentum costs in effective travel

The largest behavioural consequence, and it was not the one expected. Median velocity as a fraction
of top speed, against a median *desired* pace of 0.400:

| tick | seed 1 | seed 2 |
|---:|---:|---:|
| 200 | 0.225 | 0.232 |
| 600 | 0.220 | 0.207 |
| 1200 | 0.223 | 0.202 |

**An animal travels at roughly 55% of the speed it asked for.** Not because agility is scarce —
founders carry 0.3–0.9 world units per tick per tick against top speeds of 1–3, enough to reach a
walking velocity in two or three ticks — but because headings are re-drawn and jittered every tick
(#114) and `commitment` is small (0.05–0.25), so a wandering animal spends much of its
acceleration budget turning rather than getting anywhere.

That is realistic and it is a real change: the world got slower, and the same energy now buys less
ground. It is also a second, quieter reading of #205 — an animal whose heading is mostly noise
cannot build speed in any direction.

---

## 6. The population still works, and it is larger

Same seeds, same config, 1,200 ticks, master (`28a52ee`) against the branch. Living population:

| tick | master s1 | branch s1 | master s2 | branch s2 |
|---:|---:|---:|---:|---:|
| 0 | 200 | 200 | 200 | 200 |
| 600 | 3,630 | 5,083 | 4,698 | 5,991 |
| 1200 | **4,394** | **5,833** | **5,750** | **6,513** |

Median energy lands within 1% of master on both seeds (37.2 → 38.0, 37.3 → ~38), so this is a
larger population at the same individual condition: **carrying capacity rose 13–33%.**

That follows directly from §5 rather than being a surprise. Animals travel at 55% of the speed
they ask for, so each spends less on locomotion per tick, and the field supports more of them.
Whether that is the world we want is a tuning question for the cost table (§2.5), not a defect in
momentum — and it is exactly the kind of shift §2.8 makes MAJOR.

No invariant trips over any of these runs, including the new `no_entity_exceeds_its_top_speed`.

---

## 7. Throughput

`docs/spikes/movement_bench.py`, best of five `Movement.step` calls over a scored population, same
machine, both branches. The script runs unchanged on either side — it builds the call from the
config's own fields, since `step` took a scalar pace before #203 and a per-entity urge after it.

| entities | master | branch | change |
|---:|---:|---:|---:|
| 1,000 | 3.8 ms | 2.6 ms | −32% |
| 10,000 | 23.8 ms | 23.0 ms | −3% |
| 100,000 | 280.5 ms | 257.4 ms | −8% |

**No regression**, which is the only claim being made — the 1,000-entity figure is dominated by
fixed per-call overhead and the differences at scale are within run-to-run noise on this machine.
The change adds two float32 columns (0.8 MB at 100,000 entities) and about ten whole-array
operations per step, against a walk that already iterates cell crossings; against §2.1's 1,000 ms
tick budget, movement at 100,000 entities remains a quarter of it.

---

## 8. What was rejected

| option | why not |
|---|---|
| **pace from the chosen option's total utility** | utilities are shift-invariant, so an absolute reading carries a component no decision depends on. Measuring against the null option is the same quantity with that component removed. |
| **each drive declares its own pace** | #114 removed the single winning drive, so there is nobody to ask. Combining per-drive paces by contribution share needs a rule for negative contributions and a config number per drive, and hunger would want different paces for grass and for prey. |
| **`haste` as a measured constant rather than a gene** | "how much advantage is a lot" depends on the drive weights an animal carries, which are genes (#23). There is no world-level fact to measure. |
| **including the commitment bonus in the urge** | it would make an animal hurry for already being in motion and dawdle toward real danger for having settled: hysteresis (#100) turned into an accelerator. |
| **agility derived from size alone, no gene** | free physics, but it removes the speed-against-nimbleness axis entirely — a big animal would be slow to turn *and* nothing could evolve out of it. |
| **agility as a gene alone, not divided by size** | keeps the axis but throws away the free physics, and leaves `size` with no downside beyond upkeep. `agility / size` keeps both. |
| **capping velocity at top speed** | §2.5 rejected a speed cap outright and nothing here needs one: the bound holds by induction because velocity is written from the displacement that happened. Asserted in the invariant harness instead (§6, §8.2). |
| **stopping dead when an animal chooses rest** | a free brake — sprint one tick, stop the next, pay nothing. It is also a branch on a resting state, which #114 deliberately does not have. Braking at agility is the same rule as turning. |
