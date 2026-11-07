# The whole tick against §2.1's budget (#258, #174)

#258 had only ever timed `Behaviour.choose`. §2.1's 1,000 ms is the budget for the **whole** tick, so
this times every system in `TICK_ORDER`, on worlds built the way the game builds them
(`build_world` over `demo_world_config`). It also counts the phenotype reads #174 is about, now that
births already write genes inside the tick.

Measured on `master` at `62b33d0` (after #264 and #274) with
[`tick_budget.py`](tick_budget.py), on a 16-thread Windows machine with nothing else running (CPU
load 2–16% between runs; two runs started at 55–60% and are marked). Nothing in `core/` was changed:
the loop's `systems` tuple and `Genetics`' read methods are wrapped on the instance.

## 0. A world of 100,000 lasts about eight ticks

No map holds 100,000 animals. On either grid the founders begin starving at about tick 6, and the
population falls from there:

| tick | 256×256: acting | ms | 80×80: acting | ms |
|---:|---:|---:|---:|---:|
| 0 | 100,000 | 1,407 | 100,000 | 1,423 |
| 5 | 99,993 | 1,380 | 99,994 | 1,416 |
| 8 | 98,855 | 1,384 | 95,254 | 1,309 |
| 10 | 94,711 | 1,282 | 75,355 | 1,036 |
| 12 | 85,697 | 1,170 | 43,524 | 581 |
| 15 | 65,574 | 895 | 10,249 | 141 |
| 20 | 36,668 | 546 | | |
| 29 | 16,917 | 314 | | |

(`series` mode.) A median over a long run therefore measures a shrinking world. The first attempt
did exactly that and reported 700–740 ms "at 100k", over a median population of 52,135. The tables
below pool **ticks 1–8 of three fresh worlds** (seeds 0–2, 24 ticks), when the population is still
98,800–100,000. Tick cost tracks the acting population roughly linearly, and the grid-sized systems
add a constant on top.

**Grid sizes.** 256×256 is the grid `option_sampling_bench.Bench` uses. 80×80 is the demo world's
own (`_GRID`). The demo world is tuned for about a thousand animals, so on 80×80 even 10,000 is
far past what it can feed.

## 1. Per system, ms per tick (median over 24 ticks)

| system | 10k, 256² | 10k, 80² | **100k, 256²** | 100k, 256² (seeds 10–12)ᵃ | 100k, 80²ᵃ |
|---|---:|---:|---:|---:|---:|
| climate | 0.8 | 0.1 | 0.5 | 0.5 | 0.1 |
| plant_growth | 1.8 | 0.1 | 0.7 | 0.6 | 0.1 |
| cue_field_rebuild | 15.6 | 2.9 | 35.5 | 37.7 | 24.7 |
| forage_field_rebuild | 16.6 | 1.5 | 15.8 | 16.8 | 1.5 |
| carrion_scent_rebuild | 16.0 | 1.4 | 15.2 | 16.7 | 1.4 |
| **drive_scoring** | 78.5 | 75.9 | **913.0** | 951.5 | 924.7 |
| **movement** | 11.6 | 11.6 | **139.0** | 149.8 | 144.6 |
| exertion_recovery | 0.1 | 0.1 | 0.8 | 0.9 | 0.8 |
| feeding | 6.6 | 5.6 | 64.3 | 68.1 | 66.5 |
| drinking | 0.8 | 0.6 | 6.6 | 7.3 | 6.9 |
| predation | 2.1 | 6.6 | 83.6 | 89.0 | 121.1 |
| dehydration | 0.5 | 0.8 | 5.2 | 6.9 | 7.3 |
| metabolic_upkeep | 5.5 | 5.6 | 56.1 | 59.7 | 55.0 |
| death | 0.0 | 0.1 | 0.5 | 0.6 | 0.6 |
| decomposition | 0.5 | 0.2 | 1.7 | 1.7 | 0.2 |
| age_increment | 0.0 | 0.0 | 0.3 | 0.3 | 0.4 |
| conception | 4.2 | 4.3 | 46.1 | 50.5 | 48.0 |
| **sum of medians** | **161** | **117** | **1,385** | **1,459** | **1,404** |
| whole `advance(1)`, median (min–max) | 164 (149–194) | 117 (109–131) | **1,403 (1,312–1,501)** | 1,474 (1,382–1,516) | 1,396 (1,299–1,640) |
| share of 1,000 ms | 16% | 12% | **140%** | 147% | 140% |

ᵃ CPU load was 55–60% just before these two runs started. The 256² rows agree within 5%.

- **At 100,000 the tick is about 1.4 s, 40% over budget.** `drive_scoring` alone is 913–952 ms, the
  same as #258's 862–874 ms bench figure for `choose()`. The other sixteen systems add about
  470 ms: movement 139, predation 84–121, feeding 64, upkeep 56, conception 46, cue rebuild 36.
- **Even a `choose()` at zero would leave the 100k tick at about 470 ms.** That is in budget, but
  not by a margin that survives everything #258 still has to add (sight, #24).
- **At 10,000 the tick is 117–164 ms**, 12–16% of budget. The three grid-sized field rebuilds are
  about 48 ms of that on 256² and about 6 ms on 80², so on the large grid they are a third of a
  10k tick.
- **The breeding world is cheaper, not dearer.** At 10k on 256², ticks 150–169 (population 2,500–
  2,750, with births happening) cost 74 ms, of which conception is 1.2 ms. The table's 100k
  conception cost (46 ms) is in ticks 1–8, before any founder is mature (`maturity_age` is founded
  over 40–120 ticks). It is therefore almost entirely the `Conception.willing` phenotype read
  (§3), not pairing or allocation. No world in these runs had births *at* 100k, because none
  survives to the first maturity.

## 2. `Terrain.cell_indices`

At 100k on 256², ticks 1–4 without the profiler, then ticks 5–8 under cProfile (`profile` mode):

| | calls/tick | ms/tick | share of tick |
|---|---:|---:|---:|
| unprofiled, wrapped on the instance | 17 | 85.8 | 6.4% |
| cProfile, cumulative | 17 | 82.2 | 6.4% of profiled time |

By system (unprofiled): **drive_scoring 8 calls, 77.9 ms** (median 900,000 positions per call);
feeding 3 calls, 3.5 ms; cue rebuild, movement, upkeep, decomposition 1 each and predation 2, all
about a millisecond.

Which inputs (traced at 10k by hashing each call's `x, y`):

- In `drive_scoring`, **five calls are on the identical (n × 9) candidate block**: `Plants.forage_at`
  twice (forage and carrion scent, from `Hunger.appeal`), `Hydration.reachable_at` once
  (`Thirst.appeal`), and twice through `CueField.sample_excluding_source` → `sample` inside the one
  shared `perceived_at`. Each one is about 15 ms at 100k, so four are repeats worth about 60 ms.
- Three more calls are on the animals' own positions (`perceived_at`'s source term and
  `Fear.urgency`'s `sample_excluding_self`). These are identical to the cue rebuild's input, and
  each costs about 1 ms.
- After movement, five calls (movement's and upkeep's `return_nutrients`, feeding's two `graze`s and
  its `return_nutrients`) are on the identical post-move positions, about 1 ms each.

So `cell_indices` is about 6% of the tick. About 4% of the tick (~60 ms) is recomputing it on the
candidate block that `Candidates` already holds. The arithmetic itself is cheap. Each 900,000-
position call is a bounds check, two divides, two floors and two int64 casts.

## 3. Phenotype reads per tick (#174)

Every outermost call into `Genetics` that builds a phenotype, at 100k on 256², ticks 1–8 (`phenotype`
mode, wrapped on the instance). Rows are the median rows per call and cols the columns built.

| system | method | caller | calls/tick | rows | cols | ms/tick |
|---|---|---|---:|---:|---:|---:|
| cue_field_rebuild | `expressed_columns` | `Scent._broadcast` | 1 | 99,996 | 9 | 13.4 |
| drive_scoring | `expressed_columns` | `Scent._broadcast` | 3 | 99,996 | 9 | 40.7 |
| drive_scoring | `expressed_columns` | `Fear._channels` | 1 | 99,996 | 17 | 27.0 |
| drive_scoring | `expressed_columns` | `Fear.appeal` | 1 | 99,996 | 17 | 26.7 |
| drive_scoring | `expressed_columns` | `DriveWeight.of` | 6 | 99,996 | 1 | 25.4 |
| drive_scoring | `expressed_columns` | `Hunger.appeal` | 1 | 99,996 | 2 | 12.3 |
| drive_scoring | `expressed_columns` | `Lust.appeal` | 1 | 99,996 | 1 | 9.4 |
| drive_scoring | `expressed_columns` | `Thermal.appeal` | 1 | 99,996 | 3 | 8.4 |
| drive_scoring | `expressed_columns` | `Thermal.urgency` | 1 | 99,996 | 2 | 7.7 |
| drive_scoring | `expressed_columns` | `Behaviour.choose` | 1 | 99,996 | 2 | 7.5 |
| drive_scoring | `expressed_columns` | `Lust.urgency` | 1 | 99,996 | 1 | 4.8 |
| movement | `expressed` | `Movement.step` | 1 | 99,996 | 47 | 42.5 |
| feeding | `expressed` | `Feeding.feed` | 1 | 99,996 | 47 | 45.5 |
| feeding | `vigour` | `Feeding.feed` | 1 | 99,996 | — | 5.7 |
| predation | `expressed_at` | `Predation._wound` | 4 | 33,557 | 47 | 67.5 |
| metabolic_upkeep | `expressed` | `Ecology.upkeep` | 1 | 99,996 | 47 | 44.3 |
| conception | `expressed` | `Conception.willing` | 1 | 99,955 | 47 | 44.5 |
| **total** | | | **26** | | | **433.3** |

By system: cue rebuild 13.4, drive_scoring 169.9, movement 42.5, feeding 51.2, predation 67.5, upkeep
44.3, conception 44.5. **About 31% of the 1,403 ms tick is phenotype reads.** At 10k the same census
is 33.5 ms of a 164 ms tick (20%). No `heritable` read runs inside a tick. The metric recorder and
the debug invariants (`senescence_decay_at`) read between ticks, and `build_world` attaches neither.

Repeats within the 433 ms:

- `Scent._broadcast` builds the same 9 columns for the same population **4 times** per tick (54 ms):
  once in the cue rebuild and three times inside `choose()`.
- `Fear._channels` and `Fear.appeal` each build the same 17 columns (54 ms).
- Four systems build the full 47-column block for the whole living set: movement, feeding, upkeep,
  and conception. Conception reads one column of it (`maturity_age`), movement reads four (`size`,
  `haste`, `speed`, `agility`), and feeding reads `size` and the diet allocation.
- Predation's 67.5 ms is four `expressed_at` gathers of about 33,600 rows each (attacker and victim,
  twice), all 47 columns, for the `size` column and a few others.

What a single block would cost instead, at 97,329 rows:

| | ms |
|---|---:|
| one `expressed(living)`, 47 columns | 46.1 |
| one `heritable(living)` | 32.3 |
| one `vigour(living)` | 4.5 |
| slice 1 / 2 / 4 / 8 / 17 columns out of it | 0.19 / 0.38 / 1.24 / 2.53 / 5.80 |
| gather half the rows out of it | 2.92 |

**Rough saving from a once-per-tick block at 100k: about 300–350 ms per tick.** That is 433 ms of
reads replaced by one 46 ms build, plus about 25–35 ms of column slices and predation's row gathers,
plus whatever conception needs (§4: 0 ms if it reuses the block, or about 45 ms if it rebuilds after
`age_increment`). At 10k the same arithmetic gives about 28 ms. Most of the saving needs no cache:
narrowing the four full-block readers to `expressed_columns` and reading `_broadcast` and fear's 17
columns once each would remove much of it with nothing held across systems. This document does not
measure that split.

## 4. What changes mid-tick (facts for #174's gate)

Read from `core/world/assembly.py` (`TICK_ORDER`, `_build_systems`), `core/ecology/conception.py`,
`core/ecology/death.py`, `core/ecology/aging.py`, `core/genetics/service.py` and
`core/genetics/speciation.py` at `62b33d0`.

The phenotype of a row is a function of `genes`, `species_id` (the mask) and, for the decaying genes
(`speed`, `sight`, `scent_acuity` in the demo world) and `vigour`, of `age`.

1. **Phenotype readers, in order:** `cue_field_rebuild` (2), `drive_scoring` (5), `movement` (6),
   `feeding` (8), `predation` (10), `metabolic_upkeep` (12), `conception` (16, last). `climate`,
   `plant_growth`, the two field rebuilds, `exertion_recovery`, `drinking`, `dehydration`, `death`,
   `decomposition` and `age_increment` read none.
2. **Nothing reads phenotype after `conception` in the same tick**, because conception is the last
   system. Between ticks there are the debug invariants (`senescence_decay_at`), the metric recorder
   (`heritable`, at its cadence, attached by clients rather than by `build_world`), and
   `grow_if_crowded`, which can reallocate every store column at the boundary.
3. **Gene and `species_id` writes inside a tick:** only `Conception._gestate`, through
   `store.allocate(..., genes=..., species_id=...)`, at position 16. It writes **unborn** rows
   (negative `age`), which `living()` excludes, possibly into rows that `death` freed earlier in the
   same tick (position 13). `Genetics.set_genes` is called only by founding (`_found`).
   `Genetics.speciate` and `speciation.split` exist, but **no system calls them**. Speciation is not
   in `TICK_ORDER` (#16). Interventions run before the systems. Of the two that exist, cull releases
   rows and fence writes barriers. Neither writes genes.
4. **`age` changes at position 15** (`age_increment`). That changes the decayed phenotype and
   `vigour` of every living row. The only reader after it is `conception`, which reads
   `maturity_age`, and that gene does not decay.
5. **The living set changes mid-tick in two ways.** It shrinks at `death` (13), and no phenotype
   reader runs between `death` and `age_increment`. It grows at `age_increment` (15): a row going
   from age −1 to 0 becomes `living()`.
6. **A newborn's phenotype is read in its birth tick, once, by `Conception.willing`**, which builds
   the full block over `living()` after `age_increment` has made the newborn age 0. Its genes are not
   new that tick. They were written at its conception, a gestation term (20–60 ticks in the demo
   world) earlier. But the row was not in the living set at the start of the tick, so a block built
   at tick start would not contain it. Nothing else reads a newborn in its birth tick, because every
   other reader runs before `age_increment`.
7. So **from `cue_field_rebuild` to `metabolic_upkeep` (positions 2–12), `genes`, `species_id`,
   `age` and the living set are all constant** for every row a reader sees. The only exception is
   rows predation's wounds will later kill, which are still living until `death`. The one reader
   outside that window is `conception`, which sees advanced ages, newly born rows, and rows freed by
   death.

## 5. Reproducing

```
python docs/spikes/tick_budget.py systems   --n 100000 --grid 256          # §1, seeds 0-2
python docs/spikes/tick_budget.py systems   --n 100000 --grid 0            # 80x80
python docs/spikes/tick_budget.py series    --n 100000 --grid 256 --ticks 30   # §0
python docs/spikes/tick_budget.py systems   --n 10000 --grid 256 --warmup 150 --ticks 20 --rebuilds 1
python docs/spikes/tick_budget.py profile   --n 100000 --grid 256 --ticks 4    # §2
python docs/spikes/tick_budget.py phenotype --n 100000 --grid 256          # §3
```
