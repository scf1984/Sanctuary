# Capacity growth: the reserve, and a ceiling that turned out to be a bug

---

# ⚠️ Superseded in part — re-measured under #200

**Everything below was measured while `speciation_threshold` was suppressing reproduction 24-fold**
(#199). The reserve conclusion survives; the population curve does not.

## What survives

`reserve_fraction = 0.1` is **vindicated, with a thinner margin than it appeared to have.**
Re-measured with the gate open (`docs/spikes/capacity_rate_bench.py`, two seeds, 1,200–3,000 ticks):

| | throttled | corrected |
|---|---:|---:|
| peak allocation in a tick | 0.43% of occupancy | **3.25%** |
| 99th-percentile allocation | — | 1.8% |
| peak concurrent gestation | 19% of occupancy | **35%** |
| conceptions, seed 1, 3,000 ticks | — | 22,588 |

**Ticks on which conception ran out of rows: 0. Ticks clipped: 0.** The silent failure #200 was
filed to look for is not happening — but the reserve clears the worst observed tick by a factor of
three rather than the twenty the original figure implied.

## What does not survive: "the first world that found its own ceiling"

The claim below — that the population overshoots to 5,212, falls back to 4,761, and settles — was
an artefact of the throttle. With the gate open the population **does not settle at all** within the
observed window:

| tick | living (seed 1) | free rows | capacity |
|---:|---:|---:|---:|
| 900 | 4,372 | 5,967 | 10,560 |
| 1,500 | 5,487 | 4,741 | 10,560 |
| 2,100 | 7,013 | 3,275 | 10,560 |
| 2,700 | **7,881** | 2,407 | 10,560 |

Monotonic, no plateau, and free rows falling toward the next doubling. So the correct statement is
narrower than the one below:

- **The array is no longer binding.** That part holds — free rows are plentiful throughout and
  nothing is pinned against capacity, which is what #127 set out to fix.
- **The ecological carrying capacity is above 7,881 and has not been observed.** Crude arithmetic
  on this field's regrowth puts it far higher; finding it is a soak-test question (#48), not a
  1,200-tick spike.

The lesson is the same one #199 recorded: a plateau was measured, and it was the plateau of a bug.

---


Measured while implementing #127. That issue refused to pick a trigger without "a measurement of a
breeding world in hand", and there was no breeding world until #191.

## What the trigger has to clear

Growth happens **between** ticks (§2.3), and a tick that runs short of rows conceives fewer young
rather than raising — so a reserve that is too small does not crash anything, it silently caps
births. The reserve therefore has to cover at least one tick's allocation.

Measured on the demo world with headroom applied by hand, 3,000 ticks
(`docs/spikes/conception-and-capacity.md` covers the run):

| | |
|---|---|
| conceptions over 3,000 ticks | 3,531 |
| steepest growth | 1,033 → 3,187 living over 250 ticks |
| allocation at that rate | ~8.6 rows/tick against ~2,000 occupied — **0.43%/tick** |
| peak concurrent gestating | 608, **19% of occupancy** |

`reserve_fraction = 0.1` is roughly twenty ticks of runway at the steepest rate observed, and since
`grow` doubles, reaching it is rare.

**Measured against occupancy, not capacity.** A mostly-empty store must not keep growing after a
die-off: free rows would be plentiful in absolute terms while the ratio against capacity still read
low.

## The result: a population that is no longer array-bound

Demo world, 200 founders, growth enabled, 1,500 ticks.

| tick | living | gestating | free | capacity |
|---:|---:|---:|---:|---:|
| 0 | 200 | 0 | 20 | 220 |
| 250 | 223 | 6 | 211 | 440 |
| 500 | 324 | 37 | 79 | 440 |
| 750 | 1,033 | 287 | 440 | 1,760 |
| 1,000 | 4,246 | 582 | 2,212 | 7,040 |
| 1,250 | 5,212 | 127 | 1,701 | 7,040 |
| 1,500 | 4,761 | 166 | 2,113 | 7,040 |

Five doublings, 220 → 7,040. And then the thing that had never happened before: **the population
overshot to 5,212, fell back to 4,761, and free rows stayed available throughout.** Nothing is
pinned against the array any more — what limits this population is the food.

That is §2.3's requirement met rather than asserted: "as capacity is approached, density-dependent
mortality intensifies... so the plateau reads as ecology". Here it reads as ecology because it *is*
ecology; the array is following the population rather than bounding it.

Compare the same world before this change: living pinned at exactly 3,200 of 3,200 capacity from
tick 1,000 onward, gestating suppressed to single digits. That plateau was an array size.

## Two consequences worth knowing

**A world is now born with a reserve.** `build_world` sizes the store to the founders *plus* their
reserve. Sized to the founders exactly, a store has zero free rows and `Conception` truncates to
what is available — so the world was born sterile until the first tick boundary grew it.

**`World.founders` does not survive a growth.** A `Selection` is a mask over a store of a particular
capacity, so after a doubling its mask is shorter than the columns it would index. That is a
snapshot behaving like a snapshot (§4) rather than a wart, and the founders stop being a meaningful
population the moment anything is born — but it is a real trap for a caller that holds one across
`advance()`, so it is documented where the field is declared.

## Not measured

The **true** carrying capacity. The run above is still climbing to it — 1,500 ticks is enough to
show the array stop binding, not enough to show where the food stops. §2.5's carrying capacity is
area × primary productivity ÷ per-animal upkeep, and the crude arithmetic on this world's field
(~170,000 energy units of regrowth per tick against roughly 1.2 per animal) puts it far above 5,000.
That is a longer run than this spike, and it wants the soak-test harness (#48).
