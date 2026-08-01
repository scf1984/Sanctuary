# Conception: does the world breed, and what binds it?

---

# ⚠️ Superseded — re-measured under #200

Measured while `speciation_threshold` was rejecting 94% of candidate couples (#199), so the rates
below are roughly **24 times too low**. Corrected figures, with the gate open:

| | here | corrected |
|---|---:|---:|
| conceptions | 53 per 900 ticks per 200 animals | 22,588 per 3,000 ticks (seed 1) |
| peak allocation in a tick | — | 3.25% of occupancy |
| peak concurrent gestation | 1–6 rows against 200 | 35% of occupancy |

The **capacity-bound** conclusion below still stands for the world as it was then, and is exactly
why the figures were wrong: free rows sat at zero throughout, so the run was measuring an array and
not an ecology. It measured a *second* limit at the same time without knowing it.

See `docs/spikes/capacity-growth.md` for the corrected numbers and
`docs/spikes/capacity_rate_bench.py` for the tool.

---


Measured while implementing #20's conception slice. The second question is #127's, and this is where
its number finally comes from.

## Does it breed

`demo_world_config(200 founders, seed 0)`, 900 ticks, invariants on.

| | |
|---|---|
| distinct entities ever allocated | **253** (200 founders + 53 conceptions) |
| young carried to term and living at the end | **52** |
| youngest living age | 8 ticks |
| nutrient drift | **1.8e-16** |

`core.genetics.inheritance` has existed since #14 and this is the first time anything running has
called it. Every gene in every previous world was a founder's draw, unchanged.

## What binds it: capacity, not ecology

| tick | living | gestating | free rows |
|---:|---:|---:|---:|
| 0 | 200 | 0 | 0 |
| 300 | 194 | 6 | 0 |
| 600 | 196 | 4 | 0 |
| 900 | 199 | 1 | 0 |

**Free rows are zero at every sample.** The store is sized to hold exactly the founders (§2.3:
"the store is sized to hold exactly them; population is emergent from there"), so a world that
breeds is immediately at its ceiling and `Conception.conceive` truncates to `store.available`.

That truncation is deliberate rather than a fallback — `EntityStore.grow` may only run at a tick
boundary (§2.3) and conception is mid-tick, so a world short of rows conceives fewer young rather
than raising. But it means **the population plateau visible here is an array size, not a carrying
capacity**, which is exactly the confusion CLAUDE.md §2.3 wanted to avoid and #180 corrected the
wording for.

So this run does not measure ecology. It measures that the ecology wants more room than it has.

## What #127 gets from this

The trigger it was waiting for. Its own body says the figure has to be "chosen with a measurement of
a breeding world in hand", and there was no breeding world until now.

What this run supplies:

- **Conceptions run at roughly 53 per 900 ticks per 200 animals** — about 0.03% of the population
  per tick — while capacity is *clipping* them, so it is a floor on the true rate rather than an
  estimate of it.
- **Gestating rows are held for their whole term**, so the reserve must cover concurrent pregnancies
  and not just one tick's births. Here that is 1–6 rows against 200, with `gestation_length` drawn
  from (20, 60) ticks.
- The high-water mark is the population plus the pregnancies, and both grow together.

The rate should be re-measured once capacity no longer clips it, which is #127's own first task.

## Not measured

§2.1's generation time — "roughly one real day", about two sim-months — cannot be checked while the
population is capacity-bound, for the same reason #183 records about feeding events: the quantity
describes a population that grows freely, and this one cannot. It is a #127 follow-on.
