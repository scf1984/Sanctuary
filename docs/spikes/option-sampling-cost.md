# Spike: what option sampling costs per tick

Tracks issue #114.

## Status: measured

## Why

#114 replaces per-entity drive scoring with per-option scoring, and states the cost as
`N candidates × n_drives × n_entities` field reads per tick — "the largest single cost multiplier
in the tick", to be **benchmarked rather than asserted** (CLAUDE.md §8.5).

Two design decisions rest on the answer:

- **Candidate positions are sampled once per entity and shared by every drive.** If per-drive
  sampling were cheap, the shared block would be premature complexity.
- **More angular resolution comes from raising `N`, not from a two-stage refinement pass** (#114's
  "Rejected" section). That answer is empty if `N` is a knob nobody can afford to turn up.

## Method

`docs/spikes/option_sampling_bench.py` is throwaway spike code (CLAUDE.md §8.3) — not part of
`core/`, not imported by anything else.

It builds a world with four of the five drives — hunger, thirst, lust and fatigue — and times
`Behaviour.choose()` over the whole population. **Fear is excluded deliberately**: its cost is the
cue field, which #22 already measured and which #114 does not change, since its `appeal` is a
constant. Including it would have measured #22.

Terrain is a fixed 256×256 grid with `forage_diffusion` range 8.0 at every population size, so the
field cost is constant and the population term is isolated. Each figure is the **median of 7 timed
runs after a warmup run**.

Measured on Windows 11 / Python 3.12.10 / NumPy 2.5.1. Two independent runs; the tables below are
run 1, with run 2 agreeing to within 4% at every cell except n=1,000, where the whole measurement
is one field rebuild and run-to-run noise is ±30%.

## Results

### `choose()` in full, ms/tick

| n | N=4 | N=8 | N=16 | N=32 |
|---:|---:|---:|---:|---:|
| 1,000 | 113.9 | 110.2 | 123.5 | 113.6 |
| 10,000 | 121.7 | 124.7 | 156.3 | 197.2 |
| 100,000 | 290.2 | 361.6 | 518.7 | 811.1 |

### Where the time goes, at N=8 (nine options), ms/tick

| n | diffusion | positions | field reads | flat drives | Boltzmann draw | sum | `choose()` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 144.5 | 0.3 | 0.1 | 0.2 | 0.3 | 145.4 | 110.2 |
| 10,000 | 123.7 | 3.7 | 1.8 | 1.8 | 3.3 | 134.3 | 124.7 |
| 100,000 | 112.9 | 48.6 | 30.4 | 23.2 | 37.9 | 253.0 | 361.6 |

The sum is the parts timed individually, so it does not have to match `choose()`: at n=1,000 the
whole measurement is one field rebuild and the difference is noise, and at n=100,000 the ~109 ms
gap is the work not broken out — `Genetics.expressed`, the per-drive accumulation loop, `_record`,
and the column writes.

### The slope in `N`, ms per 1,000 entities

| n | per additional candidate | fixed floor at N=4 |
|---:|---:|---:|
| 1,000 | −0.01 | 113.9 |
| 10,000 | 0.27 | 12.2 |
| 100,000 | 0.19 | 2.9 |

## What this says

**The forage field dominates, and option sampling does not.** `Plants.forage_field()` costs
110–145 ms per tick and **does not depend on the population at all** — it is a diffusion over the
grid. At every population up to 10,000 it is essentially the whole of `choose()`. #114's stated
worry, that the option multiplier would be the largest cost in the tick, is **wrong at these grid
sizes**: the largest cost is a field rebuild that would be paid with or without option sampling.

**`N` is affordable.** 0.19 ms per candidate per 1,000 entities at n=100,000. Doubling from 8 to 16
costs 157 ms at 100,000 entities and 32 ms at 10,000. So "raise `N` and rely on per-entity jitter"
is a real answer to angular resolution and not a deflection, which is what #114 needed in order to
reject two-stage refinement.

**Sharing candidate positions across drives is worth what it costs.** Generating and clipping them
is 48.6 ms at n=100,000 — the single largest option-scaled term, larger than the field reads it
feeds. Sampling per drive would pay it four times over, ~195 ms, for an identical result.

**Against §2.1's budget.** One tick is one real second at the live rate. At 100,000 entities and
N=8, `choose()` is 362 ms — 36% of the tick, and the largest single system, but affordable. At
10,000 entities it is 125 ms, of which 124 ms is the field.

## Consequences filed

- **The forage field is rebuilt inside `Hunger.appeal`, once per drive that reads it** rather than
  once per tick. Today only hunger reads it, so the cost is paid once and these numbers stand. A
  second field-reading drive would pay the full 110 ms again — see #170.
- The field cost is a property of the diffusion, not of this issue, and it is where any real
  optimisation of the tick should start.

## Re-running this after #100

#100 made the change-aversion constant the `commitment` gene, so the harness gained a sixth gene
and `choose()` now reads the phenotype block once for two genes instead of once for one. The tables
above were **not** re-measured: an A/B of `choose()` at n=100,000, N=8, alternating between the two
trees three times each on one machine, put the best-case call at 359.2 ms before and 361.9 ms after
— under 1%, against a `forage_field()` control that itself swung 111–163 ms across the same runs.
The machine that produced those A/B numbers was running roughly 30% slower overall than the one
these tables came from, which is why absolute figures from a re-run will not match; compare within
a run, not across.
