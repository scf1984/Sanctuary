# Grazing: what intake rate makes a naive founder population viable?

Measured while implementing #19 (feeding). The question is not "what is realistic" but the narrower
one CLAUDE.md §8.1 asks of ecological tuning: explore, then lock the shape in so it cannot silently
regress.

## The constraint, which is temporary and stricter than it looks

Feeding introduces the first gene that decides whether an animal can **eat**: `diet_animal_derived`,
an allocation on [0, 1] whose plant efficiency is `(1 − a)²` (#102).

But nothing dies (#21) and nothing breeds (#20), so **selection cannot move the diet distribution.**
A founder population keeps whatever allocation it was drawn with, for the life of the world. The
intake rate therefore has to carry animals that are badly allocated *by chance*, not merely the ones
a few generations of selection would have produced.

The demo world draws `diet_animal_derived` from `(-1, 1)`, which the logistic squash reads as
allocations spanning **0.28 – 0.73** around a mean of 0.48. Plant efficiency therefore spans
**0.073 – 0.52**: the best founder is about seven times better at eating grass than the worst, and
neither can do anything about it.

## Measurement

`demo_world_config(200 founders)`, **3 seeds x 2,500 ticks**, 256x256 grid. Windows 11 / Python
3.12.10 / NumPy 2.5.1. "frac fed" is the share of the population holding energy above zero at the
end; "biomass" is the field mean, against roughly 130 for an ungrazed field. Founding energy is 180.

| `intake_rate` | frac fed | median energy | mean energy | field biomass |
|---:|---:|---:|---:|---:|
| 1.5 | 0.17 | 0.0 | 0.0 | 118.5 |
| 2.5 | 0.10 | 0.0 | 7.4 | 118.8 |
| 4.0 | 0.25 | 0.0 | 107.5 | 118.1 |
| 6.0 | 0.47 | 5.9 | 454.1 | 115.7 |
| **9.0** | **0.66** | **658.2** | **1132.1** | **110.8** |

**9.0 was taken**: about two thirds of a naive population is viable, and the badly-allocated third
is exactly what #21 would remove.

### Read the run long enough

A first pass at 800 ticks and one seed reported 0.51 / 0.56 / 0.71 fed at 2.5 / 4.0 / 6.0 —
substantially rosier than the table above, and it led to 6.0 being chosen before the longer run
corrected it. **Populations were still declining at 800 ticks.** An animal starts with 180 energy
units and burns roughly 0.2 per tick of upkeep plus locomotion, so a marginal one takes well over a
thousand ticks to actually run out; anything shorter measures the founding endowment rather than the
equilibrium.

Recorded because it is the mistake this file exists to stop the next person repeating, and because
§2.2 is explicit that a single run of a non-deterministic simulation is not evidence.

## What the field says about the binding constraint

Standing crop moves little across the sweep — roughly 130 ungrazed against 110.8 at the chosen
intake, a 15% depletion. **Food is not what limits this population**: 200 animals on 65,536 cells
cannot outrun a field whose regrowth at equilibrium is roughly 2.6 energy units per cell per tick.
The binding constraint is the animal's own energy balance.

That is worth stating because it means this table measures *metabolic viability*, not carrying
capacity. Carrying capacity is not observable until #20 grows the population into the field's
supply, and the intake rate should be revisited then — it is likely to look generous once density
matters.

## The gradient is real, and it is not the only thing acting

The allocation at which founders starve moves right as intake rises, which is the expected
direction:

| `intake_rate` | richest surviving allocation | poorest starved allocation |
|---:|---:|---:|
| 2.5 | 0.628 | 0.292 |
| 4.0 | 0.667 | 0.352 |
| 6.0 | 0.661 | 0.449 |

The two columns **overlap** at every rate: some animals with a worse allocation survive while
better-allocated ones starve. That is locomotion — animals pay `size × (transport × distance × …)`
every tick they move (#25), and where an animal wandered matters as much as what it can digest over
this timescale. The diet gradient is a tendency rather than a sorting rule, which is what it should
be.

## What is locked in

`tests/core/ecology/test_grazing_equilibrium.py`, asserting directions and distributions rather than
values (§2.2 rules out golden outputs):

- a herbivore ends richer than a carnivore in a world of grass, and energy falls monotonically as
  the allocation leaves plants (rank correlation > 0.95 over 16 allocations)
- most of a naive founder population holds its energy, over five seeds
- an animal holding less than one tick's upkeep, standing on food, survives the tick — §2.1's
  ordering, tested on the single tick where it is decidable
- standing crop settles rather than being stripped to zero or running away

Those run against a stationary fixture with locomotion excluded, deliberately: this is about the
balance between what a gut brings in and what a body costs to run, and mixing in a third term that
`core.behaviour.movement` already owns would make a regression here unattributable.

## Not measured, and why

§2.1's **~10² feeding events per lifetime** could not be checked. There are no lifetimes until #21,
and for a grazer on a plant *field* there are no discrete feeding events either — an animal takes a
mouthful every tick it stands on anything. Filed as #183 with a proposed replacement metric
(metabolic turnover: 0.205 energy units of upkeep per tick against a 180-unit pool is about 880
ticks unfed, roughly 1/600 of §2.1's one-sim-year herbivore lifespan).
