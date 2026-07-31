# Death, and what actually closes the nutrient loop

Measured while implementing #21. The headline result is that **decomposition is not what closes
§2.5's loop** — excretion is — and that finding changed what #21 shipped.

## The leak decomposition cannot fix

Follow one animal through the nutrient ledger. It eats `H`, assimilates a fraction `c`, and burns
`S` on upkeep and locomotion over its life:

| returned to the field | amount |
|---|---|
| faeces, during life | `H(1 − c)` |
| carcass, at death | `H·c − S` at best |
| **never returned** | **`S`** |

`S` is everything it ever metabolised. Over a long life that is nearly all of it, so a world with
decomposition and nothing else leaks steadily: `exported_nutrients` climbs, soil falls, and the
field starves while the ledger says the nutrients still exist.

So `Ecology.spend` returns the nutrients corresponding to energy **actually burned** — not the bill,
since the pool floors at zero — into the cell the animal is standing in. Metabolism and locomotion
are respiration: the energy is gone, the nutrients it was carried in are not. One rule, in the one
place every draw on the pool passes through, so #25's locomotion and #20's gestation are covered
without either being told about it.

## A starved animal leaves nothing, and that is arithmetic

The same ledger gives an identity worth stating on its own:

```
ledger contribution = E₀ + Σ(H·c) − Σ S  =  current energy
```

Founding records `E₀` (`Plants.record_founding_stock`), feeding adds what was assimilated, every
spend removes what was burned. **An animal's nutrient debt is exactly its energy.**

`Ecology.starving` is `energy <= 0`. So a starved animal owes nothing and can leave nothing — it has
metabolised its own body, which is what starving to death is. A carrion field was built, tested and
then removed, because in a world where starvation is the only death there is no mass to put in it
(#185). Carrion needs a body distinct from its fuel, which is #20's gestation.

## Measurement

`demo_world_config(200 founders, seed 0)`, 2,500 ticks, `debug_checks=True` so the invariant harness
runs after every tick. Windows 11 / Python 3.12.10 / NumPy 2.5.1.

| tick | alive | mean energy | field biomass | `exported_nutrients` |
|---:|---:|---:|---:|---:|
| 0 | 200 | 180.0 | 0.00 | 36,000 |
| 500 | 181 | 420.3 | 119.15 | 76,073 |
| 1000 | 158 | 794.2 | 120.53 | 125,481 |
| 1500 | 154 | 1136.9 | 120.93 | 175,089 |
| 2000 | 152 | 1480.9 | 121.07 | 225,102 |
| 2500 | 142 | 1941.0 | 121.69 | 275,618 |

**Nutrients: opening 2,596,000, closing 2,596,000, relative drift 1.08e-15.** Exactly conserved
across 2,500 ticks of growth, senescence, grazing, faeces, excretion and death.

The export ledger is a second, independent check on the identity above: at tick 2500, 142 survivors
holding a mean of 1941.0 energy units account for 275,622 against a reported 275,618 — the ledger
*is* the living population's energy, to four significant figures, because nothing else is outstanding.

## Selection is now visible

Founders draw `diet_animal_derived` from `(-1, 1)`, which the logistic squash reads as allocations
spanning 0.28–0.73 around a mean of **0.48**. After 2,500 ticks the survivors span 0.28–0.63 around
a mean of **0.415**.

The badly-allocated tail is gone, and the ceiling has come down from 0.73 to 0.63. That is the first
time in this repository that a gene has had a demonstrated fitness consequence in an assembled world
rather than in a fixture — and it is what #19's tuning note anticipated when it said the intake rate
had to carry founders that selection could not yet remove.

Nothing is inherited yet (#20), so this is differential survival rather than evolution: the
distribution is being *filtered*, not *moved*. The distinction matters for reading the number — a
population that can only lose its worst members will converge and then stop, which is exactly what
the flattening between ticks 1500 and 2000 shows.

## The population can only fall

152 alive at tick 2000 against 200 founders, with 58 rows free. Nothing is born, so this is the
honest shape of a half-closed loop and not a tuning failure. The equilibrium visible here is not a
carrying capacity — it is the size of the subset of a fixed founder draw that happens to be viable.
Carrying capacity is not observable until #20 lets the survivors breed into the field's supply.

## What is locked in

- `tests/core/ecology/test_death.py` — rows freed, freed rows reusable within a tick (§2.1's
  death-before-reproduction ordering), ids not reclaimed (#119), and that dying moves no nutrients.
- `tests/core/ecology/test_service.py::TestSpendingExcretes` — burning returns nutrients to the
  cell, only what was actually burned, and the world total is unmoved.
- `tests/core/world/test_assembly.py` — a population falls under the real loop, and a row freed by
  death is handed out again.

The conservation result above is asserted continuously rather than once: `nutrients_are_conserved`
is registered by `default_registry` and evaluated after every tick whenever `debug_checks` is on.
