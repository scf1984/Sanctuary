# Temperament under selection, and the thing that decides whether a weight is selected at all

Measured while implementing #23, which makes each drive's weight a gene so that "boldness,
sociality and parental investment arise from selection" (§2.5).

The result is not the one the issue predicts, and the reason is structural rather than ecological.

## What was expected

#23's done-when: *"a statistical test over many seeds proves a predated population evolves higher
fear weighting than an unpredated control"*. There are no predators (#179), so the first attempt
ran the reachable half — with nothing worth fearing, a high fear weight should be pure cost, since
a frightened animal forgoes food for nothing.

**It failed, and in the wrong direction.** Demo world, 200 founders, 900 ticks, three seeds:

| weight | mean at the end, as a fraction of the founding mean |
|---|---:|
| `hunger_weight` | **0.893** |
| `fear_weight` | 0.949 |

Hunger fell nearly twice as far as fear, consistently.

## Why: a flat drive's weight is a neutral gene

`Behaviour.choose` scores `utility(option) = Σ over drives of urgency × appeal(option)`.

A drive whose `appeal` is **flat** adds the same number to every option. It cannot change which
option wins — only the total, which the Boltzmann sampling then normalises away. So its weight has
almost no consequence, and a gene with no consequence is a random walk (§2.5's own rule: every gene
needs an energy cost or a selective consequence).

Which drives currently steer:

| drive | appeal | weight under selection? |
|---|---|---|
| hunger | the diffused forage field (#93) | **yes** |
| lust | the cue field, own signature as vector (#188) | **yes** |
| fatigue | all of it on the null option (#107) | **yes** |
| thirst | flat — nothing drinks (#156) | no |
| fear | flat — flight has never existed | no |

So `fear_weight` and `thirst_weight` are, today, free-drifting genes. That is a fact about what is
built rather than about temperament, and it will change the moment fear can steer.

**This is also why #23's done-when needs predation and not merely a predator.** Even with something
hunting, fear's weight cannot be selected until `Fear.appeal` returns a direction — the animal has
to be able to *act* on being afraid. #188 supplied the cue-field machinery for that and left fear's
consumer to #24.

## The second surprise: hunger is selected *down*

Falling to 0.89 of the founding mean, not rising. Consistent across seeds.

The world is food-saturated: `docs/spikes/grazing-equilibrium.md` measured that 200 animals barely
dent a field whose regrowth is ~2.6 energy units per cell per tick, and
`docs/spikes/capacity-growth.md` shows a population overshooting to 5,212 before food bites at all.
When food is underfoot everywhere, *walking toward more of it* is a cost with no return — the
locomotion bill (#25) is real and the extra grazing is not. So a lower hunger weight wins.

That is a legible ecological result rather than a bug: in a glut, foraging effort is wasted effort.
It also predicts its own reversal, which is worth checking once something limits food — under
scarcity the same gene should climb.

## What is locked in

`tests/core/ecology/test_temperament_evolves.py`, asserting directions only (§2.2):

- weights vary at founding and still vary after 600 ticks — without which everything else is vacuous
- every expressed weight stays non-negative, because it is read as a magnitude (§8.7)
- **a steering drive's weight moves further than a flat drive's**, compared across five replicates
- the population survives, and is still moving at the end — #23's named degenerate attractor
  ("never eating, never fleeing") is not reached

### On replicates

The steering-vs-flat comparison is asserted on the **median across five seeds**, not per seed. Per
seed it failed on one of three while holding comfortably in aggregate, which is exactly what §2.2
warns about: "variance between two runs of the same state can exceed the difference between A and
B". A per-seed assertion there is a coin flip dressed as a result.

Caching the runs took the module from eight minutes to 72 seconds — the runs are the cost, and
every assertion can share one world per seed.
