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

## Correction: the measurement above came from a miscalibrated world

**The numbers in the table are not reproducible, and the fault was in the same change that produced
them.** The demo world deliberately damped thirst to a weight of `0.2` — its config comment says "at
equal weights thirst outscores hunger in this climate and nothing in the world ever moves". When the
weights became genes (#23), `thirst_weight` was given the same founding range as every other drive,
`(0.6, 1.4)`, silently discarding that. Thirst then took **100% of a well-fed animal's decision**,
which is how the world behaved when the table above was measured.

Found by reading an inspection panel (#195) — the drive breakdown showed `thirst 100.0%` and
everything else at zero, which no amount of staring at a passing test would have revealed.

With `thirst_weight` founded at `(0.1, 0.3)`, restoring the intent, the effect disappears entirely.
Five seeds, 120 founders, 600 ticks, shift as a fraction of the founding mean:

| gene | median shift |
|---|---:|
| `hunger_weight` | 0.0126 |
| `thirst_weight` | 0.0160 |
| `fear_weight` | 0.0155 |
| `lust_weight` | 0.0262 |
| `fatigue_weight` | 0.0161 |

No separation, and hunger is the *smallest*. **There is no measurable steering-versus-flat effect at
this scale and duration**, and the statistical test that claimed one has been removed.

The lesson is narrow and worth keeping: a statistical test written against a world with a bug in it
locks the bug in. It passed, it was reproducible across seeds, and it was measuring the wrong world.

## Why a flat drive's weight is a neutral gene

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

`tests/core/ecology/test_temperament_evolves.py`:

- weights vary at founding and still vary after 600 ticks — without which everything else is vacuous
- every expressed weight stays non-negative, because it is read as a magnitude (§8.7)
- **scaling a flat drive's weight tenfold leaves every chosen heading bit-identical**, and scaling
  hunger's does not
- the population survives, and is still moving at the end — #23's named degenerate attractor
  ("never eating, never fleeing") is not reached

The steering claim is asserted **mechanically rather than statistically**, and that is the whole
correction. The softmax over option utilities is invariant to a shift shared by all options, so a
flat drive's weight *provably* cannot change the choice at any magnitude. Proving it is both cheaper
and stronger than measuring a population-level shadow of it — especially since the shadow turned out
to be an artefact.

The comparison also has to run against a world that has been ticked. A brand-new one has no plants —
the field starts empty and grows — so hunger's appeal is all zeros and *every* drive is flat at tick
zero, which makes the control pass for the wrong reason.

Caching one world per seed took the module from eight minutes to about 80 seconds; the runs are the
entire cost.
