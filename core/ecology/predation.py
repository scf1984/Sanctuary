"""Predation: the second energy transfer between trophic levels (CLAUDE.md §2.5, issue #179).

Before this, every animal in every world was a grazer. `diet_animal_derived` existed, was inherited,
was expressed — and its animal half bought **nothing**, so an allocation away from plants was pure
loss and selection could only ever drive it toward zero. Half of #102's encoding was unreachable,
and with it the whole reason §2.5's cue space exists: a co-evolutionary arms race needs something to
run from.

**There is no predator and no prey.** Nothing here reads a species, a category or a table. Every
animal is a potential eater of the animal beside it and a potential meal for it, and which it turns
out to be is decided entirely by two continuous genes — the diet allocation and body size. A lineage
becomes carnivorous by drifting, and stops being carnivorous the same way. That is §2.5's rule that
"nothing anywhere lists which species interacts with which", applied to the one interaction that
most invites a table.

## Killing and eating are two acts, and conflating them is what makes a nibble

The first build of this module had a strike *be* a mouthful: an attacker took `intake_rate × size`
out of its prey and swallowed it. That is wrong, and the measurement said so before the reasoning
did — over 2,000 ticks the flesh allocation fell from 0.50 to **0.04** and carnivory was selected
out of the world entirely (`docs/spikes/predation-viability.md`).

The cause is that a body is an order of magnitude bigger than a mouthful. A grazer holds ~50 energy
units and a gut processes ~4 per tick, so killing took a dozen consecutive ticks of contact that two
moving animals never have — while the allocation charged its full opportunity cost in grass from the
first tick. The trade could never pay, in any world, at any sun.

**These are two different physical limits.** Killing is bounded by force; eating is bounded by a
gut. Nothing in nature makes them the same number, and the split is what gives carrion its mass
source (#185) — the difference between them *is* the body left lying on the ground:

```
damage  = strike_power × (size / prey_size) × animal_share ** p   what the prey loses
carrion = damage                                                all of it, onto the ground
```

**A strike feeds the attacker nothing directly**, and that is the sharpest form of the same point. A
predator eats its kill by *standing on it* and grazing the carrion field next tick, exactly as a
herbivore grazes grass. Three things then exist that nobody implemented: scavenging, a reason to
defend a kill (`Carrion.graze` contends per cell), and a real payoff for #100's `commitment` gene —
a lineage too flighty to hold a bearing kills and walks away from the meal.

**Size divides, so a big animal is hard to kill.** The same jaws do less to a bigger body, which is
free physics rather than an authored defence, and it is what finally gives `size` a benefit beyond a
larger mouthful: until now it only ever charged upkeep (#17), locomotion (#25) and turning inertia
(#204). A predator/prey axis needs both directions to be worth having, and this is the second.

**The allocation gates the attempt.** `Feeding` deliberately does *not* scale a grazing mouthful by
diet — "a gut processes what a gut of that size can process, and what it *gets* is what it can
digest" — and that rule is safe there because the only victim of a wasted mouthful is the eater's own
tick. It is not safe here: a bite taken from another animal costs a *third party* its life, so an
ungated strike is not self-punishing but other-punishing, and selection cannot correct a cost that
falls on someone else. Ungated, every animal in a herd would maul whichever neighbour it stood next
to and the population would annihilate itself while every individual gene did nothing wrong.

## Nothing here decides a kill, and nothing here should

§2.5 settled this when momentum shipped (#204): *"#179 therefore needs no kill rule: contact
decides, exactly as contact already decides mating. If that issue ships a resolution formula,
momentum was missing and the formula is standing in for it."* There is no chase resolution below, no
success probability, no trait ratio. Two animals are either touching or they are not, and getting
there is a pursuit fought out in `core.behaviour.movement` where velocity is state and a heavy fast
body cannot turn as sharply as a light one.

**Death is not here either.** A strike empties a pool; `Ecology.starving` reads an empty pool and
`Death` frees the row, both unchanged and both already in the tick. So whether a strike kills is a
question about relative size that nobody had to answer: a large attacker empties a small victim
outright, and a small one wounds a large one and has to come back. **That is the multi-tick kill,
for free**, and it is the same shape #19 got from the plant field without a column either.

## What this makes possible that nothing implements

- **Cannibalism**, immediately and without a mechanic: you are an animal, so an animal-allocated
  lineage eats its own kind unless something stops it. §2.5 explicitly wants that available.
- **Fear with teeth.** Aversion, cue signature and camouflage have been inherited and expressed
  since #22 with nothing selecting on any of them, because nothing in the world was dangerous.
- **A reason for `size` to rise.** Until now size only ever charged upkeep, locomotion and turning
  inertia against a mouthful that scaled with it; being big is now also being hard to eat.

## What is deliberately absent: an appetite direction

A predator here is drawn toward *animal smell in general*, because `Hunger` blends the forage field
with the cue field summed over its channels (#179's Decision 1, answered with the option needing no
new gene). It therefore cannot specialise on one prey's signature — smell is blunt here exactly as
§2.5 requires it to be for fear, and for the same reason: the air holds a blend, and only a linear
readout composes correctly on a blend.

A per-lineage *appetite direction* in cue space — the mirror of `aversion`, pointing at what you
want to eat rather than what frightens you — is the richer answer, and it is additive to the gene
vocabulary rather than a replacement, so it stays open on #179 rather than being decided in a diff.
That is also what would let a predator's signature and its prey's aversion enter a genuine arms
race in both directions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.ecology.carrion import Carrion
from core.ecology.contact import pair_by_contact
from core.ecology.diet import Diet
from core.ecology.feeding import FeedingConfig
from core.ecology.service import Ecology
from core.entities.store import EntityStore
from core.genetics.registry import GeneRegistry, Unit
from core.genetics.service import Genetics
from core.selection import Selection


@dataclass(frozen=True)
class PredationConfig:
    """Per-world predation rules — never constants in `core/` (§2.1).

    strike_range: world units. How close two animals must be for one to strike the other. A
        *contact* distance and not a search radius, exactly as `ConceptionConfig.contact_range` is:
        what brought two animals together is movement, and by the time they are here the pursuit is
        over.

        Declared separately from the mating range rather than shared, because the two are different
        physical facts — reaching a mate and reaching prey that is trying to leave — and §2.1's
        warning is about constants that must move *together*, which these need not.
    strike_power: energy units a strike takes out of an equally-sized victim at full allocation.
        The **force** limit, and it is deliberately not the gut limit: a body is an order of
        magnitude larger than a mouthful, and setting this to `intake_rate` is precisely the defect
        the module docstring records — carnivory that cannot pay in any world.

        It must be tuned against `CarrionConfig.decay_rate` and `FeedingConfig.intake_rate` as one
        table (§2.1): how much a kill yields, how fast it rots, and how fast a predator can eat it
        are three halves of one question, and any of them alone says nothing.

    The mouthful is not declared here at all. `FeedingConfig` is what a gut can process per tick and
    how well, which is one fact about an animal whatever it is eating; a second intake rate for
    flesh would be the pair of coefficients describing one preference that §2.1 warns will drift.
    """

    strike_range: float
    strike_power: float

    def __post_init__(self) -> None:
        if self.strike_range <= 0.0:
            raise ValueError(
                f"strike_range must be positive, got {self.strike_range}; at or below zero no two "
                "animals are ever in contact and nothing can ever be eaten"
            )
        if self.strike_power <= 0.0:
            raise ValueError(
                f"strike_power must be positive, got {self.strike_power}; at or below zero no "
                "strike ever harms anything and the animal half of every diet buys nothing"
            )


class Predation:
    """Turns touching animals into wounds and bodies on the ground, once per tick.

    Owns no store column. Energy is `Ecology`'s, genes are `Genetics`', the carrion field is
    `Carrion`'s — this service decides only *who strikes whom*, which is the one judgement none of
    them should be making.
    """

    def __init__(
        self,
        store: EntityStore,
        ecology: Ecology,
        genetics: Genetics,
        carrion: Carrion,
        diet: Diet,
        genes: GeneRegistry,
        feeding: FeedingConfig,
        config: PredationConfig,
    ) -> None:
        self.store = store
        self.ecology = ecology
        self.genetics = genetics
        self.carrion = carrion
        self.diet = diet
        self.config = config
        # A body size is a bare ratio here — the attacker's against the victim's — hence
        # dimensionless. Resolved through the registry so a world declaring `size` as a length is
        # rejected at construction rather than sizing a wound in a denomination nothing could
        # notice (#112).
        self._size_index = genes.index_of(feeding.size_gene, unit=Unit.DIMENSIONLESS)

    def strike(self, selection: Selection, rng: np.random.Generator) -> None:
        """Let every touching pair in `selection` strike each other once.

        `selection` is the caller's choice of who may kill and be killed; pass the living. Nothing
        here filters, for the same reason `Ecology.drain` does not: a tick loop striking anything
        other than the living is a bug in the loop rather than a condition to absorb quietly (§8.7).

        **Both directions resolve, and simultaneously.** Pairing is symmetric — there is no attacker
        role to assign, because what makes an animal an attacker is its own allocation and nothing
        else — so each strikes the other, and each wound is measured against the pool its victim
        held *before* either landed. That is exact rather than approximate: a row appears in at most
        one pair, so each pool has exactly one claim against it. Resolving in sequence instead would
        make the outcome depend on which of two identical animals happened to sort first, which is
        the grid leaking into the ecology.
        """
        rows = selection.to_indices()
        if rows.shape[0] < 2:
            return

        first, second = pair_by_contact(
            self.store.x, self.store.y, rows, self.config.strike_range, rng
        )
        if not first.shape[0]:
            return

        held_first = self.store.energy[first].copy()
        held_second = self.store.energy[second].copy()
        self._wound(first, second, held_second)
        self._wound(second, first, held_first)

    def _wound(self, attacker: np.ndarray, victim: np.ndarray, held: np.ndarray) -> None:
        """One direction: `attacker[i]` takes `victim[i]` apart, onto the ground it is standing on.

        `held` is what the victim's pool held before this tick's strikes, which is what bounds the
        wound — a body cannot yield more than it is.
        """
        attacking = self.genetics.expressed_at(attacker)
        # Size ratio: the same jaws do less to a bigger body, so being large is a defence and not
        # only an upkeep bill. `size` is a magnitude and cannot express zero-or-negative mass
        # (§2.5), so the division needs no guard.
        outmatched = attacking[:, self._size_index] / self.genetics.expressed_at(victim)[
            :, self._size_index
        ]
        # Scaled by the *frontier*, not the raw allocation: killing is a whole-body specialisation
        # exactly as digesting flesh is, so `share ** p` prices the weapon on the same convex curve
        # #102 already prices the gut on. A linear reading was measured and is pathological — every
        # grazer 7% allocated toward meat still wounded whichever neighbour it stood beside, and the
        # population bled into the carrion field faster than anything could eat it (35,000 energy
        # units standing at 2,000 ticks). That cost falls on a third party, so selection cannot
        # correct it; the convex curve is what makes a half-hearted predator harmless.
        damage = np.minimum(
            self.config.strike_power * outmatched * self.diet.animal_efficiency(attacking),
            held,
        ).astype(np.float32)

        # The pool is emptied and the body lands where it fell. Nothing reaches the attacker here:
        # it eats by standing on the carcass next tick, which is what makes a kill worth holding a
        # bearing for (#100) and what makes scavenging exist without a mechanic.
        self.ecology.kill(victim, damage)
        self.carrion.deposit(self.store.x[victim], self.store.y[victim], damage)
