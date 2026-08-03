"""Conception: where genetics finally enters the population (CLAUDE.md §2.5, issue #20).

`core.genetics.inheritance` has been built and tested since #14 and **had never been called by a
running world.** Every gene in every world so far was a founder's draw, unchanged — filtered by
death (#21) but never passed on. This is what turns filtering into evolution: a distribution that
could only lose its worst members can now move.

**Mating is a process, not a matching.** Sensing a mate, wanting one enough to overcome the bearing
you were already holding, and walking there are the first three stages, and all three already
existed — the cue field read with the searcher's own signature (#188), the drive contest, `#100`'s
commitment, and `Movement`. This module is only the fourth stage: what happens when two willing
animals are in the same place. Nothing here searches for a partner or ranks one, because by the time
an animal arrives the searching is what got it there.

**A gestating offspring is a row with a negative age.** It is allocated at conception, holding its
inherited genes and its endowment, and `Aging` — which already runs every tick and does not care
about sign — counts it up to zero. Birth is not an event: it is `age >= 0` becoming true.

That does a surprising amount of work for one integer:

- **No pregnancy state anywhere.** No `gestation_remaining` column, and no second gene matrix to
  hold an unborn genome — the genome lives in the genome matrix, which is where genomes live. The
  alternative cost 24 MB at 100k entities to store what the store already stores.
- **The countdown is a system that exists.** `Aging.advance` is the gestation clock, unmodified.
- **`Lust` needs no change**: its maturity gate is `age >= maturity_age`, and a negative age is
  below every maturity, so an unborn animal is correctly not looking for a mate.
- **A pregnancy survives its parents.** The offspring is not attached to anybody, so both parents
  can die before its term is up and it is still born — which is right, and would have needed
  deliberate work under any design that hung the pregnancy off a parent.

The one real consequence: **an animal is born where it was conceived.** A gestating row is excluded
from movement along with everything else, so it stays put while its parents walk on. Carrying it
would require a link back to a parent — an id resolved through a dict, which is Python in the tick
loop, or a row index the free list can invalidate under it. Born-where-conceived is coherent on its
own terms (a nest site, a spawning ground) and it makes *where* an animal conceives matter, which is
pressure the world applies rather than a rule anybody wrote.

**Gestation costs energy that is moved, not burned.** The endowment comes out of both parents in
equal halves through `Ecology.transfer`, which is why that method exists: `spend` excretes the
nutrients it debits (#21), so routing gestation through it would return them to the soil *and* hand
the energy to the offspring, inventing it. Equal halves because there is no mother — sex is a
continuous allocation nobody has built yet (#99), and until it exists the asymmetry has nowhere to
come from.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.ecology.contact import pair_by_contact
from core.ecology.service import Ecology
from core.entities.store import EntityStore
from core.genetics.registry import GeneRegistry, Unit
from core.genetics.service import Genetics
from core.genetics.speciation import interbreeding_probability
from core.selection import Selection


@dataclass(frozen=True)
class ConceptionConfig:
    """Per-world reproduction rules — never constants in `core/` (§2.1).

    contact_range: world units. How close two willing animals must be to conceive. This is a
        *contact* distance and not a search radius: finding each other is the drive's business
        (#188), and by the time an animal is here it has already walked.
    offspring_energy: energy units a newborn holds, taken in equal halves from its parents. It is
        moved rather than charged, so this is the whole cost of breeding — and therefore also the
        energy an animal must hold before it can breed at all, which is why readiness needs no
        separate threshold.
    maturity_gene: ticks an animal must live before it can breed. The same gene `LustConfig` reads,
        named once per world so that wanting to breed and being able to cannot disagree.
    gestation_gene: ticks between conception and birth, as a **negative starting age**. A gene
        because life-history theory puts it under exactly the selection this world applies: a short
        gestation returns a parent to breeding sooner, a long one is a bet nothing here rewards yet
        and #185's carcasses will.
    speciation_threshold: genetic distance at which interbreeding probability reaches zero. #16 owns
        what happens *at* that distance — recording a split — and reads the same number.
    """

    contact_range: float
    offspring_energy: float
    maturity_gene: str
    gestation_gene: str
    speciation_threshold: float

    def __post_init__(self) -> None:
        if self.contact_range <= 0.0:
            raise ValueError(f"contact_range must be positive, got {self.contact_range}")
        if self.offspring_energy <= 0.0:
            raise ValueError(
                f"offspring_energy must be positive, got {self.offspring_energy}; a free offspring "
                "makes breeding costless and every lineage breeds without limit"
            )
        if self.speciation_threshold <= 0.0:
            raise ValueError(
                f"speciation_threshold must be positive, got {self.speciation_threshold}; at zero "
                "no two animals are ever compatible and nothing can ever breed"
            )


class Conception:
    """Turns co-located willing animals into gestating rows, once per tick.

    Owns no store column. Genes are `Genetics`', energy is `Ecology`'s, rows are the store's — this
    service decides only *who breeds with whom*, which is the one judgement none of them should be
    making.
    """

    def __init__(
        self,
        store: EntityStore,
        ecology: Ecology,
        genetics: Genetics,
        genes: GeneRegistry,
        config: ConceptionConfig,
    ) -> None:
        self.store = store
        self.ecology = ecology
        self.genetics = genetics
        self.config = config
        # Both are counts of ticks, so dimensionless: §2.1 makes the tick the only clock, and
        # neither is a length or an energy.
        self._maturity_index = genes.index_of(config.maturity_gene, unit=Unit.DIMENSIONLESS)
        self._gestation_index = genes.index_of(config.gestation_gene, unit=Unit.DIMENSIONLESS)

    def willing(self, selection: Selection) -> Selection:
        """The entities in `selection` old enough and rich enough to breed.

        Maturity is per-entity because it is a gene, so two animals of the same age can differ in
        whether they are ready — which is what lets selection tune age at first reproduction rather
        than a designer choosing it. The energy half is not a tuned threshold at all: it is the
        arithmetic statement that you cannot give away what you do not have.
        """
        mask = selection.to_mask()
        maturity = self.genetics.expressed(selection)[:, self._maturity_index]
        ready = np.zeros_like(mask)
        ready[mask] = (self.store.age[mask] >= maturity) & (
            self.ecology.energy(selection) > self.config.offspring_energy
        )
        return Selection.from_mask(ready)

    def conceive(self, selection: Selection, rng: np.random.Generator) -> None:
        """Pair the willing animals in `selection` that are touching, and gestate one young each.

        `selection` is the caller's choice of who may breed; pass the living. Nothing here filters
        beyond willingness, for the same reason `Ecology.drain` does not (§8.7).
        """
        rows = self.willing(selection).to_indices()
        if rows.shape[0] < 2:
            return

        first, second = self._couples(rows, rng)
        if not first.shape[0]:
            return

        # Compatibility is a probability rather than a gate, so a pair drifting toward the split
        # threshold breeds ever more rarely before it stops breeding at all — #16's "no cliff edge".
        chance = interbreeding_probability(
            self.genetics, first, second, self.config.speciation_threshold
        )
        accepted = rng.random(chance.shape[0]) < chance
        first, second = first[accepted], second[accepted]

        # Capacity is not grown here: `EntityStore.grow` must run at a tick boundary and this is
        # mid-tick (§2.3), so a world short of rows conceives fewer young rather than raising. How
        # many rows to keep spare is #127's, with the rate this produces in hand.
        room = self.store.available
        if room < first.shape[0]:
            first, second = first[:room], second[:room]
        if not first.shape[0]:
            return

        self._gestate(first, second, rng)

    def _gestate(
        self, first: np.ndarray, second: np.ndarray, rng: np.random.Generator
    ) -> None:
        """Allocate one gestating row per couple and move its endowment out of its parents."""
        n = first.shape[0]
        offspring_genes = self.genetics.inherit(first, second, rng)

        # The offspring's own gestation gene decides its term, read from the genes it just
        # inherited rather than from either parent: how long a young takes is the young's trait,
        # which is what puts it under selection at all.
        expressed = self.genetics.expression.phenotype(offspring_genes)
        term = np.rint(expressed[:, self._gestation_index]).astype(np.int64)

        ids = self.store.allocate(
            n,
            # Born where conceived: a gestating row is excluded from movement, so it stays here
            # while its parents walk on. See the module docstring for why carrying it is worse.
            x=self.store.x[first],
            y=self.store.y[first],
            z=self.store.z[first],
            # Negative, and counted up by `Aging` every tick. Birth is this reaching zero.
            age=-term,
            energy=np.zeros(n, dtype=np.float32),
            species_id=self.store.species_id[first],
            genes=offspring_genes,
        )
        rows = np.array([self.store._id_to_row[i] for i in ids.tolist()], dtype=np.int64)
        young = Selection.from_indices(rows, capacity=self.store.capacity)

        # Half each, so neither parent can be emptied by a birth the other could afford, and both
        # are poorer by exactly what the young holds — which is what makes the cost real without an
        # exchange rate being chosen.
        half = np.full(n, self.config.offspring_energy / 2.0, dtype=np.float32)
        self.ecology.transfer(Selection.from_indices(first, self.store.capacity), young, half)
        self.ecology.transfer(Selection.from_indices(second, self.store.capacity), young, half)

    def _couples(
        self, rows: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Row-index arrays of the willing animals that are touching, paired elementwise.

        The spatial half is `core.ecology.contact.pair_by_contact`, shared with predation (#179)
        because "who is close enough to interact" is one question with two consequences. What
        brought these two together is the lust drive walking them here (#188), not anything here.
        """
        return pair_by_contact(
            self.store.x, self.store.y, rows, self.config.contact_range, rng
        )
