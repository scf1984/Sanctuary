"""Emergent speciation at a genetic distance threshold (CLAUDE.md §2.5, issue #16).

Isolation is the payoff of the whole design: two sub-populations that stop exchanging genes drift
apart under `core.genetics.inheritance`, their expressed-phenotype centroids separate (measured by
`core.genetics.distance`), and past a threshold they are no longer one species. This module is the
mechanic and nothing more -- it decides *that* a split happened and records it. Whether a player
names the daughter species, and how the lineage is surfaced, is unresolved (CLAUDE.md §5) and is
deliberately left to the caller: `split()` returns an opaque id and stores no name.

What a split costs: one appended row on the species registry's `(n_species, n_genes)` mask table,
and one vectorized write into the `species_id` column. Nothing entities-shaped is reallocated,
copied, or restructured -- no gene row moves, no entity row moves, capacity does not change
(CLAUDE.md §2.3). That is the whole reason species are an id plus a mask instead of a storage
layout, and it is asserted directly in the tests by array identity, not inferred.

Reproductive isolation degrades, it does not switch (issue #16's "no cliff-edge"):
`interbreeding_probability` falls linearly from 1 at zero distance to 0 at the same threshold
`split()` fires on. So by the time two sub-populations are far enough apart to be split, pairs
drawn across them were already interbreeding at a rate near zero -- the species-id gate that takes
over afterwards removes an already-vanishing probability rather than a live one. Hybridisation
therefore thins out over generations of drift instead of ending on the tick of the split.
"""

from __future__ import annotations

import numpy as np

from core.genetics.distance import between, centroid_between
from core.genetics.service import Genetics
from core.selection import Selection


class MixedSpeciesError(Exception):
    """A selection that had to belong to a single species contained more than one."""


class Lineage:
    """Which species each species split off from.

    Roots -- species registered directly rather than produced by `split()` -- have no parent and
    are not recorded here at all; `parent_of` returns None for them, which is what terminates an
    `ancestry` walk. Only the parent link is stored: it is what ancestry display needs, and this
    module has no clock of its own to timestamp anything with (CLAUDE.md §2.1 -- the tick counter
    lives in the world loop, and inventing a parameter no caller passes today would violate §8.2).
    """

    def __init__(self) -> None:
        self._parent_by_species: dict[int, int] = {}

    def record_split(self, species_id: int, parent_species_id: int) -> None:
        """Record that `species_id` split off from `parent_species_id`.

        Raises ValueError on re-recording a species: a species id is created by exactly one
        split and its parent never changes, so a second record is a caller bug, not an update
        (CLAUDE.md §8.7).
        """
        if species_id in self._parent_by_species:
            raise ValueError(
                f"species {species_id} already descends from "
                f"{self._parent_by_species[species_id]}"
            )
        self._parent_by_species[species_id] = parent_species_id

    def parent_of(self, species_id: int) -> int | None:
        """The species `species_id` split off from, or None if it is a root."""
        return self._parent_by_species.get(species_id)

    def ancestry(self, species_id: int) -> tuple[int, ...]:
        """The chain from the root ancestor down to `species_id`, inclusive.

        A root's ancestry is just itself. Because every recorded parent is a species that already
        existed when the split happened, and `record_split` refuses to re-parent, the parent
        links form a forest -- so this walk always terminates at a root.
        """
        chain = [species_id]
        parent = self.parent_of(species_id)
        while parent is not None:
            chain.append(parent)
            parent = self.parent_of(parent)
        return tuple(reversed(chain))


def has_diverged(genetics: Genetics, a: Selection, b: Selection, threshold: float) -> bool:
    """Whether sub-populations `a` and `b` have drifted far enough apart to be separate species.

    Compares *centroids*, not individuals: speciation is a property of two populations having
    moved apart, and a single outlier pair in either one says nothing about whether the gene pools
    have separated. `a` and `b` may differ in size; both must be non-empty (`centroid_between`
    raises otherwise, CLAUDE.md §8.7).

    threshold: distance in expressed-phenotype units -- the same units and the same metric
        `interbreeding_probability` reaches zero at, so the two cannot drift apart into
        "incompatible but not yet split" or "split but still fertile".
    """
    return bool(centroid_between(genetics, a, b) >= threshold)


def split(genetics: Genetics, lineage: Lineage, diverged: Selection) -> int:
    """Make `diverged` a new species descended from its current one; returns the new species id.

    The daughter species starts with a copy of its parent's expression mask: a split changes who
    breeds with whom, not which genes anyone expresses. Divergence in expression comes afterwards,
    from the two populations' masks being edited independently -- and every gene stays stored and
    inherited in both, expressed or not, so a trait one branch stops expressing can resurface in a
    descendant (CLAUDE.md §2.3).

    `diverged` must be non-empty and entirely one species: splitting a mixed selection would give
    the daughter a single parent it did not entirely come from, silently corrupting the lineage
    record, so it raises (CLAUDE.md §8.7) rather than picking a parent.
    """
    if len(diverged) == 0:
        raise ValueError("cannot split an empty selection into a new species")

    species_ids = genetics.species_ids(diverged)
    parent_species_id = int(species_ids[0])
    if not (species_ids == parent_species_id).all():
        raise MixedSpeciesError(
            f"a split must come from one species; selection spans "
            f"{sorted(set(species_ids.tolist()))}"
        )

    new_species_id = genetics.species.derive(parent_species_id)
    genetics.speciate(diverged, new_species_id)
    lineage.record_split(new_species_id, parent_species_id)
    return new_species_id


def interbreeding_probability(
    genetics: Genetics, a: np.ndarray, b: np.ndarray, threshold: float
) -> np.ndarray:
    """(len(a),) float32 in [0, 1]: how readily each paired (a[i], b[i]) couple can produce young.

    `a` and `b` are **row index arrays**, paired elementwise, exactly as `distance.between` pairs
    them and for the same reason: a `Selection` is a mask and cannot carry a pairing that crosses
    in row space (#20). Both consulted signals appear here:

    - **species id** -- a pair from two different species scores 0. Once a split has been recorded
      the isolation is a fact of the world, not something re-derived from where two individuals
      happen to sit each tick.
    - **distance** -- within one species, compatibility falls linearly from 1 at identical
      phenotypes to 0 at `threshold`. This is the gradual degradation issue #16 asks for: a
      sub-population approaching the split threshold breeds with the parent stock ever more rarely,
      so the hard species gate that follows the split has almost nothing left to remove.

    Linear rather than some sharper falloff because nothing measured yet justifies a curve
    (CLAUDE.md §8.5); the shape is the tuning knob, the two signals are the mechanic.

    threshold: must be > 0 -- at 0 every non-identical pair would already be incompatible and the
        gradual degradation this function exists for would not exist.
    """
    if threshold <= 0:
        raise ValueError(f"threshold must be > 0, got {threshold}")

    compatibility = 1.0 - between(genetics, a, b) / threshold
    same_species = genetics.species_ids_at(a) == genetics.species_ids_at(b)
    return np.where(same_species, np.clip(compatibility, 0.0, 1.0), 0.0).astype(np.float32)
