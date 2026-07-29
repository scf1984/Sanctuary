"""The cross-species threat matrix: how dangerous each species is to each other (#22).

`W[observer, threat]` is one number — how much a member of `observer` has to fear a member of
`threat` — and it is **shared by every perception channel** (CLAUDE.md §2.5). A wolf is equally
dangerous smelled or seen; what differs between senses is only how much of it you can perceive, and
that lives in the channel. Keeping danger here and access there is what lets #24 add sight without
touching a single authored number.

The diagonal is **cannibalism**, and it is an ordinary entry: a species that eats its own young
has a positive `W[s, s]`, one that does not has zero, and neither is a special case in any code
that reads this.
"""

from __future__ import annotations

import numpy as np

from core.genetics.species import SpeciesRegistry


class Threat:
    """A square weight matrix over species ids, extended by speciation rather than by authoring.

    weights: ``(n_species, n_species)`` float32, unit-free. Row is the observer, column is the
        threat. Not symmetric, and it must not be: prey fear predators far more than predators fear
        prey, which is the whole asymmetry an ecology is built on.

    Sized to the registry at construction and grown one row and column at a time, so a species id
    is always a valid index into it.
    """

    def __init__(self, species: SpeciesRegistry, weights: np.ndarray) -> None:
        weights = np.asarray(weights, dtype=np.float32)
        expected = (species.n_species, species.n_species)
        if weights.shape != expected:
            raise ValueError(
                f"threat weights must be {expected} for {species.n_species} registered species, "
                f"got {weights.shape}"
            )
        if np.any(weights < 0):
            # A negative weight is attraction wearing fear's name. If a species should seek
            # another out, that is a different drive, not fear with the sign flipped.
            raise ValueError("threat weights must be non-negative")

        self.species = species
        self.weights = weights

    def derive(self, parent_species_id: int) -> int:
        """Extend the matrix for a daughter of `parent_species_id`; returns the new species id.

        The daughter takes its parent's **row and column**: at the moment of a split it is
        ecologically identical to its parent, so it fears exactly what its parent feared and is
        feared exactly as its parent was. Drift is what separates them afterwards.

        This mirrors `SpeciesRegistry.derive`, and it is what keeps CLAUDE.md §2.3's claim true —
        that speciation is a species-id write plus a new mask row. Without it, the first split in
        a world would either index past the end of this matrix or stop to ask the player how
        frightening the new species is.

        The registry is the source of ids: this calls `derive` on it, so the two cannot disagree
        about how many species exist or which id the daughter got.
        """
        parent_row = self.weights[parent_species_id].copy()
        parent_column = self.weights[:, parent_species_id].copy()
        child_id = self.species.derive(parent_species_id)

        grown = np.zeros((child_id + 1, child_id + 1), dtype=np.float32)
        grown[:child_id, :child_id] = self.weights
        grown[child_id, :child_id] = parent_row
        grown[:child_id, child_id] = parent_column
        # The daughter's self-weight is its parent's self-weight: two populations that were one
        # species a tick ago regard each other exactly as they regarded themselves.
        grown[child_id, child_id] = self.weights[parent_species_id, parent_species_id]

        self.weights = grown
        return child_id

    def rows_for(self, species_ids: np.ndarray) -> np.ndarray:
        """(len(species_ids), n_species) float32: each entity's row of the matrix.

        A single fancy-index gather, matching `SpeciesRegistry.masks_for`, so weighting a
        mixed-species population is one vectorized pass regardless of how many species are present
        (CLAUDE.md §2.3).
        """
        return self.weights[np.asarray(species_ids, dtype=np.int64)]
