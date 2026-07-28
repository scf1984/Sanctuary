"""Species registry: which genes each species expresses (CLAUDE.md §2.3).

A species is not a storage layout — every creature already has every gene slot (see
`core.genetics.vocabulary`). A species is a row in this registry: an expression mask over the
shared vocabulary. Genes outside a species' mask are not deleted, zeroed, or otherwise touched by
this registry; they simply do not contribute to that species' phenotype wherever the mask is
applied (`core.genetics.service.Genetics.expressed`). A lineage that stops expressing a gene keeps
carrying it and can express it again generations later if a descendant's species comes to express
it — atavism, a deliberate feature (CLAUDE.md §2.3), not a concession.

Speciation is registering a new mask and writing its id into the affected entities' `species_id`
column (`Genetics.speciate`); nothing here reallocates or restructures anything.
"""

from __future__ import annotations

import numpy as np

from core.genetics.vocabulary import GeneVocabulary


class UnknownSpeciesError(Exception):
    """A species id has no registered expression mask."""


class SpeciesRegistry:
    """Maps species id to an expression mask over one `GeneVocabulary`.

    Ids are assigned sequentially by `register()`, starting at 0, and are exactly the row index
    into this registry's mask table — so `masks_for()` is a single fancy-index gather rather than
    a per-entity loop (CLAUDE.md §2.3, §8.4), and registering a new species costs one appended
    row, never a reallocation of anything entities-shaped.
    """

    def __init__(self, vocabulary: GeneVocabulary) -> None:
        self._vocabulary = vocabulary
        self._mask_table = np.zeros((0, len(vocabulary)), dtype=np.bool_)

    @property
    def n_species(self) -> int:
        return self._mask_table.shape[0]

    def register(self, expressed_genes: tuple[str, ...]) -> int:
        """Register a new species expressing `expressed_genes`; returns its species id."""
        mask = np.zeros(len(self._vocabulary), dtype=np.bool_)
        for name in expressed_genes:
            mask[self._vocabulary.index_of(name)] = True

        species_id = self.n_species
        self._mask_table = np.vstack([self._mask_table, mask])
        return species_id

    def derive(self, parent_species_id: int) -> int:
        """Register a daughter species carrying a copy of `parent_species_id`'s mask; returns its id.

        The primitive `core.genetics.speciation.split` is built on: a population that has drifted
        beyond the distance threshold becomes a separate species without any change to what its
        members express. The copy is what makes the two masks independently editable afterwards,
        so the branches can diverge in expression later without one rewriting the other's
        phenotype.
        """
        mask = self.mask_of(parent_species_id).copy()
        species_id = self.n_species
        self._mask_table = np.vstack([self._mask_table, mask])
        return species_id

    def mask_of(self, species_id: int) -> np.ndarray:
        """(n_genes,) bool: the expression mask for one species id."""
        if not 0 <= species_id < self.n_species:
            raise UnknownSpeciesError(f"species {species_id} is not registered")
        return self._mask_table[species_id]

    def masks_for(self, species_ids: np.ndarray) -> np.ndarray:
        """(len(species_ids), n_genes) bool: each row is that entity's species' expression mask.

        A single fancy-index gather over the mask table, so masking a mixed-species selection of
        any size is one vectorized pass regardless of how many distinct species are present
        (CLAUDE.md §2.3) — never a per-species loop.
        """
        species_ids = np.asarray(species_ids, dtype=np.int64)
        out_of_range = species_ids.size and (
            (species_ids < 0).any() or (species_ids >= self.n_species).any()
        )
        if out_of_range:
            raise UnknownSpeciesError("one or more species ids are not registered")
        return self._mask_table[species_ids]
