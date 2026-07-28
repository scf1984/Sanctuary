"""Genetics domain service: vectorized gene access and species expression (CLAUDE.md §2.3, §4).

Owns the `genes` and `species_id` columns on the shared entity store. Genes are always read and
written in full, expressed or not — `expressed()` is the only place a species' mask is applied, so
nothing in this module ever zeroes, drops, or otherwise touches an unexpressed gene's stored value.
That is what lets a dormant gene resurface generations later if a descendant's species comes to
express it again (CLAUDE.md §2.3).

Trait inheritance with mutation and drift-clamping (`inherit()`) delegates its math to
`core.genetics.inheritance`, which has no notion of the store or selections -- this module's job
is only to resolve parent selections to gene rows and back (#14).
"""

from __future__ import annotations

import numpy as np

from core.entities.store import EntityStore
from core.genetics.inheritance import inherit_genes
from core.genetics.species import SpeciesRegistry
from core.selection import Selection
from core.services import ColumnRegistry, DomainService


class Genetics(DomainService):
    """Reads and writes gene rows and species assignment for selections of entities.

    species: the SpeciesRegistry whose ids are valid values for the store's `species_id` column.
        Shared with whatever else in a world needs to resolve species masks — not owned by this
        service (CLAUDE.md §4: no singletons, pass context explicitly).
    """

    owns = ("genes", "species_id")

    # Narrows DomainService.store (typed `object` since a service base is store-shape-agnostic)
    # to the concrete EntityStore this service actually reads `genes`/`species_id` off of.
    store: EntityStore

    def __init__(
        self, store: EntityStore, registry: ColumnRegistry, species: SpeciesRegistry
    ) -> None:
        super().__init__(store, registry)
        self.species = species

    def genes(self, selection: Selection) -> np.ndarray:
        """(len(selection), n_genes) float32: raw gene values, expressed or not."""
        return self.store.genes[selection.to_mask()]

    def set_genes(self, selection: Selection, values: np.ndarray) -> None:
        """Vectorized write of full gene rows for `selection` — e.g. seeding a new population."""
        self.write("genes", selection, values)

    def inherit(
        self,
        parent_a: Selection,
        parent_b: Selection,
        inherit_gain: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """(len(parent_a), n_genes) float32: one offspring gene row per parent pair.

        Pairs rows by ascending row-index order within each selection -- the same order `genes()`
        itself reads in -- so `parent_a` and `parent_b` must have equal length and it is the
        caller's responsibility to construct them so row i of one is the intended mate of row i
        of the other.

        Reads full genotypes via `genes()`, not `expressed()`: an unexpressed gene inherits (and
        can mutate) exactly like an expressed one (CLAUDE.md §2.3, #13's "done when"). This only
        computes the offspring's gene values -- writing them into a newly allocated entity is the
        caller's job, via `set_genes()` or `EntityStore.allocate(..., genes=...)`.
        """
        if len(parent_a) != len(parent_b):
            raise ValueError(
                f"parent selections must have equal length: {len(parent_a)} vs {len(parent_b)}"
            )
        return inherit_genes(self.genes(parent_a), self.genes(parent_b), inherit_gain, rng)

    def species_ids(self, selection: Selection) -> np.ndarray:
        """(len(selection),) int32: each entity's species id, in ascending row order.

        Species ids are opaque registry keys, not row indices, so handing them out does not leak
        storage layout across the service boundary (CLAUDE.md §2.3). Speciation needs this to
        gate interbreeding on whether two creatures are still the same species (#16).
        """
        return self.store.species_id[selection.to_mask()]

    def speciate(self, selection: Selection, species_id: int) -> None:
        """Assign `species_id` to every entity in `selection`.

        This is the entirety of speciation (CLAUDE.md §2.3): no gene value changes and no entity
        row moves. The new phenotype comes from `species_id`'s mask the next time `expressed()`
        is read, not from any restructuring here.
        """
        self.write(
            "species_id",
            selection,
            np.full(len(selection), species_id, dtype=np.int32),
        )

    def expressed(self, selection: Selection) -> np.ndarray:
        """(len(selection), n_genes) float32: gene values with unexpressed slots zeroed.

        This is phenotype, not genotype — the mask is applied here, at read time, and never by
        mutating storage. `genes()` above always returns the full genotype regardless of species.
        """
        mask = selection.to_mask()
        raw = self.store.genes[mask]
        species_mask = self.species.masks_for(self.store.species_id[mask])
        return np.where(species_mask, raw, 0.0).astype(np.float32)
