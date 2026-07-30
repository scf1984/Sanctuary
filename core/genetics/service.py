"""Genetics domain service: vectorized gene access and species expression (CLAUDE.md §2.3, §4).

Owns the `genes` and `species_id` columns on the shared entity store. Genes are always read and
written in full, expressed or not — `expressed()` is the only place a species' mask is applied, so
nothing in this module ever zeroes, drops, or otherwise touches an unexpressed gene's stored value.
That is what lets a dormant gene resurface generations later if a descendant's species comes to
express it again (CLAUDE.md §2.3).

Trait inheritance with mutation and drift-clamping (`inherit()`) delegates its math to
`core.genetics.inheritance`, which has no notion of the store or selections -- this module's job
is only to resolve parent selections to gene rows and back (#14). The same split applies to how a
stored value is read as a phenotype: `core.genetics.expression` owns the modes, and this service
applies them at the one place a phenotype is produced (#104).
"""

from __future__ import annotations

import numpy as np

from core.entities.store import EntityStore
from core.genetics.expression import ExpressionTable, GeneticsConfig
from core.genetics.inheritance import inherit_genes
from core.genetics.registry import GeneRegistry
from core.genetics.species import SpeciesRegistry
from core.selection import Selection
from core.services import ColumnRegistry, DomainService


class Genetics(DomainService):
    """Reads and writes gene rows and species assignment for selections of entities.

    species: the SpeciesRegistry whose ids are valid values for the store's `species_id` column.
        Shared with whatever else in a world needs to resolve species masks — not owned by this
        service (CLAUDE.md §4: no singletons, pass context explicitly).
    expression: the registry's expression modes and mutability column, resolved once
        (`core.genetics.expression`). Built here rather than passed in because it is derived data,
        and a second copy resolved against a different vocabulary is exactly the disagreement #111
        made impossible — the modes now come from the same `GeneSpec` table the costs do.
    """

    owns = ("genes", "species_id")

    # Narrows DomainService.store (typed `object` since a service base is store-shape-agnostic)
    # to the concrete EntityStore this service actually reads `genes`/`species_id` off of.
    store: EntityStore

    def __init__(
        self,
        store: EntityStore,
        registry: ColumnRegistry,
        species: SpeciesRegistry,
        genes: GeneRegistry,
        config: GeneticsConfig,
    ) -> None:
        super().__init__(store, registry)
        self.species = species
        self.config = config
        self.expression = ExpressionTable(genes, config)

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

        The one exception to reading genotype raw is the **mutability** floor, which is read through
        its magnitude mode: it is the width of the offspring's draw, so a lineage whose stored value
        has drifted below zero must still have a spread, and its size is what carries the meaning
        (#104). The floor is the mean of the two parents' *magnitudes* and not the magnitude of their
        mean — mutability at +1 and -1 is two mutable parents, and averaging before folding would
        cancel them into an offspring that cannot vary at all.
        """
        if len(parent_a) != len(parent_b):
            raise ValueError(
                f"parent selections must have equal length: {len(parent_a)} vs {len(parent_b)}"
            )
        genes_a = self.genes(parent_a)
        genes_b = self.genes(parent_b)
        column = self.expression.mutability_index
        mutability = (np.abs(genes_a[:, column]) + np.abs(genes_b[:, column])) / 2.0
        return inherit_genes(
            genes_a, genes_b, mutability, self.config.drift_margin, rng
        )

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
        """(len(selection), n_genes) float32: gene values read through their expression modes, with
        unexpressed slots zeroed.

        This is phenotype, not genotype — both the mode and the mask are applied here, at read time,
        and never by mutating storage. `genes()` above always returns the full genotype regardless of
        species. Mode first, mask second: the mask is what makes an unexpressed gene contribute
        nothing, so it must have the last word on the value handed out (#104).
        """
        mask = selection.to_mask()
        phenotype = self.expression.phenotype(self.store.genes[mask])
        species_mask = self.species.masks_for(self.store.species_id[mask])
        return np.where(species_mask, phenotype, 0.0).astype(np.float32)
