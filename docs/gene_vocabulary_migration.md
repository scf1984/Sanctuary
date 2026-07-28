# Gene vocabulary migration procedure

Tracks issue #13.

## Why

`GeneVocabulary` (`core/genetics/vocabulary.py`) is versioned and additive-only (CLAUDE.md §2.3): a
gene's column index is fixed for the life of that vocabulary version, because every entity's
`genes` row and every `SpeciesRegistry` mask is a plain array positioned against that index, with
no gene names stored alongside them. Adding a gene is not a code change alone — it changes the
shape every existing world's gene matrix must have. This document is the procedure for widening a
running or snapshotted world from vocabulary version `N` to `N+1`.

## What changes and what does not

- `GeneVocabulary.widen(*new_names)` returns version `N+1`: version `N`'s names, in the same
  order, plus the new names appended. It never mutates the version-`N` instance.
- Every existing gene's column index is unchanged. Nothing is reordered or removed — the
  vocabulary has no operation to do either, precisely so this cannot happen by accident.
- Every `SpeciesRegistry` mask built against version `N` is `(len(vocabulary),)` bool. Widened to
  `N+1`, it must grow to the new length with the new columns `False` — a species says nothing
  about a gene it predates, and defaulting to unexpressed (rather than expressed) is the safe
  reading, since expressing an untested new gene by default could silently change every existing
  species' phenotype the moment the migration runs.
- Every entity's `genes` row must grow from `(N_genes,)` to `(N_genes + len(new_names),)`. The new
  columns' initial values are a per-gene modelling decision (typically the new gene's population
  baseline), not something this procedure can supply generically.

## Procedure

1. Load the world's stored vocabulary version and confirm it is `N` (the version the running code
   still recognizes as current-minus-one). A snapshot claiming a version the code has no path from
   is a migration gap, not something to guess through — stop and add the missing step instead of
   coercing the data.
2. Compute `vocabulary_next = vocabulary_N.widen(*new_gene_names)`.
3. Widen every `SpeciesRegistry` mask: rebuild each mask as
   `np.concatenate([old_mask, np.zeros(len(new_gene_names), dtype=bool)])`. Order matters — this
   only produces the right mask because `widen()` guarantees the new names are appended, never
   interleaved.
4. Widen the entity store's `genes` column: allocate a new `(capacity, len(vocabulary_next))`
   array, copy the existing `(capacity, N_genes)` block into its low columns unchanged, and fill
   the new columns with the chosen baseline value for each new gene. This is the same shape of
   operation as `EntityStore.grow()` (`core/entities/store.py`) — replace, copy, never resize in
   place — for the same reason: a live NumPy view into the old array must not be mutated out from
   under whatever holds it mid-tick.
5. Persist `vocabulary_next`'s version alongside the migrated snapshot so a later load does not
   re-run this step against already-migrated data.
6. Do not delete the pre-migration snapshot until the migrated world has been loaded and its
   invariants (CLAUDE.md §6) checked at least once.

## What this procedure does not cover

Snapshot persistence itself (#31) does not exist yet, so there is no concrete file format for step
1/5 to read and write against today. This procedure describes the required steps in terms of the
data structures that do exist (`GeneVocabulary`, `SpeciesRegistry`, `EntityStore.genes`) so that
#31, once built, has a specification to implement against rather than needing to design gene
migration itself.
