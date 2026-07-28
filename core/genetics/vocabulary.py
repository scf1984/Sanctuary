"""Gene vocabulary: the ordered, versioned set of gene slots every creature shares (CLAUDE.md
§2.3).

Every entity has a value for every gene in the vocabulary, expressed or not (see
`core.genetics.species`). A gene's position is its column index into the store's `genes` matrix,
so the vocabulary is additive-only: adding a gene appends a column, but removing or reordering one
would change what an existing column index means, silently corrupting every world snapshot built
against the old version with nothing to flag it (CLAUDE.md §8.7). Migrating a world across
versions is documented in docs/gene_vocabulary_migration.md.
"""

from __future__ import annotations


class DuplicateGeneError(Exception):
    """A gene name was declared twice in the same vocabulary."""


class GeneVocabulary:
    """An ordered, versioned list of gene names; column index i is names[i]'s slot in the matrix.

    names: tuple[str, ...], in column order. A name's position never changes once assigned.
    version: int, starting at 1. Incremented only by widen() — constructing a vocabulary directly
        does not imply a new version relative to any other.
    """

    def __init__(self, names: tuple[str, ...], version: int = 1) -> None:
        if not names:
            raise ValueError("a vocabulary must declare at least one gene")
        seen: set[str] = set()
        for name in names:
            if name in seen:
                raise DuplicateGeneError(f"gene '{name}' declared twice")
            seen.add(name)

        self.names = tuple(names)
        self.version = version
        self._index_by_name = {name: i for i, name in enumerate(self.names)}

    def __len__(self) -> int:
        return len(self.names)

    def index_of(self, name: str) -> int:
        """The column index of `name` in the gene matrix."""
        try:
            return self._index_by_name[name]
        except KeyError:
            raise KeyError(f"'{name}' is not in gene vocabulary v{self.version}") from None

    def widen(self, *new_names: str) -> GeneVocabulary:
        """The next version: this vocabulary's genes, in the same order, plus `new_names`.

        Never mutates self — a world built against this version keeps referencing it until
        something explicitly migrates it forward (docs/gene_vocabulary_migration.md). Existing
        gene columns keep their index; new genes are only ever appended.
        """
        if not new_names:
            raise ValueError("widen() requires at least one new gene name")
        return GeneVocabulary(self.names + tuple(new_names), version=self.version + 1)
