"""Genetic distance: expressed-phenotype Euclidean distance between creatures and between
population centroids (CLAUDE.md §2.5, issue #15).

Speciation triggers on accumulated distance between isolated populations, so the metric must be
well-behaved rather than ad hoc. Both functions below are ordinary Euclidean distance
(``numpy.linalg.norm`` of a difference) over vectors from `Genetics.expressed()` — and Euclidean
distance is a norm-induced metric, so symmetry and the triangle inequality hold for *any* input
vectors, not just well-behaved ones. Nothing here needs to re-derive those properties; they are
inherited from picking a real metric instead of an arbitrary similarity score.

Unexpressed-gene treatment: distance is computed over `expressed()`, which zeroes a creature's
unexpressed gene slots at read time (CLAUDE.md §2.3), never over the raw `genes()` genotype. Two
creatures that silently carry the same dormant gene value are not "closer" for it — dormancy has no
phenotypic or reproductive consequence until something re-expresses the gene, and reproductive
incompatibility is exactly what this metric feeds into (speciation). Distance therefore tracks how
far apart two creatures' *actual* traits have drifted, which is also why comparing across two
species with different expression masks needs no special case: each side's own mask already zeroed
what it doesn't express, so the same Euclidean formula applies uniformly.
"""

from __future__ import annotations

import numpy as np

from core.genetics.service import Genetics
from core.selection import Selection


def between(genetics: Genetics, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(len(a),) float32: Euclidean distance between each row of `a` and the paired row of `b`.

    `a` and `b` are **row index arrays**, paired elementwise: row `a[i]` against row `b[i]`. They
    take indices rather than `Selection`s because a selection is a mask and therefore carries no
    order — two masks can only express a pairing whose couples do not cross in row space, and a
    pairing built from position crosses constantly. The old form rewired such a pairing silently
    instead of refusing it (#20).

    This is a positional pairing of two equal-length arrays, not a claim that the two entities at
    position i are otherwise related.
    """
    phenotype_a = genetics.expressed_at(a)
    phenotype_b = genetics.expressed_at(b)
    if phenotype_a.shape[0] != phenotype_b.shape[0]:
        raise ValueError(
            f"a and b must select the same number of rows "
            f"({phenotype_a.shape[0]} vs {phenotype_b.shape[0]})"
        )
    return np.linalg.norm(phenotype_a - phenotype_b, axis=1).astype(np.float32)


def centroid_between(genetics: Genetics, a: Selection, b: Selection) -> np.float32:
    """Euclidean distance between the mean expressed phenotype of `a` and of `b`.

    `a` and `b` may differ in size and need not be disjoint. Zero exactly when `a` and `b` have
    the same mean expressed phenotype — in particular when `a` and `b` select the same rows, which
    is the "a population against itself" case speciation compares a threshold against.

    Raises ValueError for an empty selection rather than returning NumPy's silent all-NaN mean
    (CLAUDE.md §8.7) — a centroid over zero creatures is not a smaller-magnitude answer, it is
    undefined.
    """
    if len(a) == 0 or len(b) == 0:
        raise ValueError("centroid_between requires both selections to be non-empty")
    centroid_a = genetics.expressed(a).mean(axis=0)
    centroid_b = genetics.expressed(b).mean(axis=0)
    return np.float32(np.linalg.norm(centroid_a - centroid_b))
