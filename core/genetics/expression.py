"""How a stored gene value becomes a phenotype (CLAUDE.md §2.5, issue #104).

Genes live on ℝ. That is what lets drift be symmetric — an additive margin around the parental mean
has no reason to prefer one half-line, and the multiplicative range it replaced inverted outright
below zero (`core.genetics.inheritance`). But once a gene may be negative, **each gene has to declare
how it is read**, because the answer differs by gene and cannot be guessed from the value:

- a *quantity* — size, speed, acuity, insulation — cannot be negative, and is read as a magnitude
- a *direction* — cue signature, aversion — carries information in its sign, which is what doubles
  the discriminating power of cue space (§2.5), and is read raw

The mode is therefore a property of the vocabulary, alongside a gene's cost, and it is **consulted
rather than merely declared** (§4): `Genetics.expressed` applies it, so nothing downstream has to
know or care that storage is signed.

**This is half of #111's registry, deliberately.** #111 will carry cost, expression mode and unit for
every gene in one place, generated into a readable map so the three cannot drift apart. It is blocked
on this issue for exactly that reason — the modes have to exist before something can hold them — so
`expression_modes` here mirrors the shape `MetabolismConfig.gene_costs` already uses (a per-world
mapping, every gene named, validated against the vocabulary at construction) and #111 folds the two
into one table rather than inventing a third shape.

**There is no unit-interval mode yet.** #104 names a third reading, a squash to [0, 1], for
`sex_allocation` and `selfing_rate` — genes that do not exist until #99, which is blocked by #20.
Adding the member now would be a mode nothing declares (§8.2), and adding one later is additive.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import numpy as np

from core.genetics.vocabulary import GeneVocabulary


class ExpressionMode(Enum):
    """How one gene's stored value is read as a phenotype."""

    MAGNITUDE = "magnitude"
    """A quantity: read as `abs(value)`. Negative storage is meaningless for a body, so drift across
    zero folds back rather than producing a negative size."""

    SIGNED = "signed"
    """A direction in cue space: read as stored. Sign is information, not a mistake."""


@dataclass(frozen=True)
class GeneticsConfig:
    """Per-world genetics rules — never constants in `core/` (§2.1).

    expression_modes: gene name -> how it is read. Must name every gene in the vocabulary exactly
        once; absence raises, because a gene with no declared reading would silently be taken as
        signed, and a signed `size` is a body with negative mass that also earns upkeep back (#136).
    mutability_gene: the gene whose value floors the spread of an offspring's draw
        (`core.genetics.inheritance`). Must be declared `MAGNITUDE`: it is the width of a
        distribution, so what matters is its size and not its sign.
    drift_margin: how far outside the parental min/max an offspring may land, in units of the draw's
        own spread. Must be positive; zero forbids drift outright.
    """

    expression_modes: Mapping[str, ExpressionMode]
    mutability_gene: str
    drift_margin: float

    def __post_init__(self) -> None:
        if self.drift_margin <= 0.0:
            raise ValueError(f"drift_margin must be positive, got {self.drift_margin}")


class ExpressionTable:
    """A `GeneticsConfig`'s modes resolved against one vocabulary into vectorized column masks.

    magnitude_columns: (n_genes,) bool, True where the gene is read as a magnitude, in vocabulary
        column order — so applying every mode to a phenotype block is a handful of whole-array
        operations rather than a per-gene loop (§2.3).
    mutability_index: int, the column `inherit()` reads the draw's spread floor from.
    """

    def __init__(self, vocabulary: GeneVocabulary, config: GeneticsConfig) -> None:
        declared = set(config.expression_modes)
        known = set(vocabulary.names)
        unknown = sorted(declared - known)
        if unknown:
            raise ValueError(f"expression modes name genes outside the vocabulary: {unknown}")
        undeclared = sorted(known - declared)
        if undeclared:
            raise ValueError(
                f"every gene must declare an expression mode; missing: {undeclared}"
            )

        self.config = config
        self.magnitude_columns = np.array(
            [
                config.expression_modes[name] is ExpressionMode.MAGNITUDE
                for name in vocabulary.names
            ],
            dtype=bool,
        )

        # Raises KeyError naming the vocabulary version if the gene does not exist.
        self.mutability_index = vocabulary.index_of(config.mutability_gene)
        if config.expression_modes[config.mutability_gene] is not ExpressionMode.MAGNITUDE:
            raise ValueError(
                f"mutability gene '{config.mutability_gene}' must be read as a magnitude; "
                "a signed spread would make an offspring's draw scale negative"
            )

    def phenotype(self, raw: np.ndarray) -> np.ndarray:
        """(n, n_genes) float32: `raw` gene rows read through each gene's mode.

        Magnitude genes fold across zero and signed genes pass through, which is the whole of it.
        Species expression is *not* applied here — that is `Genetics.expressed`'s job, and it runs
        after this rather than before it, because a mode maps an unexpressed gene's stored value to
        something that is not necessarily zero and masking first would express half of it.
        """
        return np.where(self.magnitude_columns, np.abs(raw), raw).astype(np.float32)
