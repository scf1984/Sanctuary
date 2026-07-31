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

**The modes are declared in `core.genetics.registry` and applied here** (#111). That split is the
layering: a `GeneSpec` states a gene's mode alongside its cost and unit, so the three cannot drift
apart, and this module is where a stored value actually becomes a phenotype. Nothing here validates
that every gene has a mode any more, because a gene without one cannot be constructed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.genetics.registry import ExpressionMode, GeneRegistry, Unit


@dataclass(frozen=True)
class GeneticsConfig:
    """Per-world genetics rules — never constants in `core/` (§2.1).

    mutability_gene: the gene whose value floors the spread of an offspring's draw
        (`core.genetics.inheritance`). Must be declared `MAGNITUDE`: it is the width of a
        distribution, so what matters is its size and not its sign.
    drift_margin: how far outside the parental min/max an offspring may land, in units of the draw's
        own spread. Must be positive; zero forbids drift outright.

    The modes themselves are not here: they are per-gene declarations and live on each `GeneSpec`
    in `core.genetics.registry` (#111).
    """

    mutability_gene: str
    drift_margin: float

    def __post_init__(self) -> None:
        if self.drift_margin <= 0.0:
            raise ValueError(f"drift_margin must be positive, got {self.drift_margin}")


class ExpressionTable:
    """A registry's modes, applied to phenotype blocks.

    magnitude_columns: (n_genes,) bool, True where the gene is read as a magnitude, in vocabulary
        column order — so applying every mode to a phenotype block is a handful of whole-array
        operations rather than a per-gene loop (§2.3). Taken from the registry rather than rebuilt,
        since a second copy resolved separately is the disagreement #111 exists to prevent.
    exponential_columns: (n_genes,) bool, True where the gene is read as `exp(value)`.
    unit_interval_columns: (n_genes,) bool, True where the gene is read as an allocation on (0, 1).
    mutability_index: int, the column `inherit()` reads the draw's spread floor from.
    """

    def __init__(self, registry: GeneRegistry, config: GeneticsConfig) -> None:
        self.config = config
        self.magnitude_columns = registry.magnitude_columns
        self.exponential_columns = registry.exponential_columns
        self.unit_interval_columns = registry.unit_interval_columns
        # Hoisted so `phenotype` — called every tick — does not reduce a boolean array per call.
        self._has_exponential = bool(self.exponential_columns.any())
        self._has_unit_interval = bool(self.unit_interval_columns.any())

        # Raises KeyError naming the vocabulary version if the gene does not exist. The spread of
        # a distribution is a bare number, hence dimensionless.
        self.mutability_index = registry.index_of(config.mutability_gene, unit=Unit.DIMENSIONLESS)
        if registry.spec(config.mutability_gene).expression_mode is not ExpressionMode.MAGNITUDE:
            raise ValueError(
                f"mutability gene '{config.mutability_gene}' must be read as a magnitude; "
                "a signed spread would make an offspring's draw scale negative"
            )

    def phenotype(self, raw: np.ndarray) -> np.ndarray:
        """(n, n_genes) float32: `raw` gene rows read through each gene's mode.

        Magnitude genes fold across zero, exponential genes are raised through `exp`, unit-interval
        genes are squashed into (0, 1), and signed genes pass through.
        Species expression is *not* applied here — that is `Genetics.expressed`'s job, and it runs
        after this rather than before it, because a mode maps an unexpressed gene's stored value to
        something that is not necessarily zero and masking first would express half of it.
        """
        phenotype = np.where(self.magnitude_columns, np.abs(raw), raw).astype(np.float32)
        # Applied to the exponential columns alone rather than to the whole block and selected
        # from. `exp` over every column overflows on any gene holding a large value — a cue
        # signature is unbounded and read raw — which raises a RuntimeWarning per tick and
        # computes `inf` only to discard it.
        if self._has_exponential:
            columns = self.exponential_columns
            phenotype[:, columns] = np.exp(phenotype[:, columns])
        if self._has_unit_interval:
            columns = self.unit_interval_columns
            # The logistic, written as `0.5 * (1 + tanh(x/2))` rather than `1 / (1 + exp(-x))`.
            # They are the same function, but the direct form overflows `exp` on any gene that has
            # drifted well below zero — a RuntimeWarning every tick, and `inf` computed only to be
            # divided away. `tanh` saturates at ±1 instead, so an extreme allocation costs nothing
            # and still lands strictly inside the interval.
            phenotype[:, columns] = 0.5 * (1.0 + np.tanh(0.5 * phenotype[:, columns]))
        return phenotype
