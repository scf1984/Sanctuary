"""Senescence: performance traits degrade with age (CLAUDE.md §2.5, #21's design note, #239).

**Degradation, not a timer.** There is no lifespan gene and no death clock. An old animal's
performance traits — speed, sight and scent acuity — are read at a fraction of their genetic value,
so it catches less, escapes less, and eventually cannot cover its own upkeep. Death then falls out
of starvation and predation, which already exist; nothing here kills anything.

**The formula is what holds the property, not a clamp:**

    rate  = base_decay / (1 + senescence_resistance)     base_decay > 0, validated below
    decay = exp(-rate x age)                             always in (0, 1]

`senescence_resistance` is read as a magnitude, so the denominator is never below 1; the `1 +` gives
it diminishing returns, so the gene cannot buy its way out of the mechanism into immortality — the
runaway that got a lifespan gene rejected (§2.5). It reduces degradation, so it charges positive
upkeep like the thermal tolerances, which is the world's (`GeneSpec.cost`), not this module's.

This module is the curve and its configuration. Where it applies — at expression time, alongside
the species mask — is `core.genetics.service`, because that is the one place a phenotype is made.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SenescenceConfig:
    """Per-world senescence rules — never constants in `core/` (§2.1).

    resistance_gene: the gene damping how fast performance traits degrade. Must be read as a
        magnitude, so the decay rate's denominator is never below 1.
    base_decay: decay rate per tick lived at zero resistance, in 1/ticks. Must be positive and
        finite: zero would be immortality, which is the outcome the mechanism exists to rule out.
        Tuned toward §2.1's lifespan row as an *outcome*, never set as one.
    decaying_genes: the performance traits that degrade. Each must be a quantity (magnitude): decay
        scales a trait toward zero, and zero is only "less of it" for a quantity. Identity traits
        (cue signature) and capacity traits (size) are deliberately not here (§2.5).
    """

    resistance_gene: str
    base_decay: float
    decaying_genes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not (math.isfinite(self.base_decay) and self.base_decay > 0.0):
            raise ValueError(
                f"base_decay must be positive and finite, got {self.base_decay}; zero would mean "
                "nothing ages, which is immortality by configuration"
            )
        if len(set(self.decaying_genes)) != len(self.decaying_genes):
            raise ValueError(f"a gene is listed to decay twice: {self.decaying_genes}")


def decay(age: np.ndarray, resistance: np.ndarray, base_decay: float) -> np.ndarray:
    """(n,) float64 in (0, 1]: the fraction of each performance trait an animal still expresses.

    age: (n,) int64 ticks, from the store's `age` column. A gestating row carries a negative age
        (#20) — that is the gestation clock counting toward birth, not years lived — so it is read
        as zero: an unborn animal has not aged, and a negative age must not read as rejuvenation.
    resistance: (n,) the expressed resistance gene, read as a magnitude.
    """
    lived = np.maximum(np.asarray(age, dtype=np.float64), 0.0)
    rate = base_decay / (1.0 + np.abs(np.asarray(resistance, dtype=np.float64)))
    return np.exp(-rate * lived)
