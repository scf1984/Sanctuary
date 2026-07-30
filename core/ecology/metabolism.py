"""Metabolic upkeep: what one tick of being alive costs, given a phenotype and a temperature
(CLAUDE.md §2.5, issue #17).

This is the mechanism that stops every animal evolving toward maximum everything. Speed, size,
sight range and the rest all draw from **one** metabolic pool, continuously, so a trait's benefit
is never free and the environment — not a designer — picks which build is worth its upkeep.

Two rules give this module its shape:

- **Cost follows expression, not genotype.** Upkeep is a function of the *expressed* phenotype
  (`core.genetics.service.Genetics.expressed`), so a species that does not express a gene neither
  pays for it nor gains from it, while still carrying and inheriting it (CLAUDE.md §2.3). Cost and
  benefit are therefore inseparable by construction rather than by convention.
- **Every gene in the vocabulary must declare a cost**, even if that cost is zero. A gene added to
  the vocabulary without an entry here fails at construction (§8.7) instead of quietly becoming a
  free trait, which is the one failure mode that would defeat the entire hard-budget design.
- **Only a gene read as a magnitude may carry a cost** (#136). A cost is bounded below by zero only
  if the phenotype it multiplies is, and what guarantees that is the gene's *expression mode*, not
  inheritance — storage is signed (§2.5, #104). A `SIGNED` gene is a cue-space direction, founded
  across zero by design, so a positive cost on one contributes a **negative** term to a sum: the
  animal is charged less for pointing its aversion one way round than the other, and selection
  acts on the discount. That is §2.5's hard budget running in reverse, and it is knowable from the
  two config tables alone, so it fails here rather than in a world.

That last rule is why this module is handed expression modes it otherwise has no use for. It reads
them once, at construction, and never during a tick — `Genetics.expressed` is what applies a mode
to a value, and `upkeep` below receives the result. #111 folds the cost table and the mode table
into one gene registry, at which point the two can no longer be supplied separately and this check
moves there; until then they are two mappings that must agree, and this is where they are made to.

The math is deliberately store-free: it takes phenotype rows and temperatures and returns energy
units per tick. `core.ecology.service.Ecology` is what binds it to the entity store, the climate
field and the energy column.

Coefficients are per-world configuration, never constants in this module — the numbers that make
an ecology legible are tuning, and CLAUDE.md §2.1 requires them tuned as a table rather than
scattered as literals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.genetics.registry import GeneRegistry, Unit


@dataclass(frozen=True)
class MetabolismConfig:
    """Per-world metabolic cost table. Every rate is energy units per tick.

    Per-gene costs are not here: they are per-gene declarations and live on each `GeneSpec` in
    `core.genetics.registry` (#111), beside the expression mode they have to agree with.

    basal_rate: energy units per tick charged to every entity regardless of phenotype. Without it,
        a species expressing only zero-cost genes would pay nothing to stay alive and could never
        starve, which is a free lunch reached by expressing less rather than by evolving more.
    thermoregulation_rate: energy units per tick per degree C of deviation from
        `neutral_temperature`, for an entity with no insulation.
    neutral_temperature: degrees C at which thermoregulation costs nothing. Deviation is
        unsigned — holding a body above a cold world or below a hot one are both work.
    insulation_gene: the gene whose expressed value damps thermoregulation cost. It must itself
        carry a positive cost, because it only ever *reduces* upkeep: a free insulation gene is
        unbounded free benefit and would be selected upward without limit in every climate,
        including the ones it does nothing for.
    """

    basal_rate: float
    thermoregulation_rate: float
    neutral_temperature: float
    insulation_gene: str

    def __post_init__(self) -> None:
        if self.basal_rate < 0:
            raise ValueError(f"basal_rate must be non-negative, got {self.basal_rate}")
        if self.thermoregulation_rate < 0:
            raise ValueError(
                f"thermoregulation_rate must be non-negative, got {self.thermoregulation_rate}"
            )


class Metabolism:
    """A gene registry's costs, plus the whole-body rates, applied to phenotype blocks.

    gene_cost: (n_genes,) float32, energy units per tick per unit of expressed gene value, in
        vocabulary column order — so trait upkeep for any number of entities is one matrix-vector
        product rather than a per-gene or per-species loop (CLAUDE.md §2.3). Taken from the
        registry rather than rebuilt from a second table.

    The checks that used to run here are gone rather than moved (#111). A gene cannot omit a cost
    or declare a negative one, because a `GeneSpec` carries both a cost and its own validation; and
    the costed-but-signed check (#136) now runs where the two facts live together, instead of this
    module taking a modes mapping it consulted once and never read again.
    """

    def __init__(self, registry: GeneRegistry, config: MetabolismConfig) -> None:
        self.config = config
        self.gene_cost = registry.cost

        # Raises KeyError naming the vocabulary version if the gene does not exist. Insulation is a
        # bare damping factor on a cost, so it carries no dimension of its own.
        self._insulation_index = registry.index_of(
            config.insulation_gene, unit=Unit.DIMENSIONLESS
        )
        if self.gene_cost[self._insulation_index] <= 0:
            raise ValueError(
                f"insulation gene '{config.insulation_gene}' must carry a positive cost; "
                "a gene that only reduces upkeep and charges nothing is a free lunch"
            )

    def upkeep(self, expressed_genes: np.ndarray, temperature: np.ndarray) -> np.ndarray:
        """(n,) float32, energy units per tick: what one tick of life costs each of `n` entities.

        expressed_genes: (n, n_genes) float32 phenotype rows — unexpressed slots already zeroed
            by the caller, which is what makes an unexpressed gene cost nothing.
        temperature: (n,) float32, degrees C, sampled at each entity's own position.

        Trait upkeep is the phenotype dotted with the cost table. Thermoregulation is the work of
        holding a body away from ambient: proportional to how far the world is from
        `neutral_temperature`, and damped by insulation with diminishing returns — the `1 +`
        keeps an uninsulated animal at the full cost rather than at infinity, and makes each
        further unit of insulation buy less than the last, so insulation trades off against its
        own upkeep instead of running away.

        Never negative, which is what lets `Ecology.drain` claim it only ever removes energy.
        **What guarantees that is the gene's expression mode, not inheritance** (#104): storage is
        signed, and a gene declared `MAGNITUDE` folds across zero when `Genetics.expressed` reads
        it, so a costed quantity arrives here non-negative however far its stored value has
        drifted. A gene declared `SIGNED` arrives as stored and may well be negative — which is
        why the constructor refuses to let one carry a cost, and why that refusal is what this
        guarantee rests on rather than any check in here (#136).
        """
        trait_upkeep = expressed_genes @ self.gene_cost
        insulation = expressed_genes[:, self._insulation_index]
        thermoregulation = (
            self.config.thermoregulation_rate
            * np.abs(temperature - self.config.neutral_temperature)
            / (1.0 + insulation)
        )
        return (self.config.basal_rate + trait_upkeep + thermoregulation).astype(np.float32)
