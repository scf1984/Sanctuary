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
    thermoregulation_rate: energy units per tick per **squared** temperature unit of deviation from
        `neutral_temperature`, for an entity with no tolerance on that side. Quadratic rather than
        linear (#176): a linear V pushes as hard a hair off neutral as far from it, which pins a
        population's tolerances to a point and leaves selection no standing variation to act on,
        while a square is gentle near neutral and steep far out, so extreme ground stays a graded
        barrier and a stranded lineage is pulled back hard. Because the unit changed, a rate tuned
        for the linear form must be re-derived, not carried over: the demo world divides its old
        linear rate by the mean deviation its animals actually stand in, so the typical bill is
        unchanged and only its shape moves.
    dehydration_penalty: how much more a completely dry animal costs to run, as a multiple of its
        ordinary upkeep — 0 means dehydration is free and the mechanic does not exist, 10 means a
        dry animal pays eleven times over. **This is the whole of how thirst kills** (#156): there
        is no dehydration death check anywhere, only a pool that empties faster. Must be tuned
        against `HydrationConfig.loss_rate` as one pair (§2.1) — the two together decide how many
        ticks from a full reserve to death, and either alone says nothing.
    neutral_temperature: temperature units at which thermoregulation costs nothing. One per world, not a
        gene: what a lineage evolves is how well it bears each side of it, not where it sits.
    heat_tolerance_gene: the gene whose expressed value damps the cost of being **hotter** than
        neutral, and only that side.
    cold_tolerance_gene: the same for being **colder** than neutral.

        Two genes rather than one symmetric `insulation` damper (#176). A symmetric damper prices
        5 and 35 identically around a neutral of 20, so an arctic and a desert lineage evolve
        toward the same animal and climate gives speciation nothing to diverge on. With one gene
        per side, a hot cell selects for heat tolerance and against cold tolerance, a cold cell the
        reverse, and a hybrid carries both at intermediate values: dead weight on one side and
        too little on the other, wherever it lives.

        Each must carry a positive cost, because each only ever *reduces* upkeep: a free one is
        unbounded free benefit and would be selected upward without limit, including in climates
        where it does nothing. They must also be different genes, or the config has quietly
        rebuilt the symmetric damper.
    """

    basal_rate: float
    thermoregulation_rate: float
    dehydration_penalty: float
    neutral_temperature: float
    heat_tolerance_gene: str
    cold_tolerance_gene: str

    def __post_init__(self) -> None:
        if self.basal_rate < 0:
            raise ValueError(f"basal_rate must be non-negative, got {self.basal_rate}")
        if self.thermoregulation_rate < 0:
            raise ValueError(
                f"thermoregulation_rate must be non-negative, got {self.thermoregulation_rate}"
            )
        if self.heat_tolerance_gene == self.cold_tolerance_gene:
            raise ValueError(
                "heat and cold tolerance must be different genes; one gene damping both sides is "
                f"the symmetric damper #176 replaced ('{self.heat_tolerance_gene}')"
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

        # Raises KeyError naming the vocabulary version if a gene does not exist. A tolerance is a
        # bare damping factor on a cost, so it carries no dimension of its own.
        self._heat_tolerance_index = registry.index_of(
            config.heat_tolerance_gene, unit=Unit.DIMENSIONLESS
        )
        self._cold_tolerance_index = registry.index_of(
            config.cold_tolerance_gene, unit=Unit.DIMENSIONLESS
        )
        for gene, index in (
            (config.heat_tolerance_gene, self._heat_tolerance_index),
            (config.cold_tolerance_gene, self._cold_tolerance_index),
        ):
            if self.gene_cost[index] <= 0:
                raise ValueError(
                    f"tolerance gene '{gene}' must carry a positive cost; "
                    "a gene that only reduces upkeep and charges nothing is a free lunch"
                )
        # (2,) int, vocabulary indices of [heat tolerance, cold tolerance]: the only columns
        # `thermoregulation` reads, so a caller can build just these instead of the whole phenotype.
        self.thermal_columns = np.array(
            [self._heat_tolerance_index, self._cold_tolerance_index], dtype=np.int64
        )

    def upkeep(self, expressed_genes: np.ndarray, temperature: np.ndarray) -> np.ndarray:
        """(n,) float32, energy units per tick: what one tick of life costs each of `n` entities.

        expressed_genes: (n, n_genes) float32 phenotype rows — unexpressed slots already zeroed
            by the caller, which is what makes an unexpressed gene cost nothing.
        temperature: (n,) float32, temperature units, sampled at each entity's own position.

        Trait upkeep is the phenotype dotted with the cost table. Thermoregulation is the work of
        holding a body away from ambient: the square of how far the world is from
        `neutral_temperature`, damped by the tolerance for *that side* with diminishing returns.
        The `1 +` keeps an untolerant animal at the full cost rather than at infinity, and makes
        each further unit of tolerance buy less than the last, so a tolerance trades off against
        its own upkeep instead of running away.

        The damping divides the squared bill rather than sitting inside the square. Inside, as
        `(deviation / (1 + tolerance)) ** 2`, doubling a tolerance would quarter the bill:
        quadratic returns on a gene that only reduces a cost, which is the runaway §2.5 forbids.

        Never negative, which is what lets `Ecology.drain` claim it only ever removes energy.
        **What guarantees that is the gene's expression mode, not inheritance** (#104): storage is
        signed, and a gene declared `MAGNITUDE` folds across zero when `Genetics.expressed` reads
        it, so a costed quantity arrives here non-negative however far its stored value has
        drifted. A gene declared `SIGNED` arrives as stored and may well be negative — which is
        why the constructor refuses to let one carry a cost, and why that refusal is what this
        guarantee rests on rather than any check in here (#136).
        """
        trait_upkeep = expressed_genes @ self.gene_cost
        thermoregulation = self.thermoregulation(
            expressed_genes[:, self.thermal_columns], temperature
        )
        return (self.config.basal_rate + trait_upkeep + thermoregulation).astype(np.float32)

    def thermoregulation(
        self, tolerances: np.ndarray, temperature: np.ndarray
    ) -> np.ndarray:
        """Energy units per tick: the thermoregulation term of `upkeep` alone.

        tolerances: (n, 2) expressed phenotype columns `thermal_columns`, heat then cold.
        temperature: (n,) for each entity where it stands, or (n, k) for k places it might stand.
            Each row is priced with that entity's own tolerances.

        Public because the thermal drive (#249) has to want exactly what this charges. A drive that
        recomputed the curve would be a second reading of one physical fact, free to drift from the
        bill: the shape of defect #112 was.
        """
        # Shaped to broadcast against `temperature`, whether it is one place per row or several.
        trailing = (1,) * (np.ndim(temperature) - 1)
        heat_tolerance = tolerances[:, 0].reshape(-1, *trailing)
        cold_tolerance = tolerances[:, 1].reshape(-1, *trailing)
        # At most one of these is non-zero for any animal, so each pays only its own side's bill
        # and carries the other side's tolerance as dead weight.
        too_hot = np.maximum(temperature - self.config.neutral_temperature, 0.0)
        too_cold = np.maximum(self.config.neutral_temperature - temperature, 0.0)
        return self.config.thermoregulation_rate * (
            too_hot**2 / (1.0 + heat_tolerance) + too_cold**2 / (1.0 + cold_tolerance)
        )
