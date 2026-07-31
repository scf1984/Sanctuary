"""One table holding everything the simulation knows about a gene (CLAUDE.md §2.5, issue #111).

Gene facts used to live in four places that could disagree: §2.5's prose table in CLAUDE.md, the
expression modes in `core.genetics.expression`, the costs in `core.ecology.metabolism`, and a
scatter of `*_gene` config fields naming genes by string. Only the costs were enforced, so a prose
table drifted from the code within two issues and nothing said so.

**Absence is unrepresentable rather than validated.** A gene *is* a `GeneSpec`, so it cannot exist
without a cost or without an expression mode — the two completeness checks that used to run at
construction are gone, not moved, because there is no longer a way to omit either. That is §8.7's
preference for a shape that cannot be wrong over a check that notices afterwards.

The other half is that the facts are **consulted, not merely declared** (§4). Costs resolve into
`cost` and modes into `magnitude_columns`, both column-ordered, and `index_of` takes the unit a
caller expects — so a config naming a gene it will read as a length gets a length or an error.
Without that last part `unit` would be documentation, which is exactly the decorative abstraction
this issue exists to remove.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from core.genetics.vocabulary import GeneVocabulary


class ExpressionMode(Enum):
    """How one gene's stored value is read as a phenotype (#104).

    Declared here rather than in `core.genetics.expression` because it is a *declaration* about a
    gene, like `Unit` and like cost, and `GeneSpec` needs all three in one object. That module
    applies the modes; this one records them.
    """

    MAGNITUDE = "magnitude"
    """A quantity: read as `abs(value)`. Negative storage is meaningless for a body, so drift across
    zero folds back rather than producing a negative size."""

    SIGNED = "signed"
    """A direction in cue space: read as stored. Sign is information, not a mistake."""

    EXPONENTIAL = "exponential"
    """A rate or a scale: read as `exp(value)`. Strictly positive on the whole real line, so the
    phenotype can never reach zero however far the gene drifts, and equal steps in storage are
    equal *ratios* in the phenotype — which is what a temperature or a rate wants, since halving
    and doubling are the symmetric pair rather than plus and minus one.

    Note this is not interchangeable with MAGNITUDE. `abs` folds at zero, so a gene drifting across
    it produces a phenotype that bounces off zero; `exp` is monotone, so drift is smooth and the
    sign of the stored value carries real information (below one versus above it)."""

    UNIT_INTERVAL = "unit_interval"
    """An allocation: read as a logistic squash into (0, 1). What the gene says is not how much of
    something a body has but how a fixed budget is *split* — 0 is all of one side, 1 all of the
    other, and the two shares sum to one by construction.

    That is why an allocation needs no clamp and no ceiling: there is no quantity to run away
    (#146). A capacity can always rise, so `water_breathing` or `feeding_speed` would be pure
    benefit bounded only by whatever the cost table happened to say; an allocation cannot, because
    every unit given to one side is taken from the other.

    The interval is held by the reading itself rather than by a clamp — the same requirement §2.5
    places on senescence decay, and for the same reason: genes drift freely, so the formula has to
    carry the property.

    Unlike senescence decay, though, the *ends* are allowed. Beyond roughly ±40 the float32
    reading saturates to exactly 0 or 1, and that is left alone deliberately: a fully specialised
    gut is a legitimate evolutionary endpoint, and the convex frontier in #102 exists precisely to
    reward getting there. Senescence could not reach zero because that would be immortality; a
    pure herbivore is just a herbivore. What must never happen is a value *outside* the interval.

    First declared by #102's diet allocation; named by #104 and reserved for #99's
    `sex_allocation` and `selfing_rate` and #146's respiration long before anything used it."""

    @property
    def always_non_negative(self) -> bool:
        """Whether this reading can ever produce a negative phenotype.

        Asked rather than hardcoded as a member list, because it is the *property* #136 needs and
        not the identity of any one mode: a cost charges correctly only when the value it
        multiplies cannot go below zero. `abs` and `exp` both promise that; a raw signed read does
        not. A mode added later answers here once instead of being remembered in the registry.
        """
        return self in (
            ExpressionMode.MAGNITUDE,
            ExpressionMode.EXPONENTIAL,
            ExpressionMode.UNIT_INTERVAL,
        )


class Unit(Enum):
    """What an expressed gene value *is*, so a caller can say what it expects to be handed.

    Only the dimensions genes actually use today are members, following `ExpressionMode`'s
    precedent of omitting a reading nothing declares (§8.2). Adding one is additive: #147's oxygen
    reserve capacity would bring an energy dimension with it.

    Deliberately dimension names rather than physical units — §2.6 keeps every quantity in the
    world's own denomination, and `tests/test_physical_units.py` fails on the physical spelling.
    """

    DIMENSIONLESS = "dimensionless"
    """A bare multiplier or a coordinate: size, insulation, acuity, cue signature, mutability."""

    LENGTH = "length"
    """World units — the denomination x, y, elevation and cell size share (#112). `speed` is a
    length because the tick is unitless, so world-units-per-tick is a length."""


@dataclass(frozen=True)
class GeneSpec:
    """One gene's complete declaration: what it costs, how it is read, and what it means.

    name: the vocabulary key. Its position in the registry is its column in the gene matrix.
    cost: energy units per tick per unit of the gene's *expressed* value. Zero is legal — §2.5
        names several genes whose counterweight is a selective consequence rather than upkeep —
        but only a gene read as a magnitude may charge anything (#136), enforced by the registry
        because that is the one place both facts are in hand.
    expression_mode: how a stored value becomes a phenotype (`core.genetics.expression`).
    unit: what the expressed value is, checked against what each caller expects.
    description: one line, and the reason the generated map is worth reading.
    """

    name: str
    cost: float
    expression_mode: ExpressionMode
    unit: Unit
    description: str

    def __post_init__(self) -> None:
        if self.cost < 0:
            # Negative upkeep is energy minted from a trait, which §2.5's closed loop forbids
            # outright — sunlight is the only income (#18).
            raise ValueError(f"gene '{self.name}' cost must be non-negative, got {self.cost}")
        if not self.description.strip():
            raise ValueError(f"gene '{self.name}' needs a description; the generated map is why")


class GeneRegistry:
    """A world's genes, in column order, with their costs and modes resolved for vectorized use.

    vocabulary: the `GeneVocabulary` these specs imply, in declaration order. Built here rather
        than supplied, so a vocabulary whose genes lack declarations cannot be constructed.
    cost: (n_genes,) float32, energy units per tick per unit of expressed value, in column order —
        so trait upkeep for any number of entities is one matrix-vector product (§2.3).
    magnitude_columns: (n_genes,) bool, True where the gene is read as a magnitude, in column
        order — so applying every mode to a phenotype block is a handful of whole-array operations.
    exponential_columns: (n_genes,) bool, True where the gene is read as `exp(value)`.
    unit_interval_columns: (n_genes,) bool, True where the gene is read as an allocation on (0, 1).
    """

    def __init__(self, specs: tuple[GeneSpec, ...]) -> None:
        if not specs:
            raise ValueError("a gene registry must declare at least one gene")

        seen: set[str] = set()
        for gene in specs:
            if gene.name in seen:
                raise ValueError(f"gene '{gene.name}' declared twice")
            seen.add(gene.name)

        # Checked here because this is the first place a cost and a mode are in the same object.
        # A signed phenotype times a positive cost is a negative term in upkeep wherever the value
        # is negative, so the bill is discounted instead of charged (#136).
        miscosted = [
            gene
            for gene in specs
            if gene.cost != 0 and not gene.expression_mode.always_non_negative
        ]
        if miscosted:
            offenders = ", ".join(
                f"{gene.name} ({gene.expression_mode.value})" for gene in miscosted
            )
            raise ValueError(
                f"only a gene whose expression mode cannot produce a negative phenotype may carry "
                f"a cost; costed but signed: {offenders}. A signed phenotype times a positive cost "
                "discounts upkeep instead of charging it (#136)"
            )

        self.specs = specs
        self._by_name = {gene.name: gene for gene in specs}
        self.vocabulary = GeneVocabulary(tuple(gene.name for gene in specs))
        self.cost = np.array([gene.cost for gene in specs], dtype=np.float32)
        self.magnitude_columns = np.array(
            [gene.expression_mode is ExpressionMode.MAGNITUDE for gene in specs], dtype=bool
        )
        self.exponential_columns = np.array(
            [gene.expression_mode is ExpressionMode.EXPONENTIAL for gene in specs], dtype=bool
        )
        self.unit_interval_columns = np.array(
            [gene.expression_mode is ExpressionMode.UNIT_INTERVAL for gene in specs], dtype=bool
        )

    def __len__(self) -> int:
        return len(self.specs)

    def spec(self, name: str) -> GeneSpec:
        """The full declaration for `name`, raising if the vocabulary has no such gene."""
        try:
            return self._by_name[name]
        except KeyError:
            raise KeyError(
                f"'{name}' is not in gene vocabulary v{self.vocabulary.version}"
            ) from None

    def index_of(self, name: str, unit: Unit | None = None) -> int:
        """The column index of `name`, optionally asserting the unit the caller will read it as.

        Passing `unit` is what makes the declaration load-bearing: both sides of a units mismatch
        are floats, so nothing downstream can notice one (§2.6, #112). A caller that genuinely does
        not care about the dimension — inheritance, distance — omits it.
        """
        gene = self.spec(name)
        if unit is not None and gene.unit is not unit:
            raise ValueError(
                f"gene '{name}' is declared in {gene.unit.value} but is being read as "
                f"{unit.value}; a mismatch here cannot fail on its own because both sides are "
                "floats (#112)"
            )
        return self.vocabulary.index_of(name)

    def describe(self) -> str:
        """The readable gene map, generated so it cannot drift from what the code consults.

        This is what replaces §2.5's hand-maintained prose table: a document nothing reads is the
        prototype's `FoodChain` in a new costume (§4), and a document *generated* from the table
        the simulation actually uses cannot disagree with it.
        """
        rows = [
            f"| {gene.name} | {gene.cost:g} | {gene.expression_mode.value} | "
            f"{gene.unit.value} | {gene.description} |"
            for gene in self.specs
        ]
        return "\n".join(
            [
                f"# Gene vocabulary v{self.vocabulary.version}",
                "",
                "| gene | cost | expression | unit | meaning |",
                "|---|---|---|---|---|",
                *rows,
            ]
        )
