"""Diet as a continuous allocation, never a category (CLAUDE.md §2.5, issue #102).

Herbivore, carnivore and omnivore look like categories, and §2.3's gene matrix cannot hold one: a
category would need a column per kind of thing, and speciation invents kinds at runtime. The same
rule that forced fear through a fixed-width cue space applies here — **nothing is a category;
everything is a position in a continuous space, and categories emerge as clusters in it.**

What that position *is* was settled in #102 after four options were weighed, and the reasoning is
worth keeping close to the code because three of the four look reasonable:

- Independent per-substrate efficiencies, each charging upkeep, is the shape #146 rejected. Read on
  [0, ∞) they need a saturating map to stay below 1, and that map lets *every* efficiency approach 1
  at once — so "you cannot be good at everything" becomes a claim about the cost table rather than a
  property of the encoding, and a world with low coefficients grows an omnivore superspecies.
- Normalised shares — softmax, or L1 over magnitudes — have a **null direction**: softmax is
  invariant to shifting every gene, L1 to scaling them. Drift along it changes no phenotype, so
  nothing selects on it, so it is an unbounded random walk that #104's per-gene clamp cannot see.
  Worse, it silently rescales what `mutability` means, because a saturated softmax moves less per
  unit of drift than one near the origin.
- A substrate space with an optimum and a breadth is the only option under which new food types
  cost no gene columns, and it stays available if substrates ever proliferate. It was not taken
  because there are three in view and they are stable (§8.3).

So: **stick-breaking.** Each split is one allocation gene read on [0, 1], and the shares it implies
sum to one by construction — no null direction, no clamp, and the trade-off is a fact about the
encoding rather than about any coefficient.

Today the tree has one branch:

    a = diet_animal_derived           plant = 1 − a,  animal-derived = a
    efficiency = share ** p           p > 1

`diet_fresh`, which splits the animal-derived half into flesh and carrion, is **not here**. It has
no reader until flesh can be eaten (#179) or something has died (#21), and the vocabulary widens per
issue as callers arrive rather than in one speculative batch (settled on #101, §8.2).

Adding it later is safe, and specifically so because of the tree: `plant` is `1 − a` whether or not
the animal-derived half is subdivided, so a later split cannot disturb a herbivore that already
exists. That is a property of stick-breaking rather than of allocations in general — under a flat
softmax over k substrates, widening k moves every existing share and would be a MAJOR fork (§2.8).

The convex exponent is what does the ecological work, exactly as #146 found for respiration. A
linear frontier takes back precisely what it gives, so nothing is punished for being a generalist
and drift parks every lineage in the middle; `p > 1` makes a generalist strictly worse than a
specialist *at that specialist's own food*, which is "jack of all trades, master of none" falling
out of the arithmetic rather than being written down.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.genetics.registry import ExpressionMode, GeneRegistry, Unit


@dataclass(frozen=True)
class DietConfig:
    """Per-world diet rules — never constants in `core/` (§2.1).

    animal_derived_gene: the allocation gene. 0 is a pure plant diet, 1 a pure animal one, and it
        must be declared `UNIT_INTERVAL`: read as a magnitude it would fold at zero, making a
        lineage allocated hard toward plants express identically to one allocated hard toward
        flesh.
    frontier_exponent: how sharply generalism is punished, the `p` in `share ** p`. Drawn per world
        under #116 rather than fixed, so one world has omnivores everywhere and another has two
        separate faunas — ecological variety from the same mechanism that already produces terrain
        variety.

    It lives beside #146's respiration exponent in the world config rather than being declared
    independently: they are two instances of one idea, and §2.1's warning about constants that must
    be tuned as a table applies to exactly that.
    """

    animal_derived_gene: str
    frontier_exponent: float

    def __post_init__(self) -> None:
        if self.frontier_exponent <= 1.0:
            raise ValueError(
                f"frontier_exponent must exceed 1, got {self.frontier_exponent}; at 1 the frontier "
                "is linear and a generalist loses exactly what it gains, so nothing selects "
                "against sitting in the middle, and below 1 being mediocre at everything is "
                "actively rewarded (#146)"
            )


class Diet:
    """One world's diet allocation, resolved against its gene vocabulary.

    Holds no state and owns no store column: it reads an already-expressed phenotype block and
    returns efficiencies. Genes are `Genetics`' to own (§2.3), and what a gene *means* for feeding
    is this module's.
    """

    def __init__(self, registry: GeneRegistry, config: DietConfig) -> None:
        self.config = config
        # Raises KeyError naming the vocabulary version if the gene does not exist. An allocation
        # is a bare fraction, hence dimensionless.
        self.animal_derived_index = registry.index_of(
            config.animal_derived_gene, unit=Unit.DIMENSIONLESS
        )
        mode = registry.spec(config.animal_derived_gene).expression_mode
        if mode is not ExpressionMode.UNIT_INTERVAL:
            raise ValueError(
                f"diet gene '{config.animal_derived_gene}' must be read as an allocation "
                f"(UNIT_INTERVAL), not {mode.value}; any other reading is a quantity, and a "
                "quantity can rise without giving anything up (#146)"
            )

    def plant_efficiency(self, phenotype: np.ndarray) -> np.ndarray:
        """(n,) float32, dimensionless in [0, 1]: what fraction of eaten plant biomass converts.

        `phenotype` is an expressed block, `(n, n_genes)`, as `Genetics.expressed` returns it.

        In [0, 1] because the allocation it reads is, and raised to a power above 1 — so this is
        structural rather than clamped, which is what §6's "energy is never created" needs. A
        conversion above 1 would mint energy out of grass; there is no branch here that could fail
        to prevent it.
        """
        plant_share = 1.0 - phenotype[:, self.animal_derived_index]
        return np.power(plant_share, self.config.frontier_exponent, dtype=np.float32)
