"""Diet as an allocation, not a set of capacities (#102, CLAUDE.md §2.5).

Test-first (§8.1): the whole contract is a pure function of a phenotype block, so every property
worth having was writable before the implementation. The ones that matter are structural — that
efficiency lands in [0, 1] without a clamp, and that a generalist is strictly worse than a
specialist at the specialist's own food — because those are what make the encoding a trade-off
rather than a quantity that can run away.
"""

import numpy as np
import pytest

from core.ecology.diet import Diet, DietConfig
from core.genetics.registry import ExpressionMode, GeneRegistry, GeneSpec, Unit

from tests.support.genes import gene_registry

GENE_NAMES = ("size", "diet_animal_derived", "mutability")
REGISTRY = gene_registry(GENE_NAMES)


def config(**overrides):
    params = dict(animal_derived_gene="diet_animal_derived", frontier_exponent=2.0)
    params.update(overrides)
    return DietConfig(**params)


def phenotype(*allocations):
    """A phenotype block whose diet column holds `allocations`, already expressed on [0, 1]."""
    block = np.zeros((len(allocations), len(GENE_NAMES)), dtype=np.float32)
    block[:, GENE_NAMES.index("diet_animal_derived")] = allocations
    return block


class TestConfigValidation:
    @pytest.mark.parametrize("exponent", [1.0, 0.5, 0.0, -1.0])
    def test_rejects_a_frontier_that_does_not_punish_generalists(self, exponent):
        """At p = 1 the frontier is linear, so a generalist loses exactly what it gains and nothing
        selects against sitting in the middle (#146). Below 1 it is concave, which actively rewards
        being mediocre at everything. #116 draws this per world, so a bad draw must raise rather
        than quietly produce a world where diet does not matter."""
        with pytest.raises(ValueError, match="frontier_exponent"):
            Diet(REGISTRY, config(frontier_exponent=exponent))

    def test_rejects_a_gene_that_is_not_read_as_an_allocation(self):
        """A magnitude-read diet gene folds at zero, so a lineage allocated hard toward plants and
        one allocated hard toward flesh would express identically."""
        registry = GeneRegistry(
            (
                GeneSpec(
                    name="diet_animal_derived",
                    cost=0.0,
                    expression_mode=ExpressionMode.MAGNITUDE,
                    unit=Unit.DIMENSIONLESS,
                    description="miscast diet gene",
                ),
            )
        )
        with pytest.raises(ValueError, match="allocation"):
            Diet(registry, config())

    def test_rejects_a_gene_outside_the_vocabulary(self):
        with pytest.raises(KeyError, match="diet_missing"):
            Diet(REGISTRY, config(animal_derived_gene="diet_missing"))


class TestPlantEfficiency:
    def diet(self, **overrides):
        return Diet(REGISTRY, config(**overrides))

    def test_a_full_plant_allocation_converts_at_the_frontier_maximum(self):
        assert self.diet().plant_efficiency(phenotype(0.0))[0] == pytest.approx(1.0)

    def test_a_full_animal_allocation_cannot_use_plants_at_all(self):
        assert self.diet().plant_efficiency(phenotype(1.0))[0] == pytest.approx(0.0)

    def test_efficiency_falls_as_the_allocation_moves_toward_animals(self):
        efficiencies = self.diet().plant_efficiency(phenotype(0.0, 0.25, 0.5, 0.75, 1.0))

        assert efficiencies.tolist() == sorted(efficiencies.tolist(), reverse=True)

    def test_the_frontier_is_convex_so_a_generalist_is_worse_than_the_average_specialist(self):
        """The property that makes this a trade-off rather than a relabelling (#146). A linear
        frontier would make the even split worth exactly the mean of the two extremes; convexity
        is what puts it strictly below, and therefore what selects against being mediocre."""
        diet = self.diet()
        specialists = diet.plant_efficiency(phenotype(0.0, 1.0))
        generalist = diet.plant_efficiency(phenotype(0.5))[0]

        assert generalist < specialists.mean()

    def test_a_steeper_frontier_punishes_the_generalist_harder(self):
        even = phenotype(0.5)
        gentle = Diet(REGISTRY, config(frontier_exponent=1.5)).plant_efficiency(even)[0]
        steep = Diet(REGISTRY, config(frontier_exponent=4.0)).plant_efficiency(even)[0]

        assert steep < gentle

    def test_the_specialist_is_untouched_by_the_frontier_exponent(self):
        """Only the middle of the allocation moves with `p`; the ends are 0 and 1 whatever it is.
        That is what makes the exponent a statement about generalism rather than a global
        efficiency dial."""
        pure = phenotype(0.0)
        for exponent in (1.5, 2.0, 8.0):
            efficiency = Diet(REGISTRY, config(frontier_exponent=exponent)).plant_efficiency(pure)
            assert efficiency[0] == pytest.approx(1.0)

    def test_efficiency_stays_in_the_unit_interval_for_every_allocation(self):
        """A conversion above 1 would create energy, which §6 forbids outright. Held by the
        formula rather than by a clamp, since the allocation it reads is already bounded."""
        allocations = np.linspace(0.0, 1.0, 401)
        efficiencies = self.diet().plant_efficiency(phenotype(*allocations))

        assert np.all(efficiencies >= 0.0)
        assert np.all(efficiencies <= 1.0)

    def test_it_reads_one_column_and_ignores_the_rest_of_the_phenotype(self):
        block = phenotype(0.3, 0.3)
        block[1, GENE_NAMES.index("size")] = 99.0

        efficiencies = self.diet().plant_efficiency(block)

        assert efficiencies[0] == pytest.approx(efficiencies[1])

    def test_the_result_is_float32_and_one_entry_per_row(self):
        efficiencies = self.diet().plant_efficiency(phenotype(0.1, 0.6, 0.9))

        assert efficiencies.shape == (3,)
        assert efficiencies.dtype == np.float32
