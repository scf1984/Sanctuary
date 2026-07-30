import numpy as np
import pytest

from core.genetics.expression import ExpressionMode
from core.genetics.registry import GeneRegistry, GeneSpec, Unit


def spec(name, cost=0.0, mode=ExpressionMode.MAGNITUDE, unit=Unit.DIMENSIONLESS):
    return GeneSpec(
        name=name,
        cost=cost,
        expression_mode=mode,
        unit=unit,
        description=f"test gene {name}",
    )


SPECS = (
    spec("size", cost=0.02),
    spec("speed", cost=0.03, unit=Unit.LENGTH),
    spec("insulation", cost=0.01),
    spec("signature_0", mode=ExpressionMode.SIGNED),
    spec("mutability"),
)


class TestGeneSpec:
    def test_rejects_a_negative_cost(self):
        # A negative cost is energy created out of a trait, which §2.5's closed loop forbids.
        with pytest.raises(ValueError, match="non-negative"):
            spec("size", cost=-0.1)

    def test_rejects_a_blank_description(self):
        # The generated map is the whole reason description is a field; an empty one makes the
        # registry's readable half worthless while still passing every other check.
        with pytest.raises(ValueError, match="description"):
            GeneSpec(
                name="size",
                cost=0.0,
                expression_mode=ExpressionMode.MAGNITUDE,
                unit=Unit.DIMENSIONLESS,
                description="   ",
            )

    def test_a_zero_cost_is_legal(self):
        assert spec("mutability", cost=0.0).cost == 0.0


class TestConstruction:
    def test_rejects_an_empty_registry(self):
        with pytest.raises(ValueError, match="at least one gene"):
            GeneRegistry(())

    def test_rejects_duplicate_gene_names(self):
        with pytest.raises(ValueError, match="declared twice"):
            GeneRegistry((spec("size"), spec("speed"), spec("size")))

    def test_vocabulary_preserves_declaration_order(self):
        registry = GeneRegistry(SPECS)
        assert registry.vocabulary.names == (
            "size",
            "speed",
            "insulation",
            "signature_0",
            "mutability",
        )

    def test_cost_is_a_column_ordered_array(self):
        registry = GeneRegistry(SPECS)
        np.testing.assert_allclose(registry.cost, [0.02, 0.03, 0.01, 0.0, 0.0], rtol=1e-6)
        assert registry.cost.dtype == np.float32

    def test_magnitude_columns_marks_only_magnitude_genes(self):
        registry = GeneRegistry(SPECS)
        np.testing.assert_array_equal(
            registry.magnitude_columns, [True, True, True, False, True]
        )


class TestCostsAndModesCannotDisagree:
    def test_rejects_a_cost_on_a_gene_not_read_as_a_magnitude(self):
        # #136: a signed phenotype times a positive cost is a negative term in upkeep wherever the
        # value is negative, so the bill is discounted rather than charged. Folding the two tables
        # together is what makes this checkable at one place.
        with pytest.raises(ValueError, match="magnitude"):
            GeneRegistry((spec("size"), spec("aversion0_0", cost=0.5, mode=ExpressionMode.SIGNED)))

    def test_allows_a_zero_cost_on_a_signed_gene(self):
        # A gene charging nothing cannot contribute a term of any sign, so the whole cue block is
        # legal exactly as §2.5's reserved table declares it.
        registry = GeneRegistry((spec("size"), spec("signature_0", mode=ExpressionMode.SIGNED)))
        assert registry.cost[1] == 0.0


class TestResolvingAGene:
    def test_index_of_returns_the_column(self):
        assert GeneRegistry(SPECS).index_of("insulation") == 2

    def test_index_of_raises_for_an_unknown_gene(self):
        with pytest.raises(KeyError, match="not in gene vocabulary"):
            GeneRegistry(SPECS).index_of("wingspan")

    def test_index_of_accepts_a_matching_expected_unit(self):
        assert GeneRegistry(SPECS).index_of("speed", unit=Unit.LENGTH) == 1

    def test_index_of_rejects_a_gene_whose_unit_is_wrong_for_the_caller(self):
        # The check that would have caught the elevation-units defect (#112): a config naming a
        # gene it will read as a length must get a length, and both sides being floats is exactly
        # why nothing else can catch it.
        with pytest.raises(ValueError, match="declared in dimensionless.*read as length"):
            GeneRegistry(SPECS).index_of("insulation", unit=Unit.LENGTH)

    def test_spec_returns_the_declaration(self):
        assert GeneRegistry(SPECS).spec("speed").unit is Unit.LENGTH


class TestGeneratedMap:
    def test_describe_names_every_gene_with_its_declared_facts(self):
        described = GeneRegistry(SPECS).describe()
        for name in ("size", "speed", "insulation", "signature_0", "mutability"):
            assert name in described
        assert "magnitude" in described
        assert "signed" in described
        assert "length" in described

    def test_describe_reports_costs(self):
        assert "0.03" in GeneRegistry(SPECS).describe()
