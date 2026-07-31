"""Expression modes: how a signed stored value becomes a phenotype (#104).

The contract is checkable in advance, so these were written before the implementation (§8.1). What
is *not* asserted here is which mode any particular gene should have — that is a per-world
declaration, and the point of the module is that nothing in `core/` decides it.
"""

import numpy as np
import pytest

from core.genetics.expression import ExpressionTable, GeneticsConfig
from core.genetics.registry import ExpressionMode, GeneRegistry, GeneSpec, Unit

from tests.support.genes import gene_registry

GENE_NAMES = ("size", "signature_0", "mutability")
GENE_REGISTRY = gene_registry(GENE_NAMES)


def config(**overrides):
    params = dict(
        mutability_gene="mutability",
        drift_margin=2.0,
    )
    params.update(overrides)
    return GeneticsConfig(**params)


class TestConfigValidation:
    def test_rejects_a_non_positive_drift_margin(self):
        with pytest.raises(ValueError, match="drift_margin"):
            config(drift_margin=0.0)

    def test_rejects_a_negative_drift_margin(self):
        with pytest.raises(ValueError, match="drift_margin"):
            config(drift_margin=-1.0)


class TestTheMutabilityGeneMustBeUsable:
    """A gene can no longer *omit* a mode — `GeneSpec` carries one, so the two completeness checks
    that used to live here are gone rather than moved (#111). What remains is what the registry
    cannot know: which gene this world nominates as its spread floor, and whether that choice is
    coherent."""

    def test_a_mutability_gene_outside_the_vocabulary_is_rejected(self):
        with pytest.raises(KeyError, match="evolvability"):
            ExpressionTable(GENE_REGISTRY, config(mutability_gene="evolvability"))

    def test_a_signed_mutability_gene_is_rejected(self):
        """It is the width of a draw, so what a negative value would mean is a negative scale."""
        signed_mutability = GeneRegistry(
            (
                GeneSpec("size", 0.0, ExpressionMode.MAGNITUDE, Unit.DIMENSIONLESS, "body scale"),
                GeneSpec(
                    "mutability", 0.0, ExpressionMode.SIGNED, Unit.DIMENSIONLESS, "spread floor"
                ),
            )
        )

        with pytest.raises(ValueError, match="magnitude"):
            ExpressionTable(signed_mutability, config())

    def test_a_mutability_gene_declared_in_the_wrong_unit_is_rejected(self):
        """The spread of a distribution is a bare number; a length there is a different quantity."""
        length_mutability = GeneRegistry(
            (
                GeneSpec("size", 0.0, ExpressionMode.MAGNITUDE, Unit.DIMENSIONLESS, "body scale"),
                GeneSpec("mutability", 0.0, ExpressionMode.MAGNITUDE, Unit.LENGTH, "spread floor"),
            )
        )

        with pytest.raises(ValueError, match="declared in length"):
            ExpressionTable(length_mutability, config())


class TestPhenotype:
    def table(self):
        return ExpressionTable(GENE_REGISTRY, config())

    def test_a_magnitude_gene_folds_across_zero(self):
        raw = np.array([[-2.0, 0.0, 0.0]], dtype=np.float32)

        assert self.table().phenotype(raw)[0, 0] == pytest.approx(2.0)

    def test_a_signed_gene_keeps_its_sign(self):
        raw = np.array([[0.0, -0.75, 0.0]], dtype=np.float32)

        assert self.table().phenotype(raw)[0, 1] == pytest.approx(-0.75)

    def test_modes_are_applied_per_column_across_a_whole_block(self):
        raw = np.array(
            [[-1.0, -1.0, -0.5], [2.0, 2.0, 0.25]],
            dtype=np.float32,
        )

        phenotype = self.table().phenotype(raw)

        assert phenotype.tolist() == [[1.0, -1.0, 0.5], [2.0, 2.0, 0.25]]

    def test_the_result_is_float32(self):
        raw = np.zeros((3, len(GENE_NAMES)), dtype=np.float32)

        assert self.table().phenotype(raw).dtype == np.float32

    def test_the_stored_row_is_not_modified(self):
        raw = np.array([[-2.0, -2.0, -2.0]], dtype=np.float32)
        original = raw.copy()

        self.table().phenotype(raw)

        np.testing.assert_array_equal(raw, original)


class TestUnitIntervalPhenotype:
    """An allocation gene: read on [0, 1] so that what it says is how a fixed budget *splits*
    (#102, #146). The reading has to hold that range on its own, because genes drift freely and a
    clamp is only a numerical backstop (§2.5)."""

    GENES = ("size", "diet_animal_derived", "mutability")

    def table(self):
        return ExpressionTable(gene_registry(self.GENES), config())

    def raw(self, value):
        return np.array([[0.0, value, 0.0]], dtype=np.float32)

    def test_a_zero_gene_expresses_an_even_split(self):
        assert self.table().phenotype(self.raw(0.0))[0, 1] == pytest.approx(0.5)

    def test_the_reading_is_monotone(self):
        table = self.table()
        values = [table.phenotype(self.raw(v))[0, 1] for v in (-4.0, -1.0, 0.0, 1.0, 4.0)]

        assert values == sorted(values)

    def test_the_bounds_hold_without_a_clamp(self):
        """The property that matters: the reading is inside [0, 1] for every stored value, held by
        the formula rather than by a clamp (§2.5 — genes drift freely, so the formula has to carry
        it)."""
        table = self.table()
        for value in (-1.0e30, -80.0, -3.0, 0.0, 3.0, 80.0, 1.0e30):
            expressed = table.phenotype(self.raw(value))[0, 1]
            assert 0.0 <= expressed <= 1.0

    def test_the_bounds_are_open_across_the_range_a_lineage_actually_drifts(self):
        for value in (-12.0, -4.0, 4.0, 12.0):
            assert 0.0 < self.table().phenotype(self.raw(value))[0, 1] < 1.0

    def test_far_drift_saturates_to_a_pure_specialist_rather_than_overshooting(self):
        """Beyond about ±40 the float32 reading reaches exactly 0 or 1, and that is deliberately
        left alone. A fully specialised gut is a legitimate evolutionary endpoint — a pure
        herbivore that cannot digest flesh — unlike senescence decay reaching zero, which would be
        literal immortality. What must never happen is a value *outside* the interval, which the
        test above pins."""
        table = self.table()

        assert table.phenotype(self.raw(-80.0))[0, 1] == 0.0
        assert table.phenotype(self.raw(80.0))[0, 1] == 1.0

    def test_an_extreme_gene_neither_overflows_nor_warns(self):
        """`1 / (1 + exp(-x))` overflows on a large negative gene and raises a RuntimeWarning every
        tick; the tanh form is the same function computed stably."""
        table = self.table()
        with np.errstate(over="raise", invalid="raise"):
            extremes = table.phenotype(
                np.array([[0.0, -1.0e30, 0.0], [0.0, 1.0e30, 0.0]], dtype=np.float32)
            )

        assert np.all(np.isfinite(extremes))
        assert extremes[0, 1] == pytest.approx(0.0)
        assert extremes[1, 1] == pytest.approx(1.0)

    def test_it_is_not_folded_like_a_magnitude(self):
        """Opposite genes are opposite allocations, not the same one. `abs` would make a lineage
        allocated hard toward plants indistinguishable from one allocated hard toward flesh."""
        table = self.table()

        assert table.phenotype(self.raw(-2.0))[0, 1] != pytest.approx(
            table.phenotype(self.raw(2.0))[0, 1]
        )

    def test_complementary_genes_split_a_whole(self):
        table = self.table()
        low = table.phenotype(self.raw(-1.5))[0, 1]
        high = table.phenotype(self.raw(1.5))[0, 1]

        assert low + high == pytest.approx(1.0)


class TestResolvedColumns:
    def test_the_mutability_column_is_resolved_by_name(self):
        table = ExpressionTable(GENE_REGISTRY, config())

        assert table.mutability_index == GENE_NAMES.index("mutability")

    def test_magnitude_columns_are_a_mask_in_vocabulary_order(self):
        table = ExpressionTable(GENE_REGISTRY, config())

        assert table.magnitude_columns.tolist() == [True, False, True]

    def test_unit_interval_columns_are_a_mask_in_vocabulary_order(self):
        table = ExpressionTable(gene_registry(("size", "diet_fresh", "mutability")), config())

        assert table.unit_interval_columns.tolist() == [False, True, False]
