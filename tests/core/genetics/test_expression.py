"""Expression modes: how a signed stored value becomes a phenotype (#104).

The contract is checkable in advance, so these were written before the implementation (§8.1). What
is *not* asserted here is which mode any particular gene should have — that is a per-world
declaration, and the point of the module is that nothing in `core/` decides it.
"""

import numpy as np
import pytest

from core.genetics.expression import ExpressionMode, ExpressionTable, GeneticsConfig
from core.genetics.vocabulary import GeneVocabulary


GENE_NAMES = ("size", "signature_0", "mutability")


def config(**overrides):
    params = dict(
        expression_modes={
            "size": ExpressionMode.MAGNITUDE,
            "signature_0": ExpressionMode.SIGNED,
            "mutability": ExpressionMode.MAGNITUDE,
        },
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


class TestEveryGeneMustDeclareAMode:
    """The same rule `MetabolismConfig` applies to costs, for a sharper reason: a gene with no
    declared reading would be taken as signed by default, and a signed `size` is a body with
    negative mass that also earns its own upkeep back (#136)."""

    def test_a_gene_with_no_declared_mode_is_rejected(self):
        modes = dict(config().expression_modes)
        del modes["size"]

        with pytest.raises(ValueError, match="size"):
            ExpressionTable(GeneVocabulary(GENE_NAMES), config(expression_modes=modes))

    def test_a_mode_declared_for_a_gene_outside_the_vocabulary_is_rejected(self):
        modes = dict(config().expression_modes)
        modes["gills"] = ExpressionMode.MAGNITUDE

        with pytest.raises(ValueError, match="gills"):
            ExpressionTable(GeneVocabulary(GENE_NAMES), config(expression_modes=modes))

    def test_a_mutability_gene_outside_the_vocabulary_is_rejected(self):
        with pytest.raises(KeyError, match="evolvability"):
            ExpressionTable(
                GeneVocabulary(GENE_NAMES), config(mutability_gene="evolvability")
            )

    def test_a_signed_mutability_gene_is_rejected(self):
        """It is the width of a draw, so what a negative value would mean is a negative scale."""
        modes = dict(config().expression_modes)
        modes["mutability"] = ExpressionMode.SIGNED

        with pytest.raises(ValueError, match="magnitude"):
            ExpressionTable(GeneVocabulary(GENE_NAMES), config(expression_modes=modes))


class TestPhenotype:
    def table(self):
        return ExpressionTable(GeneVocabulary(GENE_NAMES), config())

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


class TestResolvedColumns:
    def test_the_mutability_column_is_resolved_by_name(self):
        table = ExpressionTable(GeneVocabulary(GENE_NAMES), config())

        assert table.mutability_index == GENE_NAMES.index("mutability")

    def test_magnitude_columns_are_a_mask_in_vocabulary_order(self):
        table = ExpressionTable(GeneVocabulary(GENE_NAMES), config())

        assert table.magnitude_columns.tolist() == [True, False, True]
