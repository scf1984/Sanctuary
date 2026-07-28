import numpy as np
import pytest

from core.selection import Selection


class TestConstruction:
    def test_from_mask_roundtrips(self):
        mask = np.array([True, False, True], dtype=np.bool_)
        selection = Selection.from_mask(mask)
        assert selection.to_mask().tolist() == [True, False, True]

    def test_from_indices_sets_only_those_rows(self):
        selection = Selection.from_indices(np.array([1, 3]), capacity=5)
        assert selection.to_mask().tolist() == [False, True, False, True, False]

    def test_none_selects_nothing(self):
        assert len(Selection.none(4)) == 0

    def test_all_selects_everything(self):
        selection = Selection.all(4)
        assert len(selection) == 4
        assert selection.to_mask().all()

    def test_non_boolean_mask_rejected(self):
        with pytest.raises(ValueError):
            Selection.from_mask(np.array([1, 0, 1]))

    def test_non_1d_mask_rejected(self):
        with pytest.raises(ValueError):
            Selection(np.zeros((2, 2), dtype=np.bool_))

    def test_mask_is_copied_not_aliased(self):
        mask = np.array([True, False], dtype=np.bool_)
        selection = Selection.from_mask(mask)
        mask[0] = False
        assert selection.to_mask().tolist() == [True, False]

    def test_returned_mask_is_read_only(self):
        selection = Selection.from_mask(np.array([True, False], dtype=np.bool_))
        with pytest.raises(ValueError):
            selection.to_mask()[0] = False


class TestConversions:
    def test_to_indices_matches_mask(self):
        selection = Selection.from_mask(np.array([False, True, True, False], dtype=np.bool_))
        assert selection.to_indices().tolist() == [1, 2]

    def test_len_counts_selected_rows(self):
        selection = Selection.from_indices(np.array([0, 2, 4]), capacity=5)
        assert len(selection) == 3

    def test_capacity_matches_mask_length(self):
        assert Selection.none(7).capacity == 7


class TestComposition:
    def test_and_intersects(self):
        a = Selection.from_indices(np.array([0, 1, 2]), capacity=4)
        b = Selection.from_indices(np.array([1, 2, 3]), capacity=4)
        assert (a & b).to_indices().tolist() == [1, 2]

    def test_or_unions(self):
        a = Selection.from_indices(np.array([0]), capacity=4)
        b = Selection.from_indices(np.array([3]), capacity=4)
        assert (a | b).to_indices().tolist() == [0, 3]

    def test_invert_complements(self):
        selection = Selection.from_indices(np.array([1]), capacity=3)
        assert (~selection).to_indices().tolist() == [0, 2]

    def test_mismatched_capacity_raises_on_and(self):
        a = Selection.none(3)
        b = Selection.none(4)
        with pytest.raises(ValueError):
            a & b

    def test_mismatched_capacity_raises_on_or(self):
        a = Selection.none(3)
        b = Selection.none(4)
        with pytest.raises(ValueError):
            a | b

    def test_composition_does_not_mutate_operands(self):
        a = Selection.from_indices(np.array([0, 1]), capacity=3)
        b = Selection.from_indices(np.array([1, 2]), capacity=3)
        a & b
        assert a.to_indices().tolist() == [0, 1]
        assert b.to_indices().tolist() == [1, 2]


class TestEquality:
    def test_equal_masks_are_equal(self):
        a = Selection.from_indices(np.array([0, 2]), capacity=3)
        b = Selection.from_mask(np.array([True, False, True]))
        assert a == b

    def test_different_masks_are_not_equal(self):
        a = Selection.from_indices(np.array([0]), capacity=3)
        b = Selection.from_indices(np.array([1]), capacity=3)
        assert a != b
