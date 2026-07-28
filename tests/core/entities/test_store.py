import time

import numpy as np
import pytest

from core.entities.store import EntityStore, EntityStoreFull, UnknownEntityError


def make_store(initial_capacity=8, n_drives=3, n_genes=4):
    return EntityStore(initial_capacity=initial_capacity, n_drives=n_drives, n_genes=n_genes)


class TestConstruction:
    def test_rejects_non_positive_capacity(self):
        with pytest.raises(ValueError):
            make_store(initial_capacity=0)

    def test_rejects_non_positive_drive_count(self):
        with pytest.raises(ValueError):
            make_store(n_drives=0)

    def test_rejects_non_positive_gene_count(self):
        with pytest.raises(ValueError):
            make_store(n_genes=0)

    def test_columns_start_at_requested_shape(self):
        store = make_store(initial_capacity=8, n_drives=3, n_genes=5)
        assert store.capacity == 8
        assert store.available == 8
        assert store.drive_scores.shape == (8, 3)
        assert store.genes.shape == (8, 5)


class TestAllocate:
    def test_returns_unique_sequential_ids(self):
        store = make_store()
        ids = store.allocate(3)
        assert len(set(ids.tolist())) == 3
        assert ids.tolist() == sorted(ids.tolist())

    def test_new_rows_default_and_are_marked_alive(self):
        store = make_store()
        ids = store.allocate(2)
        rows = [store._id_to_row[i] for i in ids.tolist()]
        assert all(store.alive[rows])
        assert (store.age[rows] == 0).all()
        assert (store.energy[rows] == 0.0).all()
        assert (store.species_id[rows] == -1).all()
        assert (store.drive_scores[rows] == 0.0).all()
        assert (store.genes[rows] == 0.0).all()

    def test_seeds_initial_values_in_one_call(self):
        store = make_store(n_genes=2)
        ids = store.allocate(
            2,
            x=np.array([1.0, 2.0], dtype=np.float32),
            energy=np.array([50.0, 75.0], dtype=np.float32),
            species_id=np.array([4, 4], dtype=np.int32),
            genes=np.array([[0.25, 0.5], [0.75, 1.5]], dtype=np.float32),
        )
        rows = [store._id_to_row[i] for i in ids.tolist()]
        assert store.x[rows].tolist() == [1.0, 2.0]
        assert store.energy[rows].tolist() == [50.0, 75.0]
        assert store.species_id[rows].tolist() == [4, 4]
        assert store.genes[rows].tolist() == [[0.25, 0.5], [0.75, 1.5]]

    def test_reduces_available_capacity(self):
        store = make_store(initial_capacity=8)
        store.allocate(3)
        assert store.available == 5

    def test_wrong_length_initial_value_raises_without_mutating(self):
        store = make_store()
        with pytest.raises(ValueError):
            store.allocate(2, x=np.array([1.0], dtype=np.float32))
        assert store.available == store.capacity

    def test_unknown_column_raises_without_mutating(self):
        store = make_store()
        with pytest.raises(ValueError):
            store.allocate(1, not_a_column=np.array([1.0]))
        assert store.available == store.capacity

    def test_alive_is_not_seedable(self):
        store = make_store()
        with pytest.raises(ValueError):
            store.allocate(1, alive=np.array([False]))

    def test_raises_when_capacity_exhausted(self):
        store = make_store(initial_capacity=4)
        with pytest.raises(EntityStoreFull):
            store.allocate(5)
        # A failed allocate() must not partially consume the free list.
        assert store.available == 4

    def test_exact_remaining_capacity_succeeds(self):
        store = make_store(initial_capacity=4)
        ids = store.allocate(4)
        assert len(ids) == 4
        assert store.available == 0


class TestReleaseAndReuse:
    def test_release_marks_rows_dead(self):
        store = make_store()
        ids = store.allocate(2)
        rows = [store._id_to_row[i] for i in ids.tolist()]
        store.release(ids)
        assert not store.alive[rows].any()

    def test_released_rows_are_reused_not_grown(self):
        store = make_store(initial_capacity=4)
        ids = store.allocate(4)
        store.release(ids[:2])
        assert store.capacity == 4
        more = store.allocate(2)
        assert store.capacity == 4
        assert store.available == 0
        assert len(more) == 2

    def test_churn_never_grows_capacity(self):
        store = make_store(initial_capacity=4)
        for _ in range(50):
            ids = store.allocate(4)
            store.release(ids)
        assert store.capacity == 4

    def test_ids_are_never_reused(self):
        store = make_store(initial_capacity=4)
        first = store.allocate(4)
        store.release(first)
        second = store.allocate(4)
        assert set(first.tolist()).isdisjoint(set(second.tolist()))

    def test_release_unknown_id_raises(self):
        store = make_store()
        with pytest.raises(UnknownEntityError):
            store.release(np.array([999], dtype=np.int64))

    def test_double_release_raises(self):
        store = make_store()
        ids = store.allocate(1)
        store.release(ids)
        with pytest.raises(UnknownEntityError):
            store.release(ids)

    def test_release_unknown_id_raises_without_freeing_the_valid_ones_processed_first(self):
        store = make_store(initial_capacity=4)
        ids = store.allocate(2)
        bad_batch = np.array([ids[0], 12345], dtype=np.int64)
        with pytest.raises(UnknownEntityError):
            store.release(bad_batch)
        # release() validates every id before freeing any row.
        assert store.available == 2


class TestFreeRowMask:
    def test_all_rows_free_on_a_fresh_store(self):
        store = make_store(initial_capacity=4)
        assert store.free_row_mask().tolist() == [True, True, True, True]

    def test_allocated_rows_are_not_free(self):
        store = make_store(initial_capacity=4)
        ids = store.allocate(2)
        rows = {store._id_to_row[i] for i in ids.tolist()}
        mask = store.free_row_mask()
        assert not any(mask[row] for row in rows)
        assert mask.sum() == 2

    def test_released_rows_return_to_the_mask(self):
        store = make_store(initial_capacity=4)
        ids = store.allocate(4)
        store.release(ids[:1])
        assert store.free_row_mask().sum() == 1

    def test_returned_mask_is_a_snapshot_not_a_live_view(self):
        store = make_store(initial_capacity=4)
        mask = store.free_row_mask()
        store.allocate(4)
        assert mask.tolist() == [True, True, True, True]


class TestGrowth:
    def test_doubles_capacity(self):
        store = make_store(initial_capacity=8)
        store.grow()
        assert store.capacity == 16

    def test_new_rows_join_the_free_list(self):
        store = make_store(initial_capacity=4)
        store.allocate(4)
        assert store.available == 0
        store.grow()
        assert store.capacity == 8
        assert store.available == 4

    def test_preserves_existing_entity_data(self):
        store = make_store(initial_capacity=4, n_drives=2)
        ids = store.allocate(
            2,
            x=np.array([10.0, 20.0], dtype=np.float32),
            energy=np.array([5.0, 6.0], dtype=np.float32),
        )
        store.grow()
        rows = [store._id_to_row[i] for i in ids.tolist()]
        assert store.x[rows].tolist() == [10.0, 20.0]
        assert store.energy[rows].tolist() == [5.0, 6.0]
        assert store.alive[rows].all()

    def test_ids_allocated_before_growth_still_release_correctly(self):
        store = make_store(initial_capacity=4)
        ids = store.allocate(2)
        store.grow()
        store.release(ids)
        assert store.available == store.capacity

    def test_rows_free_before_growth_remain_usable_after(self):
        store = make_store(initial_capacity=4)
        ids = store.allocate(4)
        store.release(ids[:1])
        store.grow()
        new_ids = store.allocate(1, energy=np.array([42.0], dtype=np.float32))
        row = store._id_to_row[new_ids[0].item()]
        assert store.energy[row] == 42.0

    def test_no_live_view_survives_a_resize(self):
        store = make_store(initial_capacity=4)
        store.allocate(4, energy=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        pre_growth_energy_view = store.energy

        store.grow()

        # grow() replaces the column with a fresh array rather than resizing in place, so the
        # array a caller captured before growth is a distinct object from the live column...
        assert store.energy is not pre_growth_energy_view
        # ...and mutating the new live column leaves the old view's values untouched, proving
        # the resize did not alias or corrupt whatever held the old reference.
        store.energy[:] = -1.0
        assert pre_growth_energy_view.tolist() == [1.0, 2.0, 3.0, 4.0]


class TestGrowthUnderLoad:
    def test_growth_at_spike_scale_stays_within_budget(self):
        # docs/spikes/soa-throughput.md (issue #1) measured a single-column doubling copy at
        # 100,000 rows at 2.53ms. This store's grow() copies nine columns (one 2D), so a
        # generously wide multiple of that -- not the raw spike number -- is the right budget:
        # this guards against an accidental O(n^2) regression, not against normal constant-factor
        # variation.
        store = make_store(initial_capacity=100_000, n_drives=5)
        store.allocate(100_000)

        start = time.perf_counter()
        store.grow()
        elapsed = time.perf_counter() - start

        assert elapsed < 0.2
        assert store.capacity == 200_000
