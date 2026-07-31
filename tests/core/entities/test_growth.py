"""Capacity growth at the tick boundary (#127).

Test-first (§8.1): the policy is a pure predicate over a store, and where it is allowed to run is a
structural claim — both were writable before the implementation.
"""

import numpy as np
import pytest

from core.entities.growth import GrowthConfig, grow_if_crowded
from core.entities.store import EntityStore


def store_with(occupied, capacity=16):
    store = EntityStore(initial_capacity=capacity, n_drives=1, n_genes=2)
    if occupied:
        store.allocate(occupied)
    return store


class TestConfigValidation:
    @pytest.mark.parametrize("fraction", [0.0, -0.1])
    def test_rejects_a_reserve_that_only_fires_when_already_full(self, fraction):
        with pytest.raises(ValueError, match="reserve_fraction"):
            GrowthConfig(reserve_fraction=fraction)


class TestTheReserve:
    def test_a_roomy_store_does_not_grow(self):
        store = store_with(2, capacity=16)

        assert not grow_if_crowded(store, GrowthConfig(reserve_fraction=0.1))
        assert store.capacity == 16

    def test_a_crowded_store_doubles(self):
        store = store_with(15, capacity=16)

        assert grow_if_crowded(store, GrowthConfig(reserve_fraction=0.1))
        assert store.capacity == 32

    def test_a_full_store_grows(self):
        store = store_with(16, capacity=16)

        assert grow_if_crowded(store, GrowthConfig(reserve_fraction=0.1))
        assert store.capacity == 32

    def test_the_reserve_is_measured_against_occupancy_not_capacity(self):
        """A mostly-empty store must not keep growing after a die-off. Measured against capacity,
        free rows would be plentiful in absolute terms and the ratio would still read low."""
        store = store_with(64, capacity=64)
        grow_if_crowded(store, GrowthConfig(reserve_fraction=0.5))
        assert store.capacity == 128

        # Almost everything dies; free rows are now enormous relative to who is left.
        store.release(store.row_ids()[store.alive][:60])

        assert not grow_if_crowded(store, GrowthConfig(reserve_fraction=0.5))
        assert store.capacity == 128

    def test_growth_preserves_what_was_there(self):
        store = store_with(0, capacity=4)
        ids = store.allocate(4, energy=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))

        grow_if_crowded(store, GrowthConfig(reserve_fraction=0.1))

        rows = [store._id_to_row[i] for i in ids.tolist()]
        assert store.energy[rows] == pytest.approx([1.0, 2.0, 3.0, 4.0])
        assert store.alive[rows].all()

    def test_growth_makes_room_that_allocate_can_use(self):
        """The whole point: a store that was full can hand out rows again on the next tick."""
        store = store_with(8, capacity=8)
        assert store.available == 0

        grow_if_crowded(store, GrowthConfig(reserve_fraction=0.1))

        assert store.allocate(4).shape == (4,)

    def test_an_empty_store_is_left_alone(self):
        """Occupancy is zero, so the reserve it implies is zero and nothing is short of anything."""
        store = store_with(0, capacity=8)

        assert not grow_if_crowded(store, GrowthConfig(reserve_fraction=0.5))
        assert store.capacity == 8

    def test_one_call_grows_once(self):
        """Doubling is amortised, not a loop: a store far below its reserve catches up over ticks
        rather than resizing repeatedly inside one boundary."""
        store = store_with(64, capacity=64)

        grow_if_crowded(store, GrowthConfig(reserve_fraction=4.0))

        assert store.capacity == 128
