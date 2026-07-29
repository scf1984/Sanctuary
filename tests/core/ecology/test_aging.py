import numpy as np
import pytest

from core.ecology.aging import Aging
from core.entities.store import EntityStore
from core.selection import Selection
from core.services import ColumnOwnershipError, ColumnRegistry
from core.world.tick import TickLoop


def make_world(initial_capacity=8):
    store = EntityStore(initial_capacity=initial_capacity, n_drives=1, n_genes=1)
    registry = ColumnRegistry()
    return store, registry, Aging(store, registry)


def aging_loop(store, aging):
    """A tick loop running aging alone, over whoever is alive when the tick reaches it.

    The `alive` mask is read inside the system rather than captured outside it, because rows are
    allocated and released as the world runs and a captured Selection would go on aging last
    tick's population. This is the wiring a world assembly (#115) does for every system.
    """
    return TickLoop(store, systems=[lambda: aging.advance(Selection.from_mask(store.alive))])


def ages_of(store, ids):
    rows = [store._id_to_row[i] for i in np.asarray(ids).tolist()]
    return store.age[rows].tolist()


class TestColumnOwnership:
    def test_claims_the_age_column(self):
        _, registry, _ = make_world()
        assert registry.owner_of("age") == "Aging"

    def test_a_rival_service_cannot_also_claim_age(self):
        store, registry, _ = make_world()

        class RivalAging(Aging):
            pass

        with pytest.raises(ColumnOwnershipError):
            RivalAging(store, registry)


class TestTicksLived:
    def test_a_single_tick_ages_everyone_alive_by_one(self):
        store, _, aging = make_world()
        ids = store.allocate(3)

        aging_loop(store, aging).advance(1)

        assert ages_of(store, ids) == [1, 1, 1]

    def test_age_counts_whole_ticks_lived(self):
        store, _, aging = make_world()
        ids = store.allocate(2)

        aging_loop(store, aging).advance(10)

        assert ages_of(store, ids) == [10, 10]

    def test_an_entity_allocated_partway_through_is_younger_by_its_offset(self):
        """Age is ticks *lived*, not ticks elapsed, so a newcomer is not born into the world's
        history. This also pins that the system ages whoever is alive at the tick it runs, rather
        than the population that existed when the loop was built.
        """
        store, _, aging = make_world()
        loop = aging_loop(store, aging)
        founder = store.allocate(1)

        loop.advance(3)
        newcomer = store.allocate(1)
        loop.advance(4)

        assert ages_of(store, founder) == [7]
        assert ages_of(store, newcomer) == [4]

    def test_a_reused_row_does_not_inherit_its_predecessors_age(self):
        """A capacity of one forces the free list to hand the same row back out, which is the
        case that would silently make a newborn old: `age` is per-occupant, not per-row.
        """
        store, _, aging = make_world(initial_capacity=1)
        loop = aging_loop(store, aging)
        first = store.allocate(1)
        loop.advance(5)
        assert ages_of(store, first) == [5]

        store.release(first)
        second = store.allocate(1)
        assert ages_of(store, second) == [0]

        loop.advance(2)
        assert ages_of(store, second) == [2]

    def test_batched_advance_matches_stepwise_advance(self):
        """CLAUDE.md §2.4: the wake schedule decides when compute happens, never how fast the
        world moves. Batching a week of catch-up into one call must age the world by exactly what
        a week of live ticks would.
        """
        batched_store, _, batched_aging = make_world()
        batched_ids = batched_store.allocate(3)
        aging_loop(batched_store, batched_aging).advance(1000)

        stepwise_store, _, stepwise_aging = make_world()
        stepwise_ids = stepwise_store.allocate(3)
        stepwise_loop = aging_loop(stepwise_store, stepwise_aging)
        for _ in range(1000):
            stepwise_loop.advance(1)

        assert ages_of(batched_store, batched_ids) == ages_of(stepwise_store, stepwise_ids)
        assert ages_of(batched_store, batched_ids) == [1000, 1000, 1000]

    def test_only_the_selection_ages(self):
        store, _, aging = make_world()
        ids = store.allocate(3)
        rows = [store._id_to_row[i] for i in ids.tolist()]

        aging.advance(Selection.from_indices(np.array(rows[:2]), capacity=store.capacity))

        assert store.age[rows].tolist() == [1, 1, 0]
