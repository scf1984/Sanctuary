import numpy as np
import pytest

from core.entities.store import EntityStore
from core.selection import Selection
from core.view import EntityView, TickContext


def make_store_and_selection():
    store = EntityStore(initial_capacity=4, n_drives=2)
    ids = store.allocate(
        2,
        x=np.array([1.0, 2.0], dtype=np.float32),
        energy=np.array([10.0, 20.0], dtype=np.float32),
    )
    row = store._id_to_row[ids[1].item()]
    selection = Selection.from_indices(np.array([row]), capacity=store.capacity)
    return store, selection


class TestConstruction:
    def test_exposes_requested_columns_as_python_scalars(self):
        store, selection = make_store_and_selection()
        view = EntityView(store, selection, ("x", "energy"), TickContext())
        assert view.x == pytest.approx(2.0)
        assert view.energy == pytest.approx(20.0)

    def test_array_valued_column_becomes_a_python_list(self):
        store, selection = make_store_and_selection()
        view = EntityView(store, selection, ("drive_scores",), TickContext())
        assert isinstance(view.drive_scores, list)

    def test_unrequested_column_raises_attribute_error(self):
        store, selection = make_store_and_selection()
        view = EntityView(store, selection, ("x",), TickContext())
        with pytest.raises(AttributeError):
            view.energy

    def test_rejects_a_selection_of_more_than_one_row(self):
        store = EntityStore(initial_capacity=4, n_drives=2)
        store.allocate(2)
        selection = Selection.all(store.capacity)
        with pytest.raises(ValueError):
            EntityView(store, selection, ("x",), TickContext())

    def test_rejects_a_selection_of_zero_rows(self):
        store, _ = make_store_and_selection()
        with pytest.raises(ValueError):
            EntityView(store, Selection.none(store.capacity), ("x",), TickContext())


class TestHotLoopGuard:
    def test_construction_succeeds_outside_a_tick(self):
        store, selection = make_store_and_selection()
        EntityView(store, selection, ("x",), TickContext())  # does not raise

    def test_construction_is_rejected_during_a_tick(self):
        store, selection = make_store_and_selection()
        tick_context = TickContext()
        with tick_context.tick():
            with pytest.raises(RuntimeError):
                EntityView(store, selection, ("x",), tick_context)

    def test_construction_succeeds_again_after_the_tick_ends(self):
        store, selection = make_store_and_selection()
        tick_context = TickContext()
        with tick_context.tick():
            pass
        EntityView(store, selection, ("x",), tick_context)  # does not raise


class TestTickContext:
    def test_tick_is_not_reentrant(self):
        tick_context = TickContext()
        with tick_context.tick():
            with pytest.raises(RuntimeError):
                with tick_context.tick():
                    pass

    def test_in_tick_resets_after_an_exception(self):
        tick_context = TickContext()
        with pytest.raises(ValueError):
            with tick_context.tick():
                raise ValueError("boom")
        assert tick_context.in_tick is False
