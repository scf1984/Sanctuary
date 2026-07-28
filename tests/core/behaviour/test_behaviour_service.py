import numpy as np
import pytest

from core.behaviour.service import Behaviour, DriveRegistrationError
from core.entities.store import EntityStore
from core.selection import Selection
from core.services import ColumnOwnershipError, ColumnRegistry


class ConstantDrive:
    """A drive whose score is authored by the test rather than derived from the world.

    Every drive the simulation ships reads fields (energy, climate, health) that the scoring loop
    knows nothing about, so exercising the loop itself means supplying scores directly.
    """

    def __init__(self, name, values):
        self.name = name
        self._values = np.asarray(values, dtype=np.float32)

    def score(self, selection):
        return self._values[selection.to_mask()]


def make_store(capacity=8, n_drives=4):
    return EntityStore(initial_capacity=capacity, n_drives=n_drives, n_genes=2)


def selection_for(store, ids):
    rows = [store._id_to_row[i] for i in np.asarray(ids).tolist()]
    return Selection.from_indices(np.array(rows, dtype=np.int64), capacity=store.capacity)


def per_row(store, **row_values):
    """A (capacity,) float32 column with `row_values` placed at the named rows."""
    column = np.zeros(store.capacity, dtype=np.float32)
    for row, value in row_values.items():
        column[int(row)] = value
    return column


class TestColumnOwnership:
    def test_claims_the_drive_scores_column(self):
        registry = ColumnRegistry()
        Behaviour(make_store(), registry)
        assert registry.owner_of("drive_scores") == "Behaviour"

    def test_a_rival_service_cannot_also_claim_drive_scores(self):
        store, registry = make_store(), ColumnRegistry()
        Behaviour(store, registry)

        class RivalBehaviour(Behaviour):
            pass

        with pytest.raises(ColumnOwnershipError):
            RivalBehaviour(store, registry)

    def test_behaviour_cannot_write_a_column_it_does_not_own(self):
        store = make_store()
        behaviour = Behaviour(store, ColumnRegistry())
        ids = store.allocate(1)

        with pytest.raises(ColumnOwnershipError):
            behaviour.write("energy", selection_for(store, ids), np.zeros(1, dtype=np.float32))


class TestRegistration:
    def test_drive_names_are_reported_in_registration_order(self):
        behaviour = Behaviour(make_store(), ColumnRegistry())
        behaviour.register(ConstantDrive("hunger", np.zeros(8)))
        behaviour.register(ConstantDrive("fear", np.zeros(8)))

        assert behaviour.drive_names == ("hunger", "fear")

    def test_two_drives_cannot_share_a_name(self):
        """Names are how the viewer and every downstream consumer address a drive; two drives
        answering to one name would make `driven_by` silently report the wrong one.
        """
        behaviour = Behaviour(make_store(), ColumnRegistry())
        behaviour.register(ConstantDrive("hunger", np.zeros(8)))

        with pytest.raises(DriveRegistrationError):
            behaviour.register(ConstantDrive("hunger", np.zeros(8)))

    def test_registering_past_the_column_width_fails_loudly(self):
        """The store's drive_scores block is allocated at a fixed width. Overflowing it is a
        world-construction error, and it is caught here rather than at the first tick (§8.7).
        """
        behaviour = Behaviour(make_store(n_drives=2), ColumnRegistry())
        behaviour.register(ConstantDrive("hunger", np.zeros(8)))
        behaviour.register(ConstantDrive("thirst", np.zeros(8)))

        with pytest.raises(DriveRegistrationError):
            behaviour.register(ConstantDrive("fear", np.zeros(8)))

    def test_a_drive_returning_the_wrong_length_fails_loudly(self):
        """A scalar or length-1 return would broadcast into the score column silently, giving
        every entity one animal's motivation.
        """
        store = make_store()
        behaviour = Behaviour(store, ColumnRegistry())
        behaviour.register(ConstantDrive("hunger", np.zeros(8)))
        ids = store.allocate(3)

        class ScalarDrive:
            name = "broken"

            def score(self, selection):
                return np.float32(1.0)

        behaviour.register(ScalarDrive())

        with pytest.raises(ValueError):
            behaviour.score(selection_for(store, ids))


class TestScoring:
    def test_scores_are_written_in_registration_order(self):
        store = make_store()
        behaviour = Behaviour(store, ColumnRegistry())
        ids = store.allocate(2)
        rows = selection_for(store, ids).to_indices()
        behaviour.register(ConstantDrive("hunger", per_row(store, **{str(rows[0]): 0.25})))
        behaviour.register(ConstantDrive("thirst", per_row(store, **{str(rows[1]): 0.75})))

        selection = selection_for(store, ids)
        behaviour.score(selection)

        # Sliced to the registered prefix; the store was built with columns to spare.
        assert behaviour.scores(selection)[:, :2] == pytest.approx(
            np.array([[0.25, 0.0], [0.0, 0.75]], dtype=np.float32)
        )

    def test_scoring_leaves_entities_outside_the_selection_untouched(self):
        store = make_store()
        behaviour = Behaviour(store, ColumnRegistry())
        ids = store.allocate(2)
        behaviour.register(ConstantDrive("hunger", np.full(store.capacity, 0.5)))

        scored = selection_for(store, ids[:1])
        untouched = selection_for(store, ids[1:])
        behaviour.score(scored)

        assert behaviour.scores(untouched) == pytest.approx(np.zeros((1, 4), dtype=np.float32))

    def test_adding_a_drive_needs_no_change_to_the_scoring_loop(self):
        """Issue #22's "done when": a new drive is registered, not wired into a dispatch chain.

        The loop is exercised with two drives and then three, with nothing about the call
        changing -- which is the property the registry exists to provide.
        """
        store = make_store()
        behaviour = Behaviour(store, ColumnRegistry())
        ids = store.allocate(1)
        selection = selection_for(store, ids)
        row = str(selection.to_indices()[0])
        behaviour.register(ConstantDrive("hunger", per_row(store, **{row: 0.1})))
        behaviour.register(ConstantDrive("thirst", per_row(store, **{row: 0.2})))

        behaviour.score(selection)
        assert behaviour.breakdown(selection) == {
            "hunger": pytest.approx([0.1]),
            "thirst": pytest.approx([0.2]),
        }

        behaviour.register(ConstantDrive("fear", per_row(store, **{row: 0.9})))
        behaviour.score(selection)

        assert behaviour.breakdown(selection)["fear"] == pytest.approx([0.9])
        assert behaviour.driven_by("fear", selection) == selection


class TestCompetition:
    def test_the_highest_scoring_drive_wins(self):
        store = make_store()
        behaviour = Behaviour(store, ColumnRegistry())
        ids = store.allocate(3)
        rows = [str(r) for r in selection_for(store, ids).to_indices()]
        behaviour.register(
            ConstantDrive("hunger", per_row(store, **{rows[0]: 0.9, rows[1]: 0.1, rows[2]: 0.0}))
        )
        behaviour.register(
            ConstantDrive("fear", per_row(store, **{rows[0]: 0.2, rows[1]: 0.8, rows[2]: 0.0}))
        )

        selection = selection_for(store, ids)
        behaviour.score(selection)

        assert behaviour.driven_by("hunger", selection) == selection_for(store, ids[:1])
        assert behaviour.driven_by("fear", selection) == selection_for(store, ids[1:2])

    def test_an_entity_with_no_active_drive_is_driven_by_nothing(self):
        """A creature that is fed, cool, safe, immature and healthy has no reason to act. Letting
        argmax hand it to whichever drive happens to be registered first would fabricate a
        motivation out of an all-zero row (§8.7).
        """
        store = make_store()
        behaviour = Behaviour(store, ColumnRegistry())
        ids = store.allocate(1)
        selection = selection_for(store, ids)
        behaviour.register(ConstantDrive("hunger", np.zeros(store.capacity)))
        behaviour.register(ConstantDrive("fear", np.zeros(store.capacity)))

        behaviour.score(selection)

        assert behaviour.winning_drive(selection) == pytest.approx([-1])
        assert len(behaviour.driven_by("hunger", selection)) == 0
        assert len(behaviour.driven_by("fear", selection)) == 0

    def test_ties_resolve_to_the_earlier_registered_drive(self):
        """Issue #22 requires ties resolve deterministically given the same scores. Registration
        order is the only ordering available, and it is stable across runs.
        """
        store = make_store()
        behaviour = Behaviour(store, ColumnRegistry())
        ids = store.allocate(1)
        behaviour.register(ConstantDrive("hunger", np.full(store.capacity, 0.5)))
        behaviour.register(ConstantDrive("fear", np.full(store.capacity, 0.5)))

        selection = selection_for(store, ids)
        behaviour.score(selection)

        assert behaviour.driven_by("hunger", selection) == selection
        assert len(behaviour.driven_by("fear", selection)) == 0

    def test_an_unregistered_score_column_never_wins(self):
        """The store's block is wider than the drives registered against it; the trailing columns
        hold zeros that argmax would otherwise be free to select.
        """
        store = make_store(n_drives=4)
        behaviour = Behaviour(store, ColumnRegistry())
        ids = store.allocate(1)
        selection = selection_for(store, ids)
        behaviour.register(ConstantDrive("hunger", np.full(store.capacity, 0.3)))

        behaviour.score(selection)

        assert behaviour.winning_drive(selection) == pytest.approx([0])

    def test_driven_by_rejects_a_name_no_drive_answers_to(self):
        behaviour = Behaviour(make_store(), ColumnRegistry())
        behaviour.register(ConstantDrive("hunger", np.zeros(8)))

        with pytest.raises(KeyError):
            behaviour.driven_by("gluttony", Selection.none(8))


class TestInspection:
    def test_breakdown_names_every_registered_drives_score(self):
        """§3.3's click-to-inspect needs "why did it do that", which is the whole score row
        labelled by drive, not just the winner.
        """
        store = make_store()
        behaviour = Behaviour(store, ColumnRegistry())
        ids = store.allocate(1)
        selection = selection_for(store, ids)
        row = str(selection.to_indices()[0])
        behaviour.register(ConstantDrive("hunger", per_row(store, **{row: 0.4})))
        behaviour.register(ConstantDrive("fatigue", per_row(store, **{row: 0.6})))

        behaviour.score(selection)

        assert behaviour.breakdown(selection) == {
            "hunger": pytest.approx([0.4]),
            "fatigue": pytest.approx([0.6]),
        }
