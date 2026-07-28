import numpy as np
import pytest

from core.selection import Selection
from core.services import ColumnOwnershipError, ColumnRegistry, DomainService


class FakeStore:
    """A minimal stand-in for EntityStore: just the column arrays a service might own."""

    def __init__(self, capacity: int) -> None:
        self.energy = np.zeros(capacity, dtype=np.float32)
        self.species_id = np.full(capacity, -1, dtype=np.int32)
        self.drive_scores = np.zeros((capacity, 2), dtype=np.float32)


class Ecology(DomainService):
    owns = ("energy",)


class Genetics(DomainService):
    owns = ("species_id",)


class Behaviour(DomainService):
    owns = ("drive_scores",)


class NoColumnsService(DomainService):
    pass


class TestDeclaringServices:
    def test_ecology_genetics_and_behaviour_can_be_declared_against_the_base(self):
        store = FakeStore(capacity=4)
        registry = ColumnRegistry()
        ecology = Ecology(store, registry)
        genetics = Genetics(store, registry)
        behaviour = Behaviour(store, registry)

        assert registry.owner_of("energy") == "Ecology"
        assert registry.owner_of("species_id") == "Genetics"
        assert registry.owner_of("drive_scores") == "Behaviour"
        assert ecology.store is store
        assert genetics.store is store
        assert behaviour.store is store

    def test_service_must_declare_at_least_one_column(self):
        with pytest.raises(ValueError):
            NoColumnsService(FakeStore(4), ColumnRegistry())

    def test_second_service_claiming_an_owned_column_is_rejected(self):
        registry = ColumnRegistry()
        Ecology(FakeStore(4), registry)

        class RivalEcology(DomainService):
            owns = ("energy",)

        with pytest.raises(ColumnOwnershipError):
            RivalEcology(FakeStore(4), registry)

    def test_reclaiming_the_same_column_under_the_same_owner_name_does_not_conflict(self):
        registry = ColumnRegistry()
        Ecology(FakeStore(4), registry)
        Ecology(FakeStore(4), registry)  # same owner name reclaiming "energy" — not a conflict


class TestColumnWriteEnforcement:
    def test_service_can_write_its_own_column(self):
        store = FakeStore(capacity=4)
        ecology = Ecology(store, registry=ColumnRegistry())
        selection = Selection.from_indices(np.array([1, 2]), capacity=4)

        ecology.write("energy", selection, np.array([10.0, 20.0], dtype=np.float32))

        assert store.energy.tolist() == [0.0, 10.0, 20.0, 0.0]

    def test_service_cannot_write_a_column_it_does_not_own(self):
        store = FakeStore(capacity=4)
        registry = ColumnRegistry()
        ecology = Ecology(store, registry)
        Genetics(store, registry)
        selection = Selection.all(4)

        with pytest.raises(ColumnOwnershipError):
            ecology.write("species_id", selection, np.array([1, 1, 1, 1], dtype=np.int32))

        # The rejected write must not have touched the column it doesn't own.
        assert (store.species_id == -1).all()

    def test_error_names_the_actual_owner(self):
        store = FakeStore(capacity=4)
        registry = ColumnRegistry()
        ecology = Ecology(store, registry)
        Genetics(store, registry)

        with pytest.raises(ColumnOwnershipError, match="Genetics"):
            ecology.write("species_id", Selection.all(4), np.array([1, 1, 1, 1]))
