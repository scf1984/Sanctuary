import numpy as np
import pytest

from core.genetics.species import SpeciesRegistry, UnknownSpeciesError
from core.genetics.vocabulary import GeneVocabulary


def make_registry():
    vocabulary = GeneVocabulary(("size", "speed", "sight", "clutch_size"))
    return SpeciesRegistry(vocabulary), vocabulary


class TestRegister:
    def test_ids_are_assigned_sequentially_from_zero(self):
        registry, _ = make_registry()
        first = registry.register(("size", "speed"))
        second = registry.register(("sight",))
        assert first == 0
        assert second == 1
        assert registry.n_species == 2

    def test_mask_marks_only_expressed_genes(self):
        registry, _ = make_registry()
        species_id = registry.register(("size", "sight"))
        assert registry.mask_of(species_id).tolist() == [True, False, True, False]

    def test_species_expressing_no_genes_is_allowed(self):
        registry, _ = make_registry()
        species_id = registry.register(())
        assert not registry.mask_of(species_id).any()

    def test_unknown_gene_name_raises(self):
        registry, _ = make_registry()
        with pytest.raises(KeyError):
            registry.register(("not_a_gene",))


class TestMaskOf:
    def test_unregistered_species_raises(self):
        registry, _ = make_registry()
        with pytest.raises(UnknownSpeciesError):
            registry.mask_of(0)

    def test_negative_species_id_raises(self):
        registry, _ = make_registry()
        registry.register(("size",))
        with pytest.raises(UnknownSpeciesError):
            registry.mask_of(-1)


class TestMasksFor:
    def test_vectorized_lookup_across_mixed_species(self):
        registry, _ = make_registry()
        species_a = registry.register(("size", "speed"))
        species_b = registry.register(("sight", "clutch_size"))

        masks = registry.masks_for(np.array([species_a, species_b, species_a], dtype=np.int64))

        assert masks.shape == (3, 4)
        assert masks[0].tolist() == [True, True, False, False]
        assert masks[1].tolist() == [False, False, True, True]
        assert masks[2].tolist() == [True, True, False, False]

    def test_empty_selection_returns_empty_array(self):
        registry, _ = make_registry()
        registry.register(("size",))
        masks = registry.masks_for(np.array([], dtype=np.int64))
        assert masks.shape == (0, 4)

    def test_unregistered_species_id_in_batch_raises(self):
        registry, _ = make_registry()
        registry.register(("size",))
        with pytest.raises(UnknownSpeciesError):
            registry.masks_for(np.array([0, 99], dtype=np.int64))
