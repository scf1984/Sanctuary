import numpy as np
import pytest

from core.entities.store import EntityStore
from core.genetics.service import Genetics
from core.genetics.species import SpeciesRegistry
from core.genetics.vocabulary import GeneVocabulary
from core.selection import Selection
from core.services import ColumnOwnershipError, ColumnRegistry


GENE_NAMES = ("size", "speed", "sight", "camouflage")


def make_world(initial_capacity=8):
    store = EntityStore(initial_capacity=initial_capacity, n_drives=1, n_genes=len(GENE_NAMES))
    vocabulary = GeneVocabulary(GENE_NAMES)
    species = SpeciesRegistry(vocabulary)
    genetics = Genetics(store, ColumnRegistry(), species)
    return store, species, genetics


def selection_for(store, ids):
    rows = [store._id_to_row[i] for i in np.asarray(ids).tolist()]
    return Selection.from_indices(np.array(rows, dtype=np.int64), capacity=store.capacity)


class TestColumnOwnership:
    def test_claims_genes_and_species_id(self):
        store, species, _ = make_world()
        registry = ColumnRegistry()
        Genetics(store, registry, species)
        assert registry.owner_of("genes") == "Genetics"
        assert registry.owner_of("species_id") == "Genetics"

    def test_a_rival_service_cannot_also_claim_genes(self):
        store, species, _ = make_world()
        registry = ColumnRegistry()
        Genetics(store, registry, species)

        class RivalGenetics(Genetics):
            pass

        with pytest.raises(ColumnOwnershipError):
            RivalGenetics(store, registry, species)


class TestGenesReadWrite:
    def test_set_and_get_genes_for_a_selection(self):
        store, _, genetics = make_world()
        ids = store.allocate(2)
        selection = selection_for(store, ids)

        values = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
        genetics.set_genes(selection, values)

        assert genetics.genes(selection).tolist() == values.tolist()

    def test_reads_and_writes_are_vectorized_across_mixed_species(self):
        store, species, genetics = make_world()
        species_a = species.register(("size", "speed"))
        species_b = species.register(("sight", "camouflage"))
        ids = store.allocate(3, species_id=np.array([species_a, species_b, species_a], dtype=np.int32))
        selection = selection_for(store, ids)

        values = np.array(
            [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]], dtype=np.float32
        )
        genetics.set_genes(selection, values)

        expressed = genetics.expressed(selection)
        assert expressed.tolist() == [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 2.0],
            [3.0, 3.0, 0.0, 0.0],
        ]
        # The genotype is untouched by masking -- only expressed() applies it.
        assert genetics.genes(selection).tolist() == values.tolist()


class TestSpeciate:
    def test_speciate_writes_species_id_for_every_row_in_the_selection(self):
        store, species, genetics = make_world()
        species_id = species.register(("size",))
        ids = store.allocate(2)
        selection = selection_for(store, ids)

        genetics.speciate(selection, species_id)

        rows = [store._id_to_row[i] for i in ids.tolist()]
        assert (store.species_id[rows] == species_id).all()


class TestUnexpressedGenesSurviveInheritance:
    """CLAUDE.md §2.3 / issue #13's "done when": an unexpressed gene is inert but still
    inherited, so it can resurface generations later if a descendant's species expresses it.

    The mutation/drift mechanics of inheritance itself belong to #14; here we only need to show
    that copying a full gene row into an offspring -- the primitive #14 will build on -- carries
    every gene along regardless of what either parent's or the offspring's species expresses.
    """

    def test_offspring_gene_row_is_intact_even_when_the_species_never_expressed_it(self):
        store, species, genetics = make_world()
        # "camouflage" is dormant in this lineage: neither species expresses it.
        ancestor_species = species.register(("size", "speed"))
        descendant_species = species.register(("size", "sight"))

        [parent_id] = store.allocate(1, species_id=np.array([ancestor_species], dtype=np.int32))
        parent_selection = selection_for(store, [parent_id])
        # Exactly representable in float32 so equality assertions below aren't rounding-sensitive.
        parent_genes = np.array([[0.5, 0.25, 0.125, 0.75]], dtype=np.float32)
        genetics.set_genes(parent_selection, parent_genes)

        # camouflage (index 3) is unexpressed by the parent's species -- confirm the phenotype
        # hides it even though we just wrote a real value for it.
        assert genetics.expressed(parent_selection).tolist() == [[0.5, 0.25, 0.0, 0.0]]

        [offspring_id] = store.allocate(
            1, species_id=np.array([descendant_species], dtype=np.int32)
        )
        offspring_selection = selection_for(store, [offspring_id])
        # Stand-in for #14's inheritance: copy the parent's full genotype into the offspring.
        genetics.set_genes(offspring_selection, genetics.genes(parent_selection))

        # The raw genotype survived the copy untouched, including "camouflage" -- unexpressed by
        # both the ancestor's and the descendant's species the whole time.
        assert genetics.genes(offspring_selection).tolist() == parent_genes.tolist()

        # The descendant's species doesn't express camouflage or speed either, so the phenotype
        # still hides both -- but the genotype carrying speed=0.25 and camouflage=0.125 never left.
        assert genetics.expressed(offspring_selection).tolist() == [[0.5, 0.0, 0.125, 0.0]]

        # Speciation is only ever an id write (CLAUDE.md §2.3): re-registering a species that
        # expresses camouflage and reassigning the offspring to it reveals the dormant value
        # unchanged -- no gene write happened between inheritance and now.
        camouflage_expressing_species = species.register(("size", "camouflage"))
        genetics.speciate(offspring_selection, camouflage_expressing_species)
        assert genetics.expressed(offspring_selection).tolist() == [[0.5, 0.0, 0.0, 0.75]]


class TestInherit:
    def test_rejects_unequal_length_parent_selections(self):
        store, _, genetics = make_world()
        ids = store.allocate(3)
        parent_a = selection_for(store, ids[:2])
        parent_b = selection_for(store, ids[:1])

        with pytest.raises(ValueError):
            genetics.inherit(parent_a, parent_b, inherit_gain=1.5, rng=np.random.default_rng(0))

    def test_returns_one_offspring_row_per_parent_pair_within_clamp_range(self):
        store, _, genetics = make_world()
        ids = store.allocate(4)
        parent_a_selection = selection_for(store, ids[:2])
        parent_b_selection = selection_for(store, ids[2:])
        genetics.set_genes(
            parent_a_selection,
            np.array([[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        )
        genetics.set_genes(
            parent_b_selection,
            np.array([[5.0, 6.0, 7.0, 8.0], [2.0, 2.0, 2.0, 2.0]], dtype=np.float32),
        )
        inherit_gain = 1.5

        offspring_genes = genetics.inherit(
            parent_a_selection, parent_b_selection, inherit_gain, np.random.default_rng(0)
        )

        parent_a_genes = genetics.genes(parent_a_selection)
        parent_b_genes = genetics.genes(parent_b_selection)
        low = np.minimum(parent_a_genes, parent_b_genes) / inherit_gain
        high = np.maximum(parent_a_genes, parent_b_genes) * inherit_gain
        assert offspring_genes.shape == (2, 4)
        assert (offspring_genes >= low).all()
        assert (offspring_genes <= high).all()

    def test_offspring_genes_can_be_written_into_a_newly_allocated_entity(self):
        store, _, genetics = make_world()
        [parent_a_id] = store.allocate(1)
        [parent_b_id] = store.allocate(1)
        parent_a_selection = selection_for(store, [parent_a_id])
        parent_b_selection = selection_for(store, [parent_b_id])
        genetics.set_genes(parent_a_selection, np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32))
        genetics.set_genes(parent_b_selection, np.array([[5.0, 6.0, 7.0, 8.0]], dtype=np.float32))

        offspring_genes = genetics.inherit(
            parent_a_selection, parent_b_selection, inherit_gain=1.5, rng=np.random.default_rng(0)
        )
        [offspring_id] = store.allocate(1, genes=offspring_genes)
        offspring_selection = selection_for(store, [offspring_id])

        assert genetics.genes(offspring_selection).tolist() == offspring_genes.tolist()
