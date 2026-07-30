import numpy as np
import pytest

from core.entities.store import EntityStore
from core.genetics.distance import between, centroid_between
from core.genetics.expression import GeneticsConfig
from core.genetics.service import Genetics
from core.genetics.species import SpeciesRegistry
from core.selection import Selection
from core.services import ColumnRegistry

from tests.support.genes import gene_registry

GENE_NAMES = ("size", "speed", "sight", "camouflage", "clutch_size", "gestation", "mutability")

# Every gene declares how its stored value is read (#104). These are all quantities, so all fold
# across zero; `mutability` is in the vocabulary because inheritance's spread floor is a gene, and
# every world needs one even when — as here — nothing in these tests breeds.
GENETICS_CONFIG = GeneticsConfig(
    mutability_gene="mutability",
    drift_margin=2.0,
)
GENE_REGISTRY = gene_registry(GENE_NAMES)


def make_world(rng, n_entities):
    """A store seeded with `n_entities` creatures split across three species with distinct,
    overlapping expression masks and random gene values -- so property tests below exercise
    distance across genuinely different masks, not just one species end to end.
    """
    store = EntityStore(initial_capacity=n_entities, n_drives=1, n_genes=len(GENE_NAMES))
    vocabulary = GENE_REGISTRY
    species = SpeciesRegistry(vocabulary)
    genetics = Genetics(store, ColumnRegistry(), species, vocabulary, GENETICS_CONFIG)

    species_ids = [
        species.register(
            tuple(
                rng.choice(
                    GENE_NAMES, size=rng.integers(1, len(GENE_NAMES) + 1), replace=False
                ).tolist()
            )
        )
        for _ in range(3)
    ]
    assigned = rng.choice(species_ids, size=n_entities).astype(np.int32)
    genes = rng.uniform(-5.0, 5.0, size=(n_entities, len(GENE_NAMES))).astype(np.float32)
    ids = store.allocate(n_entities, species_id=assigned, genes=genes)
    rows = np.array([store._id_to_row[i] for i in ids.tolist()], dtype=np.int64)
    return store, genetics, rows


def selection_of(store, rows, indices):
    mask = np.zeros(store.capacity, dtype=np.bool_)
    mask[rows[np.asarray(indices, dtype=np.int64)]] = True
    return Selection.from_mask(mask)


def three_populations(seed, k=5, n_entities=15):
    """Three equal-size, independently-drawn (possibly overlapping) selections over one random
    world, for exercising `between`'s pairwise metric properties.
    """
    rng = np.random.default_rng(seed)
    store, genetics, rows = make_world(rng, n_entities)
    a = selection_of(store, rows, rng.choice(n_entities, size=k, replace=False))
    b = selection_of(store, rows, rng.choice(n_entities, size=k, replace=False))
    c = selection_of(store, rows, rng.choice(n_entities, size=k, replace=False))
    return genetics, a, b, c


class TestBetweenBasics:
    def test_distance_is_euclidean_over_expressed_phenotype(self):
        store = EntityStore(initial_capacity=2, n_drives=1, n_genes=len(GENE_NAMES))
        vocabulary = GENE_REGISTRY
        species = SpeciesRegistry(vocabulary)
        genetics = Genetics(store, ColumnRegistry(), species, vocabulary, GENETICS_CONFIG)
        species_id = species.register(GENE_NAMES)  # expresses every gene: exact Euclidean check

        ids = store.allocate(2, species_id=np.array([species_id, species_id], dtype=np.int32))
        rows = np.array([store._id_to_row[i] for i in ids.tolist()], dtype=np.int64)
        store.genes[rows] = np.array(
            [[0.0] * len(GENE_NAMES), [3.0, 4.0, *([0.0] * (len(GENE_NAMES) - 2))]],
            dtype=np.float32,
        )

        a = selection_of(store, rows, [0])
        b = selection_of(store, rows, [1])
        assert between(genetics, a, b)[0] == pytest.approx(5.0)  # 3-4-5 triangle

    def test_mismatched_length_selections_raise(self):
        rng = np.random.default_rng(1)
        store, genetics, rows = make_world(rng, n_entities=4)
        a = selection_of(store, rows, [0, 1])
        b = selection_of(store, rows, [2])
        with pytest.raises(ValueError):
            between(genetics, a, b)

    def test_unexpressed_genes_do_not_affect_distance(self):
        store = EntityStore(initial_capacity=2, n_drives=1, n_genes=len(GENE_NAMES))
        vocabulary = GENE_REGISTRY
        species = SpeciesRegistry(vocabulary)
        genetics = Genetics(store, ColumnRegistry(), species, vocabulary, GENETICS_CONFIG)
        species_id = species.register(("size", "speed"))

        ids = store.allocate(2, species_id=np.array([species_id, species_id], dtype=np.int32))
        rows = np.array([store._id_to_row[i] for i in ids.tolist()], dtype=np.int64)
        genes = np.zeros((2, len(GENE_NAMES)), dtype=np.float32)
        genes[:, GENE_NAMES.index("size")] = 1.0
        genes[:, GENE_NAMES.index("speed")] = 2.0
        genes[:, GENE_NAMES.index("sight")] = [999.0, -123.0]
        genes[:, GENE_NAMES.index("camouflage")] = [-999.0, 456.0]
        store.genes[rows] = genes

        a = selection_of(store, rows, [0])
        b = selection_of(store, rows, [1])
        # Only "size" and "speed" are expressed by this species, and both creatures agree on
        # those -- so distance is zero even though "sight"/"camouflage" differ wildly, because
        # those slots are unexpressed and never enter the comparison.
        assert between(genetics, a, b)[0] == pytest.approx(0.0)


class TestCentroidBetweenBasics:
    def test_empty_selection_raises(self):
        rng = np.random.default_rng(2)
        store, genetics, rows = make_world(rng, n_entities=3)
        a = selection_of(store, rows, [0])
        empty = Selection.none(store.capacity)
        with pytest.raises(ValueError):
            centroid_between(genetics, a, empty)


@pytest.mark.parametrize("seed", range(25))
class TestMetricProperties:
    def test_between_is_symmetric(self, seed):
        genetics, a, b, _ = three_populations(seed)
        np.testing.assert_allclose(between(genetics, a, b), between(genetics, b, a))

    def test_between_satisfies_triangle_inequality(self, seed):
        genetics, a, b, c = three_populations(seed)
        direct = between(genetics, a, c)
        via_b = between(genetics, a, b) + between(genetics, b, c)
        assert (direct <= via_b + 1e-4).all()

    def test_between_a_selection_and_itself_is_zero(self, seed):
        genetics, a, _, _ = three_populations(seed)
        np.testing.assert_allclose(between(genetics, a, a), 0.0, atol=1e-5)

    def test_centroid_between_is_symmetric(self, seed):
        genetics, a, b, _ = three_populations(seed)
        assert centroid_between(genetics, a, b) == pytest.approx(centroid_between(genetics, b, a))

    def test_centroid_between_satisfies_triangle_inequality(self, seed):
        genetics, a, b, c = three_populations(seed)
        direct = centroid_between(genetics, a, c)
        via_b = centroid_between(genetics, a, b) + centroid_between(genetics, b, c)
        assert direct <= via_b + 1e-4

    def test_centroid_between_a_population_and_itself_is_zero(self, seed):
        genetics, a, _, _ = three_populations(seed)
        assert centroid_between(genetics, a, a) == pytest.approx(0.0, abs=1e-5)
