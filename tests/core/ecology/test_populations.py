import numpy as np
import pytest

from core.ecology.populations import Populations, PopulationsConfig
from core.entities.store import EntityStore
from core.genetics.species import SpeciesRegistry
from core.genetics.vocabulary import GeneVocabulary
from core.selection import Selection
from core.world.terrain import Terrain

GENE_NAMES = ("size", "scent")


def make_field(grid=11, cell_size=1.0, diffusion_range=2.0, n_species=2):
    terrain = Terrain(np.zeros((grid, grid), dtype=np.float32), cell_size=cell_size)
    species = SpeciesRegistry(GeneVocabulary(GENE_NAMES))
    for _ in range(n_species):
        species.register(GENE_NAMES)
    store = EntityStore(initial_capacity=16, n_drives=1, n_genes=len(GENE_NAMES))
    field = Populations(terrain, species, PopulationsConfig(diffusion_range=diffusion_range))
    return store, species, field


def spawn(store, species_ids, xs, ys):
    ids = store.allocate(
        len(species_ids),
        species_id=np.array(species_ids, dtype=np.int32),
        x=np.array(xs, dtype=np.float32),
        y=np.array(ys, dtype=np.float32),
    )
    rows = [store._id_to_row[i] for i in ids.tolist()]
    return Selection.from_indices(np.array(rows, dtype=np.int64), store.capacity)


class TestConfig:
    def test_rejects_a_non_positive_diffusion_range(self):
        with pytest.raises(ValueError):
            PopulationsConfig(diffusion_range=0.0)


class TestBinning:
    def test_an_entity_registers_where_it_stands(self):
        store, _, field = make_field()
        population = spawn(store, [0], [5.0], [5.0])

        field.rebuild(store, population)

        assert field.sample(np.array([5.0]), np.array([5.0]))[0, 0] > 0.0

    def test_species_do_not_bleed_into_each_others_planes(self):
        """The whole point of a per-species field: a herbivore's presence must not read as a
        predator's, or a threat matrix weighting them differently is meaningless.
        """
        store, _, field = make_field()
        population = spawn(store, [0], [5.0], [5.0])

        field.rebuild(store, population)
        sampled = field.sample(np.array([5.0]), np.array([5.0]))

        assert sampled[0, 0] > 0.0
        assert sampled[0, 1] == pytest.approx(0.0)

    def test_concentration_falls_off_with_distance(self):
        store, _, field = make_field(grid=21)
        population = spawn(store, [0], [10.0], [10.0])

        field.rebuild(store, population)
        near, far = field.sample(np.array([11.0, 16.0]), np.array([10.0, 10.0]))[:, 0]

        assert near > far

    def test_more_animals_mean_more_concentration(self):
        """Ten wolves at 30 metres must outweigh one wolf at 25 — the reading a nearest-neighbour
        query gets backwards, and a reason the field model is not merely the cheaper one.
        """
        store, _, field = make_field(grid=21)
        lone = spawn(store, [0], [10.0], [10.0])
        field.rebuild(store, lone)
        one = field.sample(np.array([10.0]), np.array([10.0]))[0, 0]

        store2, _, field2 = make_field(grid=21)
        crowd = spawn(store2, [0] * 5, [10.0] * 5, [10.0] * 5)
        field2.rebuild(store2, crowd)
        five = field2.sample(np.array([10.0]), np.array([10.0]))[0, 0]

        assert five > one

    def test_rebuilding_forgets_the_previous_tick(self):
        """The field holds no state between rebuilds, so a stale reading is impossible rather
        than merely unlikely — an animal that moved away leaves nothing behind.
        """
        store, _, field = make_field(grid=21)
        population = spawn(store, [0], [10.0], [10.0])
        field.rebuild(store, population)

        store.x[population.to_indices()] = 2.0
        field.rebuild(store, population)

        assert field.sample(np.array([10.0]), np.array([10.0]))[0, 0] == pytest.approx(0.0)

    def test_only_the_given_population_is_binned(self):
        # Far enough apart that the blur cannot reach: three box passes of half-width 2 spread at
        # most 6 cells along each axis independently, and these are 28 apart.
        store, _, field = make_field(grid=41)
        both = spawn(store, [0, 0], [30.0, 2.0], [30.0, 2.0])
        just_one = Selection.from_indices(both.to_indices()[:1], store.capacity)

        field.rebuild(store, just_one)

        assert field.sample(np.array([2.0]), np.array([2.0]))[0, 0] == pytest.approx(0.0)


class TestSpeciation:
    def test_a_species_registered_after_construction_gets_a_plane(self):
        """Speciation is a mask row and an id write (CLAUDE.md §2.3); this field must not need
        telling that it happened.
        """
        store, species, field = make_field(n_species=1)
        assert field.concentration.shape[0] == 1

        daughter = species.derive(0)
        population = spawn(store, [daughter], [5.0], [5.0])
        field.rebuild(store, population)

        assert field.sample(np.array([5.0]), np.array([5.0]))[0, daughter] > 0.0


class TestEdges:
    def test_the_map_edge_is_not_spuriously_empty(self):
        """Zero-padding the blur would make borders read as emptier than they are, and prey would
        find the world's edge safe — an artifact of the grid that selection would still act on.

        Normalizing by in-world cells instead gives a *reflecting* edge, so a cornered animal
        reads as more concentrated, not less: its scent has nowhere outward to go. That is the
        right direction — being cornered should not make you harder to find.
        """
        store, _, field = make_field(grid=21)
        corner = spawn(store, [0], [0.0], [0.0])
        field.rebuild(store, corner)
        at_corner = field.sample(np.array([0.0]), np.array([0.0]))[0, 0]

        store2, _, field2 = make_field(grid=21)
        middle = spawn(store2, [0], [10.0], [10.0])
        field2.rebuild(store2, middle)
        at_middle = field2.sample(np.array([10.0]), np.array([10.0]))[0, 0]

        assert at_corner >= at_middle

    def test_presence_does_not_wrap_around_the_world(self):
        """A cumulative-sum blur that wrapped would carry scent from one edge of the map to the
        other, making opposite corners neighbours.
        """
        store, _, field = make_field(grid=21)
        population = spawn(store, [0], [0.0], [0.0])

        field.rebuild(store, population)

        assert field.sample(np.array([20.0]), np.array([20.0]))[0, 0] == pytest.approx(0.0)

    def test_a_position_outside_the_world_is_rejected(self):
        store, _, field = make_field(grid=11)
        field.rebuild(store, Selection.none(store.capacity))

        with pytest.raises(ValueError):
            field.sample(np.array([99.0]), np.array([0.0]))


class TestDiffusionRange:
    def test_a_wider_range_spreads_presence_further(self):
        narrow_store, _, narrow = make_field(grid=41, diffusion_range=1.0)
        field_population = spawn(narrow_store, [0], [20.0], [20.0])
        narrow.rebuild(narrow_store, field_population)

        wide_store, _, wide = make_field(grid=41, diffusion_range=6.0)
        wide_population = spawn(wide_store, [0], [20.0], [20.0])
        wide.rebuild(wide_store, wide_population)

        probe_x, probe_y = np.array([28.0]), np.array([20.0])
        assert wide.sample(probe_x, probe_y)[0, 0] > narrow.sample(probe_x, probe_y)[0, 0]
