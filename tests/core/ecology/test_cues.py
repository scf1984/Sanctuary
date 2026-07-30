import numpy as np
import pytest

from core.ecology.cues import CueField, CueFieldConfig, Scent, ScentGenes
from core.entities.store import EntityStore
from core.genetics.expression import GeneticsConfig
from core.genetics.service import Genetics
from core.genetics.species import SpeciesRegistry
from core.selection import Selection
from core.services import ColumnRegistry
from core.world.terrain import Terrain

from tests.support.genes import gene_registry

CHANNELS = 3
SIGNATURE_GENES = tuple(f"signature_{i}" for i in range(CHANNELS))
GENE_NAMES = ("scent_emission", *SIGNATURE_GENES, "mutability")
SCENT_GENES = ScentGenes(emission_gene="scent_emission", signature_genes=SIGNATURE_GENES)

# A signature is a position in cue space, so its sign is information and it is read raw (#104).
# Emission is a broadcast strength and cannot be negative; mutability is a spread.
GENETICS_CONFIG = GeneticsConfig(
    mutability_gene="mutability",
    drift_margin=2.0,
)
GENE_REGISTRY = gene_registry(GENE_NAMES)


def make_field(grid=21, cell_size=1.0, diffusion_range=2.0, channels=CHANNELS):
    terrain = Terrain(np.zeros((grid, grid), dtype=np.float32), cell_size=cell_size)
    return terrain, CueField(terrain, channels, CueFieldConfig(diffusion_range=diffusion_range))


def signature(*values):
    """A (1, CHANNELS) signature row."""
    return np.array([values], dtype=np.float32)


class TestConfig:
    def test_rejects_a_non_positive_diffusion_range(self):
        with pytest.raises(ValueError):
            CueFieldConfig(diffusion_range=0.0)

    def test_rejects_a_field_with_no_channels(self):
        terrain = Terrain(np.zeros((5, 5), dtype=np.float32), cell_size=1.0)
        with pytest.raises(ValueError):
            CueField(terrain, 0, CueFieldConfig(diffusion_range=1.0))


class TestDeposit:
    def test_a_broadcaster_registers_where_it_stands(self):
        _, field = make_field()
        field.rebuild(np.array([10.0]), np.array([10.0]), np.array([1.0]), signature(1, 0, 0))

        assert field.sample(np.array([10.0]), np.array([10.0]))[0, 0] > 0.0

    def test_channels_do_not_bleed_into_each_other(self):
        """Cue space only means anything if a signature in one channel stays there — otherwise
        every creature smells alike and aversion cannot discriminate.
        """
        _, field = make_field()
        field.rebuild(np.array([10.0]), np.array([10.0]), np.array([1.0]), signature(1, 0, 0))

        sampled = field.sample(np.array([10.0]), np.array([10.0]))[0]

        assert sampled[0] > 0.0
        assert sampled[1] == pytest.approx(0.0)
        assert sampled[2] == pytest.approx(0.0)

    def test_emission_and_signature_are_different_things(self):
        """Emission is how loud, signature is what kind. A quiet distinctive animal and a loud
        bland one must not be interchangeable.
        """
        _, loud = make_field()
        loud.rebuild(np.array([10.0]), np.array([10.0]), np.array([4.0]), signature(1, 0, 0))
        _, quiet = make_field()
        quiet.rebuild(np.array([10.0]), np.array([10.0]), np.array([1.0]), signature(1, 0, 0))

        probe = (np.array([10.0]), np.array([10.0]))
        assert loud.sample(*probe)[0, 0] == pytest.approx(4.0 * quiet.sample(*probe)[0, 0])

    def test_concentration_falls_off_with_distance(self):
        _, field = make_field()
        field.rebuild(np.array([10.0]), np.array([10.0]), np.array([1.0]), signature(1, 0, 0))

        near, far = field.sample(np.array([11.0, 16.0]), np.array([10.0, 10.0]))[:, 0]

        assert near > far

    def test_broadcasters_accumulate(self):
        """Ten wolves 30 world units off outweigh one at 25 — the reading a nearest-neighbour query gets
        backwards, and a reason the field is not merely the cheaper model.
        """
        _, one = make_field()
        one.rebuild(np.array([10.0]), np.array([10.0]), np.array([1.0]), signature(1, 0, 0))
        _, five = make_field()
        five.rebuild(
            np.full(5, 10.0),
            np.full(5, 10.0),
            np.ones(5),
            np.tile(signature(1, 0, 0), (5, 1)),
        )

        probe = (np.array([10.0]), np.array([10.0]))
        assert five.sample(*probe)[0, 0] > one.sample(*probe)[0, 0]

    def test_rebuilding_forgets_the_previous_tick(self):
        _, field = make_field()
        field.rebuild(np.array([10.0]), np.array([10.0]), np.array([1.0]), signature(1, 0, 0))
        field.rebuild(np.array([2.0]), np.array([2.0]), np.array([1.0]), signature(1, 0, 0))

        assert field.sample(np.array([10.0]), np.array([10.0]))[0, 0] == pytest.approx(0.0)

    def test_an_empty_population_leaves_an_empty_field(self):
        _, field = make_field()
        field.rebuild(
            np.zeros(0), np.zeros(0), np.zeros(0), np.zeros((0, CHANNELS), dtype=np.float32)
        )

        assert field.sample(np.array([10.0]), np.array([10.0]))[0] == pytest.approx(
            np.zeros(CHANNELS)
        )

    def test_rejects_a_signature_of_the_wrong_width(self):
        _, field = make_field()
        with pytest.raises(ValueError):
            field.rebuild(
                np.array([1.0]), np.array([1.0]), np.array([1.0]), np.zeros((1, 2), np.float32)
            )

    def test_rejects_negative_emission(self):
        _, field = make_field()
        with pytest.raises(ValueError):
            field.rebuild(
                np.array([1.0]), np.array([1.0]), np.array([-1.0]), signature(1, 0, 0)
            )


class TestSelfExclusion:
    """CLAUDE.md §2.5: a creature does not perceive itself.

    Without this, any lineage whose aversion overlapped its own signature — every cannibal — would
    read as permanently terrified standing alone in an empty world.
    """

    def test_a_lone_broadcaster_smells_nothing(self):
        _, field = make_field()
        x, y, emission, sig = (
            np.array([10.0]),
            np.array([10.0]),
            np.array([2.0]),
            signature(1, 1, 1),
        )
        field.rebuild(x, y, emission, sig)

        perceived = field.sample_excluding_self(x, y, emission, sig)

        assert perceived[0] == pytest.approx(np.zeros(CHANNELS), abs=1e-6)

    def test_exclusion_is_exact_at_the_map_edge_too(self):
        """The blur is normalized by in-world cells, so a corner cell returns more of its own
        deposit than a central one. A single interior constant would leave residual self-scent
        exactly where prey get cornered.
        """
        _, field = make_field()
        x, y, emission, sig = (
            np.array([0.0]),
            np.array([0.0]),
            np.array([3.0]),
            signature(1, 0, 0),
        )
        field.rebuild(x, y, emission, sig)

        assert field.sample_excluding_self(x, y, emission, sig)[0, 0] == pytest.approx(
            0.0, abs=1e-6
        )

    def test_a_neighbour_is_still_smelled(self):
        """Excluding yourself must not exclude everyone else standing where you are."""
        _, field = make_field()
        x, y = np.full(2, 10.0), np.full(2, 10.0)
        emission = np.array([1.0, 1.0])
        sig = np.tile(signature(1, 0, 0), (2, 1))
        field.rebuild(x, y, emission, sig)

        perceived = field.sample_excluding_self(x, y, emission, sig)

        assert perceived[0, 0] > 0.0

    def test_self_response_is_larger_in_a_corner_than_in_the_middle(self):
        _, field = make_field()

        assert field.self_response[0, 0] > field.self_response[10, 10]


class TestEdges:
    def test_the_map_edge_is_not_spuriously_empty(self):
        """Zero-padding the blur would make borders read as emptier than they are, and prey would
        find the world's edge safe. The reflecting edge errs the other way — a cornered animal is
        slightly more detectable, which is the safer error.
        """
        _, corner = make_field()
        corner.rebuild(np.array([0.0]), np.array([0.0]), np.array([1.0]), signature(1, 0, 0))
        _, middle = make_field()
        middle.rebuild(np.array([10.0]), np.array([10.0]), np.array([1.0]), signature(1, 0, 0))

        at_corner = corner.sample(np.array([0.0]), np.array([0.0]))[0, 0]
        at_middle = middle.sample(np.array([10.0]), np.array([10.0]))[0, 0]

        assert at_corner >= at_middle

    def test_scent_does_not_wrap_around_the_world(self):
        _, field = make_field()
        field.rebuild(np.array([0.0]), np.array([0.0]), np.array([1.0]), signature(1, 0, 0))

        assert field.sample(np.array([20.0]), np.array([20.0]))[0, 0] == pytest.approx(0.0)

    def test_a_position_outside_the_world_is_rejected(self):
        _, field = make_field()
        with pytest.raises(ValueError):
            field.sample(np.array([99.0]), np.array([0.0]))


class TestDiffusionRange:
    def test_a_wider_range_carries_scent_further(self):
        _, narrow = make_field(grid=41, diffusion_range=1.0)
        narrow.rebuild(np.array([20.0]), np.array([20.0]), np.array([1.0]), signature(1, 0, 0))
        _, wide = make_field(grid=41, diffusion_range=6.0)
        wide.rebuild(np.array([20.0]), np.array([20.0]), np.array([1.0]), signature(1, 0, 0))

        probe = (np.array([28.0]), np.array([20.0]))
        assert wide.sample(*probe)[0, 0] > narrow.sample(*probe)[0, 0]


class TestScentBinder:
    """The binder exists so the emission and signature genes are named exactly once."""

    def build(self, grid=21):
        vocabulary = GENE_REGISTRY
        species = SpeciesRegistry(vocabulary)
        store = EntityStore(initial_capacity=8, n_drives=1, n_genes=len(GENE_NAMES))
        genetics = Genetics(store, ColumnRegistry(), species, vocabulary, GENETICS_CONFIG)
        terrain = Terrain(np.zeros((grid, grid), dtype=np.float32), cell_size=1.0)
        field = CueField(terrain, CHANNELS, CueFieldConfig(diffusion_range=2.0))
        scent = Scent(store, genetics, field, vocabulary, SCENT_GENES)
        return store, species, genetics, scent

    def spawn(self, store, species_id, xs, ys):
        ids = store.allocate(
            len(xs),
            species_id=np.full(len(xs), species_id, dtype=np.int32),
            x=np.array(xs, dtype=np.float32),
            y=np.array(ys, dtype=np.float32),
        )
        rows = [store._id_to_row[i] for i in ids.tolist()]
        return Selection.from_indices(np.array(rows, dtype=np.int64), store.capacity)

    def genes_row(self, **values):
        row = np.zeros((1, len(GENE_NAMES)), dtype=np.float32)
        for name, value in values.items():
            row[0, GENE_NAMES.index(name)] = value
        return row

    def test_rejects_signature_genes_that_do_not_match_the_channel_count(self):
        vocabulary = GENE_REGISTRY
        store = EntityStore(initial_capacity=2, n_drives=1, n_genes=len(GENE_NAMES))
        genetics = Genetics(
            store, ColumnRegistry(), SpeciesRegistry(vocabulary), vocabulary, GENETICS_CONFIG
        )
        terrain = Terrain(np.zeros((5, 5), dtype=np.float32), cell_size=1.0)
        field = CueField(terrain, CHANNELS, CueFieldConfig(diffusion_range=1.0))

        with pytest.raises(ValueError):
            Scent(
                store,
                genetics,
                field,
                vocabulary,
                ScentGenes(emission_gene="scent_emission", signature_genes=("signature_0",)),
            )

    def test_a_neighbours_scent_is_perceived(self):
        store, species, genetics, scent = self.build()
        species_id = species.register(GENE_NAMES)
        one = self.spawn(store, species_id, [10.0], [10.0])
        other = self.spawn(store, species_id, [11.0], [10.0])
        genetics.set_genes(one, self.genes_row(scent_emission=1.0, signature_0=1.0))
        genetics.set_genes(other, self.genes_row(scent_emission=1.0, signature_1=1.0))
        scent.rebuild(one | other)

        assert scent.perceive(one)[0, 1] > 0.0

    def test_a_creature_does_not_perceive_its_own_scent(self):
        store, species, genetics, scent = self.build()
        species_id = species.register(GENE_NAMES)
        alone = self.spawn(store, species_id, [10.0], [10.0])
        genetics.set_genes(alone, self.genes_row(scent_emission=5.0, signature_0=2.0))
        scent.rebuild(alone)

        assert scent.perceive(alone)[0] == pytest.approx(np.zeros(CHANNELS), abs=1e-5)

    def test_an_unexpressed_signature_gene_is_not_broadcast(self):
        """Expression, not genotype — the same rule that makes an unexpressed gene cost nothing
        (#17). A species that does not express a scent gene does not smell of it.
        """
        store, species, genetics, scent = self.build()
        odourless = species.register(("scent_emission",))
        smelly = species.register(GENE_NAMES)
        quiet = self.spawn(store, odourless, [10.0], [10.0])
        observer = self.spawn(store, smelly, [11.0], [10.0])
        genetics.set_genes(quiet, self.genes_row(scent_emission=5.0, signature_0=5.0))
        genetics.set_genes(observer, self.genes_row(scent_emission=0.0))
        scent.rebuild(quiet | observer)

        assert scent.perceive(observer)[0] == pytest.approx(np.zeros(CHANNELS), abs=1e-6)
