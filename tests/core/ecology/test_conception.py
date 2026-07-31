"""Conception and gestation: a young is a row with a negative age (#20).

Test-first (§8.1): who is willing, who is close enough, what a young costs and when it is born are
all checkable in advance. What is *not* here is whether the resulting generation time matches
§2.1's baseline — that is ecological tuning with no failing test to write, and it cannot be measured
while capacity binds (#127).
"""

import numpy as np
import pytest

from core.ecology.conception import Conception, ConceptionConfig
from core.ecology.metabolism import Metabolism, MetabolismConfig
from core.ecology.service import Ecology
from core.entities.store import EntityStore
from core.genetics.expression import GeneticsConfig
from core.genetics.service import Genetics
from core.genetics.species import SpeciesRegistry
from core.selection import Selection
from core.services import ColumnRegistry
from core.world.climate import Climate, ClimateConfig
from core.world.terrain import Terrain

from tests.support.genes import gene_registry
from tests.support.plants import plant_field

GENE_NAMES = ("size", "insulation", "maturity_age", "gestation_length", "mutability")
GENE_REGISTRY = gene_registry(GENE_NAMES, {"insulation": 1.0})
GENETICS_CONFIG = GeneticsConfig(mutability_gene="mutability", drift_margin=2.0)
METABOLISM_CONFIG = MetabolismConfig(
    basal_rate=1.0,
    thermoregulation_rate=0.5,
    neutral_temperature=20.0,
    insulation_gene="insulation",
)


def config(**overrides):
    params = dict(
        contact_range=2.0,
        offspring_energy=40.0,
        maturity_gene="maturity_age",
        gestation_gene="gestation_length",
        speciation_threshold=100.0,
    )
    params.update(overrides)
    return ConceptionConfig(**params)


def make_world(capacity=16, **overrides):
    store = EntityStore(initial_capacity=capacity, n_drives=1, n_genes=len(GENE_NAMES))
    columns = ColumnRegistry()
    species = SpeciesRegistry(GENE_REGISTRY)
    genetics = Genetics(store, columns, species, GENE_REGISTRY, GENETICS_CONFIG)
    terrain = Terrain(np.zeros((21, 21), dtype=np.float32), cell_size=1.0)
    climate = Climate(
        terrain, ClimateConfig(equator_y=0.0, equator_temperature=20.0, latitude_gradient=0.0)
    )
    ecology = Ecology(
        store,
        columns,
        genetics,
        climate,
        Metabolism(GENE_REGISTRY, METABOLISM_CONFIG),
        plant_field(terrain, climate, founding_stock=1000.0 * capacity),
    )
    return (
        store,
        species,
        genetics,
        ecology,
        Conception(store, ecology, genetics, GENE_REGISTRY, config(**overrides)),
    )


def populate(store, species, positions, energy=100.0, age=50, maturity=10.0, gestation=5.0):
    n = len(positions)
    genes = np.zeros((n, len(GENE_NAMES)), dtype=np.float32)
    genes[:, GENE_NAMES.index("maturity_age")] = maturity
    genes[:, GENE_NAMES.index("gestation_length")] = gestation
    genes[:, GENE_NAMES.index("size")] = 1.0
    ids = store.allocate(
        n,
        x=np.array([p[0] for p in positions], dtype=np.float32),
        y=np.array([p[1] for p in positions], dtype=np.float32),
        energy=np.full(n, energy, dtype=np.float32),
        age=np.full(n, age, dtype=np.int64),
        species_id=np.full(n, species.register(GENE_NAMES), dtype=np.int32),
        genes=genes,
    )
    rows = np.array([store._id_to_row[i] for i in ids.tolist()], dtype=np.int64)
    return Selection.from_indices(rows, capacity=store.capacity)


def living(store):
    return Selection.from_mask(store.alive & (store.age >= 0))


def gestating(store):
    return Selection.from_mask(store.alive & (store.age < 0))


class TestConfigValidation:
    @pytest.mark.parametrize(
        "changes, message",
        [
            ({"contact_range": 0.0}, "contact_range"),
            ({"offspring_energy": 0.0}, "offspring_energy"),
            ({"speciation_threshold": 0.0}, "speciation_threshold"),
        ],
    )
    def test_rejects_tuning_that_makes_breeding_impossible_or_free(self, changes, message):
        with pytest.raises(ValueError, match=message):
            config(**changes)


class TestWillingness:
    def test_an_animal_below_its_own_maturity_gene_is_not_willing(self):
        store, species, _, _, conception = make_world()
        population = populate(store, species, [(1.0, 1.0)], age=5, maturity=10.0)

        assert len(conception.willing(population)) == 0

    def test_maturity_is_per_entity_because_it_is_a_gene(self):
        store, species, genetics, _, conception = make_world()
        population = populate(store, species, [(1.0, 1.0), (1.5, 1.5)], age=8)
        genes = genetics.genes(population)
        genes[:, GENE_NAMES.index("maturity_age")] = [5.0, 20.0]
        genetics.set_genes(population, genes)

        assert len(conception.willing(population)) == 1

    def test_an_animal_that_cannot_afford_the_endowment_is_not_willing(self):
        """Not a tuned threshold — you cannot give away what you do not have."""
        store, species, _, _, conception = make_world(offspring_energy=40.0)
        population = populate(store, species, [(1.0, 1.0)], energy=30.0)

        assert len(conception.willing(population)) == 0


class TestConceiving:
    def test_two_touching_animals_make_one_gestating_row(self):
        store, species, _, _, conception = make_world()
        populate(store, species, [(1.0, 1.0), (1.5, 1.0)])

        conception.conceive(living(store), np.random.default_rng(0))

        assert len(gestating(store)) == 1
        assert len(living(store)) == 2

    def test_animals_out_of_contact_do_not_conceive(self):
        store, species, _, _, conception = make_world(contact_range=2.0)
        populate(store, species, [(1.0, 1.0), (15.0, 15.0)])

        conception.conceive(living(store), np.random.default_rng(0))

        assert len(gestating(store)) == 0

    def test_a_lone_animal_cannot_conceive(self):
        store, species, _, _, conception = make_world()
        populate(store, species, [(1.0, 1.0)])

        conception.conceive(living(store), np.random.default_rng(0))

        assert len(gestating(store)) == 0

    def test_nobody_willing_means_nothing_conceived(self):
        store, species, _, _, conception = make_world()
        populate(store, species, [(1.0, 1.0), (1.5, 1.0)], age=1, maturity=10.0)

        conception.conceive(living(store), np.random.default_rng(0))

        assert len(gestating(store)) == 0

    def test_an_empty_world_is_a_no_op(self):
        store, _, _, _, conception = make_world()

        conception.conceive(living(store), np.random.default_rng(0))

        assert store.alive.sum() == 0


class TestTheGestatingYoung:
    def _conceive(self, gestation=5.0, **overrides):
        store, species, genetics, ecology, conception = make_world(**overrides)
        parents = populate(store, species, [(1.0, 1.0), (1.5, 1.0)], gestation=gestation)
        conception.conceive(living(store), np.random.default_rng(0))
        return store, genetics, ecology, parents, gestating(store)

    def test_it_starts_at_a_negative_age(self):
        """The whole mechanic: a young is a row that has not reached zero yet, and `Aging` — which
        already runs every tick and does not care about sign — is the gestation clock."""
        store, _, _, _, young = self._conceive(gestation=5.0)

        assert store.age[young.to_mask()][0] == -5

    def test_the_term_comes_from_the_youngs_own_gene(self):
        """How long a young takes is the young's trait, not either parent's — which is what puts it
        under selection at all."""
        store, _, _, _, young = self._conceive(gestation=12.0)

        assert store.age[young.to_mask()][0] == -12

    def test_it_is_born_where_it_was_conceived(self):
        """A gestating row is excluded from movement, so it stays put while its parents walk on.
        Carrying it would need a link back to a parent — see the module docstring."""
        store, _, _, _, young = self._conceive()

        assert store.x[young.to_mask()][0] == pytest.approx(1.0)

    def test_it_inherits_its_parents_species(self):
        store, _, _, parents, young = self._conceive()

        assert store.species_id[young.to_mask()][0] == store.species_id[parents.to_mask()][0]

    def test_its_genes_are_drawn_from_both_parents(self):
        """The first time `core.genetics.inheritance` is reached by anything that runs."""
        store, genetics, _, parents, young = self._conceive()
        parent_size = genetics.genes(parents)[:, GENE_NAMES.index("size")]
        child_size = genetics.genes(young)[0, GENE_NAMES.index("size")]

        assert abs(child_size - parent_size.mean()) < 1.0

    def test_it_holds_exactly_the_endowment(self):
        store, _, ecology, _, young = self._conceive(offspring_energy=40.0)

        assert ecology.energy(young) == pytest.approx([40.0])


class TestBirth:
    def test_a_young_is_born_when_its_age_reaches_zero(self):
        """Birth is not an event — it is `age >= 0` becoming true, so nothing has to fire."""
        store, species, _, _, conception = make_world()
        populate(store, species, [(1.0, 1.0), (1.5, 1.0)], gestation=3.0)
        conception.conceive(living(store), np.random.default_rng(0))
        assert len(living(store)) == 2

        for expected in (2, 2, 3):
            store.age[store.alive] += 1
            assert len(living(store)) == expected

    def test_the_young_survives_both_its_parents(self):
        """It is attached to nobody, so a pregnancy cannot be aborted by a death — which under any
        design that hung the state off a parent would have needed deliberate work."""
        store, species, _, _, conception = make_world()
        parents = populate(store, species, [(1.0, 1.0), (1.5, 1.0)], gestation=3.0)
        conception.conceive(living(store), np.random.default_rng(0))
        store.release(store.row_ids()[parents.to_mask()])

        store.age[store.alive] += 3

        assert len(living(store)) == 1


class TestTheCostOfBreeding:
    def _conceive(self, parent_energy=100.0, offspring_energy=40.0):
        store, species, _, ecology, conception = make_world(offspring_energy=offspring_energy)
        parents = populate(store, species, [(1.0, 1.0), (1.5, 1.0)], energy=parent_energy)
        conception.conceive(living(store), np.random.default_rng(0))
        return ecology, parents, gestating(store)

    def test_both_parents_pay_half(self):
        ecology, parents, _ = self._conceive(parent_energy=100.0, offspring_energy=40.0)

        assert ecology.energy(parents) == pytest.approx([80.0, 80.0])

    def test_the_population_pool_is_unchanged(self):
        """Gestation moves energy rather than burning it, so what the parents lose is exactly what
        the young holds (#21: `spend` would have excreted it *and* handed it over)."""
        ecology, parents, young = self._conceive()

        assert ecology.energy(parents).sum() + ecology.energy(young).sum() == pytest.approx(200.0)

    def test_conceiving_excretes_no_nutrients(self):
        store, species, _, ecology, conception = make_world()
        populate(store, species, [(1.0, 1.0), (1.5, 1.0)])
        soil_before = ecology.plants.soil_nutrients.copy()
        total_before = ecology.plants.total_nutrients()

        conception.conceive(living(store), np.random.default_rng(0))

        np.testing.assert_array_equal(ecology.plants.soil_nutrients, soil_before)
        assert ecology.plants.total_nutrients() == pytest.approx(total_before, rel=1e-9)


class TestCapacity:
    def test_a_full_store_conceives_nobody_rather_than_raising(self):
        """`EntityStore.grow` may only run at a tick boundary (§2.3) and this is mid-tick, so a
        world short of rows has fewer young. How many to keep spare is #127's."""
        store, species, _, _, conception = make_world(capacity=2)
        populate(store, species, [(1.0, 1.0), (1.5, 1.0)])

        conception.conceive(living(store), np.random.default_rng(0))

        assert store.alive.sum() == 2

    def test_it_conceives_only_as_many_as_there_are_rows(self):
        store, species, _, _, conception = make_world(capacity=5)
        populate(store, species, [(1.0, 1.0), (1.4, 1.0), (8.0, 8.0), (8.4, 8.0)])

        conception.conceive(living(store), np.random.default_rng(0))

        assert store.alive.sum() == 5
        assert store.available == 0


class TestCompatibility:
    def test_two_species_do_not_interbreed(self):
        """`interbreeding_probability` scores a cross-species pair at zero, so isolation is a fact
        of the world once recorded rather than re-derived each tick (#16)."""
        store, species, _, _, conception = make_world()
        genes = np.zeros((2, len(GENE_NAMES)), dtype=np.float32)
        genes[:, GENE_NAMES.index("maturity_age")] = 10.0
        genes[:, GENE_NAMES.index("gestation_length")] = 5.0
        store.allocate(
            2,
            x=np.array([1.0, 1.5], dtype=np.float32),
            y=np.array([1.0, 1.0], dtype=np.float32),
            energy=np.full(2, 100.0, dtype=np.float32),
            age=np.full(2, 50, dtype=np.int64),
            species_id=np.array(
                [species.register(GENE_NAMES), species.register(GENE_NAMES)], dtype=np.int32
            ),
            genes=genes,
        )

        conception.conceive(living(store), np.random.default_rng(0))

        assert len(gestating(store)) == 0
