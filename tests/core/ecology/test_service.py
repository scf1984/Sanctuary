import numpy as np
import pytest

from core.ecology.metabolism import Metabolism, MetabolismConfig
from core.ecology.service import Ecology
from core.entities.store import EntityStore
from core.genetics.service import Genetics
from core.genetics.species import SpeciesRegistry
from core.genetics.vocabulary import GeneVocabulary
from core.selection import Selection
from core.services import ColumnOwnershipError, ColumnRegistry
from core.world.climate import Climate, ClimateConfig
from core.world.terrain import Terrain


GENE_NAMES = ("size", "speed", "sight", "insulation")

METABOLISM_CONFIG = MetabolismConfig(
    gene_costs={"size": 2.0, "speed": 3.0, "sight": 0.0, "insulation": 1.0},
    basal_rate=1.0,
    thermoregulation_rate=0.5,
    neutral_temperature=20.0,
    insulation_gene="insulation",
)


def make_world(initial_capacity=8, temperature=20.0):
    """A store, its services, and a flat world whose temperature is uniform by construction.

    A constant elevation with the equator on the world's only row makes `temperature_at` return
    `temperature` everywhere, so a test that is not about climate never has to care where it put
    its entities.
    """
    store = EntityStore(initial_capacity=initial_capacity, n_drives=1, n_genes=len(GENE_NAMES))
    registry = ColumnRegistry()
    species = SpeciesRegistry(GeneVocabulary(GENE_NAMES))
    genetics = Genetics(store, registry, species)
    terrain = Terrain(np.zeros((4, 4), dtype=np.float32), cell_size=10.0)
    climate = Climate(
        terrain,
        ClimateConfig(equator_y=0.0, equator_temperature=temperature, latitude_gradient=0.0),
    )
    metabolism = Metabolism(GeneVocabulary(GENE_NAMES), METABOLISM_CONFIG)
    ecology = Ecology(store, registry, genetics, climate, metabolism)
    return store, registry, species, genetics, ecology


def selection_for(store, ids):
    rows = [store._id_to_row[i] for i in np.asarray(ids).tolist()]
    return Selection.from_indices(np.array(rows, dtype=np.int64), capacity=store.capacity)


def gene_row(**genes):
    row = np.zeros(len(GENE_NAMES), dtype=np.float32)
    for name, value in genes.items():
        row[GENE_NAMES.index(name)] = value
    return row


class TestColumnOwnership:
    def test_claims_the_energy_column(self):
        _, registry, *_ = make_world()
        assert registry.owner_of("energy") == "Ecology"

    def test_a_rival_service_cannot_also_claim_energy(self):
        store, registry, _, genetics, ecology = make_world()

        class RivalEcology(Ecology):
            pass

        with pytest.raises(ColumnOwnershipError):
            RivalEcology(store, registry, genetics, ecology.climate, ecology.metabolism)

    def test_ecology_cannot_write_a_column_it_does_not_own(self):
        store, _, _, _, ecology = make_world()
        ids = store.allocate(1)
        selection = selection_for(store, ids)

        with pytest.raises(ColumnOwnershipError):
            ecology.write("genes", selection, np.zeros((1, len(GENE_NAMES)), dtype=np.float32))


class TestUpkeepFromExpressedGenes:
    def test_an_unexpressed_gene_costs_nothing(self):
        """A gene's cost and its benefit are inseparable (issue #17): a species that does not
        express a trait does not pay for it, and one that does pays every tick it lives.
        """
        store, _, species, genetics, ecology = make_world()
        sprinter = species.register(("size", "speed", "sight", "insulation"))
        plodder = species.register(("size", "sight", "insulation"))
        ids = store.allocate(2, species_id=np.array([sprinter, plodder], dtype=np.int32))
        selection = selection_for(store, ids)
        # Identical genotypes -- only the expression masks differ.
        genetics.set_genes(selection, np.stack([gene_row(speed=4.0), gene_row(speed=4.0)]))

        upkeep = ecology.upkeep(selection)

        assert upkeep[0] == pytest.approx(1.0 + 3.0 * 4.0)
        assert upkeep[1] == pytest.approx(1.0)

    def test_upkeep_reads_the_temperature_at_each_entitys_own_position(self):
        """Thermoregulation is a per-entity cost sampled from the climate field, so two members
        of one species in different places pay differently for identical genes.
        """
        store = EntityStore(initial_capacity=4, n_drives=1, n_genes=len(GENE_NAMES))
        registry = ColumnRegistry()
        species = SpeciesRegistry(GeneVocabulary(GENE_NAMES))
        genetics = Genetics(store, registry, species)
        # A north-south gradient: 30 degC at y=0, falling 1 degC per world unit.
        terrain = Terrain(np.zeros((21, 21), dtype=np.float32), cell_size=1.0)
        climate = Climate(
            terrain,
            ClimateConfig(equator_y=0.0, equator_temperature=30.0, latitude_gradient=1.0),
        )
        ecology = Ecology(
            store,
            registry,
            genetics,
            climate,
            Metabolism(GeneVocabulary(GENE_NAMES), METABOLISM_CONFIG),
        )
        species_id = species.register(GENE_NAMES)
        ids = store.allocate(
            2,
            x=np.array([0.0, 0.0], dtype=np.float32),
            y=np.array([10.0, 20.0], dtype=np.float32),
            species_id=np.full(2, species_id, dtype=np.int32),
        )
        selection = selection_for(store, ids)

        upkeep = ecology.upkeep(selection)

        # y=10 sits at 20 degC (neutral, no thermal cost); y=20 sits at 10 degC, ten degrees
        # below neutral, so it pays 0.5 J/tick per degree on top of the same basal rate.
        assert upkeep[0] == pytest.approx(1.0)
        assert upkeep[1] == pytest.approx(1.0 + 0.5 * 10.0)


class TestDrain:
    def test_drain_subtracts_exactly_one_tick_of_upkeep(self):
        store, _, species, genetics, ecology = make_world()
        species_id = species.register(GENE_NAMES)
        ids = store.allocate(
            2,
            energy=np.array([100.0, 100.0], dtype=np.float32),
            species_id=np.full(2, species_id, dtype=np.int32),
        )
        selection = selection_for(store, ids)
        genetics.set_genes(selection, np.stack([gene_row(speed=1.0), gene_row(speed=2.0)]))

        upkeep = ecology.upkeep(selection)
        ecology.drain(selection)

        assert ecology.energy(selection) == pytest.approx(100.0 - upkeep)

    def test_drain_leaves_entities_outside_the_selection_untouched(self):
        store, _, species, _, ecology = make_world()
        species_id = species.register(GENE_NAMES)
        ids = store.allocate(
            2,
            energy=np.array([100.0, 100.0], dtype=np.float32),
            species_id=np.full(2, species_id, dtype=np.int32),
        )
        drained = selection_for(store, ids[:1])
        untouched = selection_for(store, ids[1:])

        ecology.drain(drained)

        assert ecology.energy(untouched) == pytest.approx([100.0])

    def test_drain_floors_at_zero_and_never_goes_negative(self):
        """CLAUDE.md §2.5: the metabolic pool is a hard budget. Upkeep can empty it, never
        overdraw it -- the invariant registered by #7 asserts exactly this every tick.
        """
        store, _, species, _, ecology = make_world()
        species_id = species.register(GENE_NAMES)
        ids = store.allocate(
            1,
            energy=np.array([0.5], dtype=np.float32),
            species_id=np.array([species_id], dtype=np.int32),
        )
        selection = selection_for(store, ids)

        ecology.drain(selection)

        assert ecology.energy(selection) == pytest.approx([0.0])

    def test_drain_never_raises_an_entitys_energy(self):
        store, _, species, genetics, ecology = make_world()
        species_id = species.register(GENE_NAMES)
        rng = np.random.default_rng(3)
        ids = store.allocate(
            8,
            energy=rng.uniform(0.0, 50.0, size=8).astype(np.float32),
            species_id=np.full(8, species_id, dtype=np.int32),
        )
        selection = selection_for(store, ids)
        genetics.set_genes(
            selection, rng.uniform(0.0, 5.0, size=(8, len(GENE_NAMES))).astype(np.float32)
        )

        before = ecology.energy(selection).copy()
        ecology.drain(selection)

        # Energy is never created here (issue #17): plant growth (#18) and feeding (#19) own
        # every path that adds to the pool.
        assert (ecology.energy(selection) <= before).all()


class TestSpend:
    """Upkeep is not the only draw on the pool: locomotion charges through here too (#25), because
    this service owns `energy` and a mover cannot subtract from the column itself.
    """

    def _one_entity(self, energy):
        store, _, species, _, ecology = make_world()
        species_id = species.register(GENE_NAMES)
        ids = store.allocate(
            1,
            energy=np.array([energy], dtype=np.float32),
            species_id=np.array([species_id], dtype=np.int32),
        )
        return ecology, selection_for(store, ids)

    def test_it_subtracts_the_charge(self):
        ecology, selection = self._one_entity(100.0)

        ecology.spend(selection, np.array([30.0], dtype=np.float32))

        assert ecology.energy(selection) == pytest.approx([70.0])

    def test_it_floors_at_zero_rather_than_running_a_debt(self):
        """The same hard-budget rule `drain` obeys, asserted on the general charge because that is
        now where the floor lives (CLAUDE.md §2.5).
        """
        ecology, selection = self._one_entity(10.0)

        ecology.spend(selection, np.array([25.0], dtype=np.float32))

        assert ecology.energy(selection) == pytest.approx([0.0])

    def test_a_negative_charge_is_rejected_rather_than_becoming_income(self):
        """Energy enters the world as sunlight (#18) and through feeding (#19). A cost with its
        sign flipped would be a third, unaudited income and would break §2.5's closed loop with
        nothing to flag it (§8.7).
        """
        ecology, selection = self._one_entity(100.0)

        with pytest.raises(ValueError, match="cannot be negative"):
            ecology.spend(selection, np.array([-5.0], dtype=np.float32))

        assert ecology.energy(selection) == pytest.approx([100.0])


class TestStarving:
    def test_selects_the_entities_whose_pool_has_run_out(self):
        store, _, species, _, ecology = make_world()
        species_id = species.register(GENE_NAMES)
        ids = store.allocate(
            3,
            energy=np.array([0.0, 10.0, 0.0], dtype=np.float32),
            species_id=np.full(3, species_id, dtype=np.int32),
        )
        alive = selection_for(store, ids)

        starving = ecology.starving(alive)

        assert starving == selection_for(store, [ids[0], ids[2]])

    def test_ignores_free_rows_that_merely_hold_zero_energy(self):
        """A released row's energy column still reads 0, and #21 will turn this selection into
        deaths -- so a free row must never appear in it.
        """
        store, _, species, _, ecology = make_world()
        species_id = species.register(GENE_NAMES)
        ids = store.allocate(
            2,
            energy=np.array([10.0, 10.0], dtype=np.float32),
            species_id=np.full(2, species_id, dtype=np.int32),
        )
        store.release(ids[:1])

        starving = ecology.starving(Selection.all(store.capacity))

        assert len(starving) == 0

    def test_an_entity_drained_to_empty_becomes_starving(self):
        store, _, species, _, ecology = make_world()
        species_id = species.register(GENE_NAMES)
        ids = store.allocate(
            1,
            energy=np.array([0.5], dtype=np.float32),
            species_id=np.array([species_id], dtype=np.int32),
        )
        selection = selection_for(store, ids)
        assert len(ecology.starving(selection)) == 0

        ecology.drain(selection)

        assert ecology.starving(selection) == selection
