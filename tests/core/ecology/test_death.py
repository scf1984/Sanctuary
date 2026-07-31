"""Death: an emptied pool frees a row (#21, CLAUDE.md §2.5).

Test-first (§8.1): what death does to the store is exactly checkable in advance — which rows are
freed, which survive, and that a freed row can be handed out again.

There is no carcass here, and that is a consequence rather than an omission. An animal's nutrient
debt is exactly its energy — founding puts `E₀` on the export ledger, feeding adds what it
assimilated, and every `spend` removes what it burned — and `Ecology.starving` is `energy <= 0`. So
an animal that starves owes nothing and leaves nothing: it has metabolised its own body. Carrion
needs a death that is not starvation, and a body distinct from its fuel, which is #20's gestation.
"""

import numpy as np
import pytest

from core.ecology.death import Death
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

GENE_NAMES = ("size", "insulation", "mutability")
GENE_REGISTRY = gene_registry(GENE_NAMES, {"insulation": 1.0})
GENETICS_CONFIG = GeneticsConfig(mutability_gene="mutability", drift_margin=2.0)
METABOLISM_CONFIG = MetabolismConfig(
    basal_rate=1.0,
    thermoregulation_rate=0.5,
    neutral_temperature=20.0,
    insulation_gene="insulation",
)


def make_world(capacity=8):
    store = EntityStore(initial_capacity=capacity, n_drives=1, n_genes=len(GENE_NAMES))
    columns = ColumnRegistry()
    species = SpeciesRegistry(GENE_REGISTRY)
    genetics = Genetics(store, columns, species, GENE_REGISTRY, GENETICS_CONFIG)
    terrain = Terrain(np.zeros((11, 11), dtype=np.float32), cell_size=1.0)
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
    return store, species, ecology, Death(store, ecology)


def populate(store, species, energies):
    n = len(energies)
    ids = store.allocate(
        n,
        x=np.full(n, 5.0, dtype=np.float32),
        y=np.full(n, 5.0, dtype=np.float32),
        energy=np.array(energies, dtype=np.float32),
        species_id=np.full(n, species.register(GENE_NAMES), dtype=np.int32),
    )
    rows = [store._id_to_row[i] for i in ids.tolist()]
    return ids, Selection.from_indices(np.array(rows, dtype=np.int64), capacity=store.capacity)


class TestReaping:
    def test_an_emptied_animal_stops_being_alive(self):
        store, species, _, death = make_world()
        _, population = populate(store, species, [0.0, 50.0])

        death.reap(population)

        assert store.alive.sum() == 1

    def test_the_survivor_is_the_one_with_energy(self):
        store, species, ecology, death = make_world()
        ids, population = populate(store, species, [0.0, 50.0])

        death.reap(population)

        survivors = Selection.from_mask(store.alive)
        assert ecology.energy(survivors) == pytest.approx([50.0])

    def test_a_freed_row_returns_to_the_free_list(self):
        store, species, _, death = make_world()
        _, population = populate(store, species, [0.0, 50.0])
        free_before = store.available

        death.reap(population)

        assert store.available == free_before + 1

    def test_a_freed_row_can_be_handed_to_somebody_else(self):
        """§2.1 runs death before reproduction precisely so this is true within one tick: a world
        at capacity can still breed because the dead have already made room."""
        store, species, _, death = make_world(capacity=2)
        _, population = populate(store, species, [0.0, 50.0])

        death.reap(population)
        reused = store.allocate(1, energy=np.array([10.0], dtype=np.float32))

        assert store.alive.sum() == 2
        assert reused.shape == (1,)

    def test_a_dead_row_keeps_no_claim_on_its_old_id(self):
        """Ids are never reused, so the renderer can tell a row that changed hands from one that
        did not (#119). Death is the event that makes that distinction real."""
        store, species, _, death = make_world()
        ids, population = populate(store, species, [0.0, 50.0])

        death.reap(population)

        assert ids[0] not in store.row_ids()

    def test_nothing_starving_means_nothing_dies(self):
        store, species, _, death = make_world()
        _, population = populate(store, species, [30.0, 50.0])

        death.reap(population)

        assert store.alive.sum() == 2

    def test_an_empty_selection_is_a_no_op(self):
        store, species, _, death = make_world()
        populate(store, species, [0.0, 50.0])

        death.reap(Selection.from_mask(np.zeros(store.capacity, dtype=bool)))

        assert store.alive.sum() == 2

    def test_it_only_reaps_inside_the_selection(self):
        """`reap` takes the caller's choice of who is subject to death, exactly as `drain` takes
        the caller's choice of who metabolises."""
        store, species, _, death = make_world()
        ids, _ = populate(store, species, [0.0, 0.0])
        first = Selection.from_indices(
            np.array([store._id_to_row[ids[0].item()]], dtype=np.int64), capacity=store.capacity
        )

        death.reap(first)

        assert store.alive.sum() == 1

    def test_everything_can_die_at_once(self):
        store, species, _, death = make_world()
        _, population = populate(store, species, [0.0, 0.0, 0.0])

        death.reap(population)

        assert store.alive.sum() == 0
        assert store.available == store.capacity


class TestConservation:
    def test_dying_neither_creates_nor_destroys_nutrients(self):
        """A starved animal owes nothing, so its death moves nothing. The property still needs
        pinning: if death ever *did* deposit a carcass it would have to come off the ledger, and
        this is the test that would notice it being invented instead."""
        store, species, ecology, death = make_world()
        _, population = populate(store, species, [0.0, 50.0])
        opening = ecology.plants.total_nutrients()

        death.reap(population)

        assert ecology.plants.total_nutrients() == pytest.approx(opening, rel=1e-9)
