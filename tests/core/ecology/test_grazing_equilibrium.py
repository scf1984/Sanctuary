"""Grazing tuning, locked in after the fact (CLAUDE.md §8.1).

Ecological tuning cannot be test-driven: there is no failing test to write for "does this intake
rate produce a legible ecology", because the answer is only visible once the world runs. So the
exploration happened first (docs/spikes/grazing-equilibrium.md) and this module pins the *shape* of
what it found, so the tuning cannot silently regress.

Everything here asserts a direction or a distribution, never an exact value — §2.2 rules out
golden-output tests, and a threshold copied from one run of a non-deterministic simulation is a
golden output wearing a statistician's coat.
"""

import numpy as np
import pytest

from core.ecology.carrion import Carrion, CarrionConfig
from core.ecology.diet import Diet, DietConfig
from core.ecology.feeding import Feeding, FeedingConfig
from core.ecology.metabolism import Metabolism, MetabolismConfig
from core.ecology.plants import Plants, PlantsConfig
from core.ecology.service import Ecology
from core.entities.store import EntityStore
from core.genetics.expression import GeneticsConfig
from core.genetics.service import Genetics
from core.genetics.species import SpeciesRegistry
from core.selection import Selection
from core.services import ColumnRegistry
from core.world.climate import Climate, ClimateConfig
from core.world.diffusion import DiffusionConfig
from core.world.terrain import Terrain
from core.world.water import Water

from tests.support.genes import gene_registry

GENE_NAMES = ("size", "insulation", "diet_animal_derived", "mutability")
GENE_REGISTRY = gene_registry(GENE_NAMES, {"size": 0.02, "insulation": 0.01})
GENETICS_CONFIG = GeneticsConfig(mutability_gene="mutability", drift_margin=2.0)

PLANTS_CONFIG = PlantsConfig(
    solar_constant=8.0,
    latitude_tilt=0.0,
    min_growth_temperature=0.0,
    optimal_growth_temperature=22.0,
    max_growth_temperature=45.0,
    nutrient_per_biomass=1.0,
    initial_soil_nutrients=400.0,
    senescence_rate=0.02,
    saturation_accumulation=20.0,
    max_rooting_depth=0.5,
    forage_diffusion=DiffusionConfig(range=4.0, climb_penalty=0.5),
)
METABOLISM_CONFIG = MetabolismConfig(
    basal_rate=0.05,
    thermoregulation_rate=0.01,
    neutral_temperature=20.0,
    insulation_gene="insulation",
)


def grazing_world(allocations, intake_rate=9.0, assimilation_max=0.5, frontier_exponent=2.0):
    """A settled plant field and one stationary grazer per allocation, each on its own cell.

    Stationary on purpose: this module is about the *balance* between what a gut brings in and what
    a body costs to run, and locomotion is a third term that `core.behaviour.movement` already owns
    and tests. Mixing it in would make a regression here unattributable.
    """
    terrain = Terrain(np.zeros((16, 16), dtype=np.float32), cell_size=1.0)
    climate = Climate(
        terrain, ClimateConfig(equator_y=0.0, equator_temperature=20.0, latitude_gradient=0.0)
    )
    plants = Plants(terrain, climate, Water.generate(terrain), PLANTS_CONFIG)
    for _ in range(400):
        plants.grow()
    # Founders are handed 180 energy units each below; the export ledger has to account for
    # bodies the field never supplied before anything can excrete against it (#21).
    plants.record_founding_stock(180.0 * len(allocations))

    n = len(allocations)
    store = EntityStore(initial_capacity=n, n_drives=1, n_genes=len(GENE_NAMES))
    columns = ColumnRegistry()
    species = SpeciesRegistry(GENE_REGISTRY)
    genetics = Genetics(store, columns, species, GENE_REGISTRY, GENETICS_CONFIG)
    ecology = Ecology(
        store,
        columns,
        genetics,
        climate,
        Metabolism(GENE_REGISTRY, METABOLISM_CONFIG),
        plants,
    )
    carrion = Carrion(terrain, plants, CarrionConfig(decay_rate=0.1))
    feeding = Feeding(
        store,
        plants,
        carrion,
        genetics,
        ecology,
        Diet(
            GENE_REGISTRY,
            DietConfig(
                animal_derived_gene="diet_animal_derived", frontier_exponent=frontier_exponent
            ),
        ),
        GENE_REGISTRY,
        FeedingConfig(
            intake_rate=intake_rate, assimilation_max=assimilation_max, size_gene="size"
        ),
    )

    genes = np.zeros((n, len(GENE_NAMES)), dtype=np.float32)
    genes[:, GENE_NAMES.index("size")] = 1.0
    genes[:, GENE_NAMES.index("diet_animal_derived")] = allocations
    ids = store.allocate(
        n,
        # One grazer per cell, so nobody contends and each animal's balance is its own.
        x=np.arange(n, dtype=np.float32) % 16.0,
        y=(np.arange(n, dtype=np.float32) // 16.0),
        energy=np.full(n, 180.0, dtype=np.float32),
        species_id=np.full(n, species.register(GENE_NAMES), dtype=np.int32),
        genes=genes,
    )
    rows = [store._id_to_row[i] for i in np.asarray(ids).tolist()]
    population = Selection.from_indices(np.array(rows, dtype=np.int64), capacity=store.capacity)
    return plants, store, ecology, feeding, population


def live(plants, ecology, feeding, population, ticks):
    """Run the feeding half of a tick: grow, eat, pay upkeep — in TICK_ORDER's relative order."""
    for _ in range(ticks):
        plants.grow()
        feeding.feed(population)
        ecology.drain(population)


class TestTheAllocationHasATeeth:
    """§2.5: every gene needs either an energy cost or a selective consequence. The diet genes cost
    nothing, so the consequence has to be real — otherwise the allocation is a free random walk."""

    def test_a_herbivore_ends_richer_than_a_carnivore_in_a_world_of_grass(self):
        allocations = np.array([-4.0, 4.0], dtype=np.float32)
        plants, store, ecology, feeding, population = grazing_world(allocations)

        live(plants, ecology, feeding, population, ticks=300)

        herbivore, carnivore = ecology.energy(population)
        assert herbivore > carnivore

    def test_energy_falls_monotonically_as_the_allocation_leaves_plants(self):
        allocations = np.linspace(-4.0, 4.0, 16, dtype=np.float32)
        plants, store, ecology, feeding, population = grazing_world(allocations)

        live(plants, ecology, feeding, population, ticks=300)

        final = ecology.energy(population)
        # Rank correlation against the allocation, which is what a direction claim means for a
        # curve that is not linear. -1 is a perfect decreasing relationship.
        order = np.argsort(np.argsort(final))
        expected = np.arange(len(allocations))[::-1]
        assert np.corrcoef(order, expected)[0, 1] > 0.95


class TestANaiveFounderIsViable:
    """The tuning constraint that is specific to this issue, and temporary.

    Nothing dies (#21) and nothing breeds (#20), so **selection cannot move the diet distribution**
    — a founder population keeps whatever allocation it was drawn with, forever. The intake rate
    therefore has to carry animals that are badly allocated by chance, not merely the ones a few
    generations of selection would have produced. Once #20 and #21 land this becomes generous, and
    that is the point at which it should be revisited rather than now.
    """

    def founders(self, seed):
        # The demo world's founding range for this gene, expressed through the logistic squash.
        return np.random.default_rng(seed).uniform(-1.0, 1.0, 64).astype(np.float32)

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_most_of_a_naive_founder_population_holds_its_energy(self, seed):
        plants, store, ecology, feeding, population = grazing_world(self.founders(seed))
        opening = ecology.energy(population).copy()

        live(plants, ecology, feeding, population, ticks=400)

        held = ecology.energy(population) >= opening
        assert held.mean() > 0.5

    def test_a_tick_feeds_before_it_charges(self):
        """§2.1's ordering, tested as the one tick where it is decidable rather than inferred from
        a long run: an animal holding less than one tick's upkeep, standing on food it can use,
        must survive — "died on top of food" reads as a bug even when the arithmetic is right.

        Note this cannot be asserted over many ticks instead, because a stationary grazer strips
        its own cell and then legitimately starves on bare ground. That is grazing pressure, not a
        violation.
        """
        plants, store, ecology, feeding, population = grazing_world(np.array([-4.0]))
        owed = ecology.upkeep(population)[0]
        store.energy[population.to_mask()] = owed * 0.5

        plants.grow()
        feeding.feed(population)
        ecology.drain(population)

        assert ecology.energy(population)[0] > 0.0


class TestTheFieldSurvivesBeingGrazed:
    def test_standing_crop_settles_rather_than_being_stripped_or_running_away(self):
        """A grazed field must reach an equilibrium against regrowth. Stripped to zero means the
        herd outruns the sunlight; unbounded growth means grazing is not touching it."""
        allocations = np.full(64, -4.0, dtype=np.float32)
        plants, store, ecology, feeding, population = grazing_world(allocations)

        live(plants, ecology, feeding, population, ticks=200)
        early = plants.biomass.mean()
        live(plants, ecology, feeding, population, ticks=400)
        late = plants.biomass.mean()

        assert early > 0.0 and late > 0.0
        assert late == pytest.approx(early, rel=0.25)
