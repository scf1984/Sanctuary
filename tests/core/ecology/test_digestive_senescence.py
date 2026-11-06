"""An old gut extracts less: digestion yield decays with age on the senescence curve (#244).

Performance-trait senescence (#239) kills an old animal only if its intake depends on a decaying
trait. One that came to rest on a cell with food at hand needs no speed and no acuity to keep
eating, so it lived forever. The fix keeps §2.5's rule that death falls out of starvation rather than
an age check: what an animal *assimilates* from each mouthful falls with the same `decay` factor,
so the gain eventually drops under an upkeep floor that does not decay (basal rate, capacity
traits), and `Ecology.starving` does the rest.

Only the yield decays, not the mouthful. Scaling both would make the realised intake `decay²`: two
coefficients for one decline, the compounding `core.ecology.feeding` already refuses for diet
efficiency. What an old gut cannot use is faeces, so conservation is untouched.
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
from core.genetics.senescence import SenescenceConfig, decay
from core.genetics.service import Genetics
from core.genetics.species import SpeciesRegistry
from core.selection import Selection
from core.services import ColumnRegistry
from core.world.climate import Climate, ClimateConfig
from core.world.diffusion import DiffusionConfig
from core.world.terrain import Terrain
from core.world.water import Water

from tests.support.genes import gene_registry

GENE_NAMES = (
    "size",
    "heat_tolerance",
    "cold_tolerance",
    "diet_animal_derived",
    "senescence_resistance",
    "speed",
    "mutability",
)
# Tolerances are costed only because `Metabolism` refuses a free one; the world sits at the neutral
# temperature, so they never bite.
GENE_REGISTRY = gene_registry(GENE_NAMES, {"heat_tolerance": 1.0, "cold_tolerance": 1.0})

BASE_DECAY = 0.01
SENESCENCE = SenescenceConfig(
    resistance_gene="senescence_resistance",
    base_decay=BASE_DECAY,
    decaying_genes=("speed",),
)

PLANTS_CONFIG = PlantsConfig(
    solar_constant=10.0,
    latitude_tilt=0.0,
    min_growth_temperature=0.0,
    optimal_growth_temperature=25.0,
    max_growth_temperature=45.0,
    nutrient_per_biomass=0.1,
    initial_soil_nutrients=100.0,
    senescence_rate=0.05,
    saturation_accumulation=50.0,
    max_rooting_depth=0.5,
    forage_diffusion=DiffusionConfig(range=4.0, climb_penalty=0.5),
)

METABOLISM_CONFIG = MetabolismConfig(
    dehydration_penalty=0.0,
    basal_rate=0.2,
    thermoregulation_rate=0.5,
    neutral_temperature=25.0,
    heat_tolerance_gene="heat_tolerance",
    cold_tolerance_gene="cold_tolerance",
)

ASSIMILATION_MAX = 0.6
# The plant allocation saturates to a specialist at -20, so its diet efficiency is ~1 and the
# assimilated share is the ceiling times the age decay alone.
PLANT_SPECIALIST = -20.0


def make_world(senescence=SENESCENCE, intake_rate=1.0):
    """A flat, uniformly neutral 11x11 world with a settled plant field and nothing alive yet."""
    terrain = Terrain(np.zeros((11, 11), dtype=np.float32), cell_size=1.0)
    climate = Climate(
        terrain,
        ClimateConfig(equator_y=0.0, equator_temperature=25.0, latitude_gradient=0.0),
    )
    plants = Plants(terrain, climate, Water.generate(terrain), PLANTS_CONFIG)
    for _ in range(80):
        plants.grow()
    # Bodies are handed energy directly here, so the ledger must already account for it (#21).
    plants.record_founding_stock(10_000.0)
    store = EntityStore(initial_capacity=8, n_drives=1, n_genes=len(GENE_NAMES))
    columns = ColumnRegistry()
    species = SpeciesRegistry(GENE_REGISTRY)
    genetics = Genetics(
        store,
        columns,
        species,
        GENE_REGISTRY,
        GeneticsConfig(mutability_gene="mutability", drift_margin=2.0, senescence=senescence),
    )
    ecology = Ecology(
        store, columns, genetics, climate, Metabolism(GENE_REGISTRY, METABOLISM_CONFIG), plants
    )
    diet = Diet(
        GENE_REGISTRY, DietConfig(animal_derived_gene="diet_animal_derived", frontier_exponent=2.0)
    )
    feeding = Feeding(
        store,
        plants,
        Carrion(terrain, plants, CarrionConfig(decay_rate=0.1)),
        genetics,
        ecology,
        diet,
        GENE_REGISTRY,
        FeedingConfig(intake_rate=intake_rate, assimilation_max=ASSIMILATION_MAX, size_gene="size"),
    )
    return store, species, plants, ecology, feeding


def animals_at(store, species, positions, ages, resistance=0.0, energy=100.0):
    """Identical plant specialists at `positions`, differing only in age (and optionally resistance)."""
    n = len(positions)
    species_id = species.register(GENE_NAMES)
    genes = np.zeros((n, len(GENE_NAMES)), dtype=np.float32)
    genes[:, GENE_NAMES.index("size")] = 1.0
    genes[:, GENE_NAMES.index("diet_animal_derived")] = PLANT_SPECIALIST
    genes[:, GENE_NAMES.index("senescence_resistance")] = resistance
    ids = store.allocate(
        n,
        x=np.array([p[0] for p in positions], dtype=np.float32),
        y=np.array([p[1] for p in positions], dtype=np.float32),
        energy=np.full(n, energy),
        age=np.asarray(ages, dtype=np.int64),
        species_id=np.full(n, species_id, dtype=np.int32),
        genes=genes,
    )
    rows = np.array([store._id_to_row[i] for i in np.asarray(ids).tolist()], dtype=np.int64)
    return rows


def one(store, row):
    return Selection.from_indices(np.array([row], dtype=np.int64), capacity=store.capacity)


class TestAnOldGutExtractsLess:
    def test_an_old_animal_gains_the_young_ones_gain_times_its_decay(self):
        store, species, _, ecology, feeding = make_world()
        young, old = animals_at(store, species, [(2.0, 2.0), (8.0, 8.0)], ages=[0, 100])

        feeding.feed(Selection.from_indices(np.array([young, old]), capacity=store.capacity))

        gained_young = store.energy[young] - 100.0
        gained_old = store.energy[old] - 100.0
        assert gained_young > 0.0
        expected = float(decay(np.array([100]), np.array([0.0]), BASE_DECAY)[0])
        assert gained_old / gained_young == pytest.approx(expected, rel=1e-5)

    def test_the_mouthful_itself_does_not_shrink(self):
        """Only the yield decays. Shrinking the mouthful too would compound to decay², the same
        double-counting `Feeding` refuses for diet efficiency."""
        store, species, plants, _, feeding = make_world()
        young, old = animals_at(store, species, [(2.0, 2.0), (8.0, 8.0)], ages=[0, 300])
        before_young, before_old = plants.biomass[2, 2], plants.biomass[8, 8]

        feeding.feed(Selection.from_indices(np.array([young, old]), capacity=store.capacity))

        taken_young = before_young - plants.biomass[2, 2]
        taken_old = before_old - plants.biomass[8, 8]
        assert taken_old == pytest.approx(taken_young, rel=1e-6)

    def test_what_an_old_gut_cannot_use_returns_to_the_soil(self):
        store, species, plants, ecology, feeding = make_world()
        (old,) = animals_at(store, species, [(5.0, 5.0)], ages=[300])
        total_before = plants.total_nutrients()

        feeding.feed(one(store, old))

        assert plants.total_nutrients() == pytest.approx(total_before, rel=1e-9)

    def test_resistance_keeps_more_of_the_gut_at_the_same_age(self):
        store, species, _, _, feeding = make_world()
        (frail,) = animals_at(store, species, [(2.0, 2.0)], ages=[200], resistance=0.0)
        (sturdy,) = animals_at(store, species, [(8.0, 8.0)], ages=[200], resistance=3.0)

        feeding.feed(Selection.from_indices(np.array([frail, sturdy]), capacity=store.capacity))

        assert store.energy[sturdy] > store.energy[frail]

    def test_a_world_that_does_not_age_feeds_every_age_alike(self):
        """Service-level fixtures opt out of senescence (an assembled world cannot, #239). Opting
        out must mean no age effect at all, matching `Genetics.expressed`."""
        store, species, _, _, feeding = make_world(senescence=None)
        young, old = animals_at(store, species, [(2.0, 2.0), (8.0, 8.0)], ages=[0, 5_000])

        feeding.feed(Selection.from_indices(np.array([young, old]), capacity=store.capacity))

        assert store.energy[old] == pytest.approx(store.energy[young], rel=1e-9)


def live_until_starved(senescence, resistance=0.0, horizon=3_000):
    """Run one motionless animal on a regrowing cell: grow, age, eat, pay upkeep. Returns the tick it
    starved on, or None if it outlived the horizon. There is no age check anywhere in the loop;
    starvation is read from the energy pool exactly as `Death` reads it."""
    store, species, plants, ecology, feeding = make_world(senescence=senescence)
    (row,) = animals_at(store, species, [(5.0, 5.0)], ages=[0], resistance=resistance, energy=50.0)
    animal = one(store, row)
    for tick in range(horizon):
        plants.grow()
        store.age[row] += 1
        feeding.feed(animal)
        ecology.drain(animal)
        if len(ecology.starving(animal)):
            return tick
    return None


class TestAMotionlessOldAnimalStarvesOnAFullCell:
    def test_without_ageing_it_never_starves(self):
        """The control, and the defect #244 reported: food at hand, nothing decaying, no end."""
        assert live_until_starved(senescence=None) is None

    def test_with_ageing_it_starves_without_any_age_check(self):
        assert live_until_starved(senescence=SENESCENCE) is not None

    def test_a_resistant_animal_starves_later(self):
        frail = live_until_starved(senescence=SENESCENCE, resistance=0.0)
        sturdy = live_until_starved(senescence=SENESCENCE, resistance=1.0)
        assert frail is not None and sturdy is not None
        assert sturdy > frail
