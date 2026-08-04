"""Feeding: the first energy transfer between trophic levels (#19, CLAUDE.md §2.5).

Test-first (§8.1): the arithmetic of a transfer is checkable in advance, and the two properties
that matter are conservation laws rather than behaviour — energy gained can never exceed the energy
content of what was eaten (§6), and nutrients are neither created nor destroyed by the passage
through an animal.

What is *not* here is the tuning: whether the intake rate produces §2.1's ~10² feeding events per
lifetime is an ecological question with no failing test to write first, so it is explored and then
locked in by `tests/core/ecology/test_grazing_equilibrium.py`.
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
# `insulation` is here only because `Metabolism` requires a costed one — a gene that reduces
# upkeep and charges nothing is a free lunch (§2.5). These tests keep the world at the neutral
# temperature so it never actually bites.
GENE_REGISTRY = gene_registry(GENE_NAMES, {"insulation": 1.0})
GENETICS_CONFIG = GeneticsConfig(mutability_gene="mutability", drift_margin=2.0)

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
    basal_rate=1.0,
    thermoregulation_rate=0.5,
    neutral_temperature=25.0,
    insulation_gene="insulation",
)


def make_world(intake_rate=1.0, assimilation_max=0.6, frontier_exponent=2.0, settle_ticks=80):
    """A flat, uniformly warm 11x11 world with a settled plant field and nothing alive in it yet."""
    terrain = Terrain(np.zeros((11, 11), dtype=np.float32), cell_size=1.0)
    climate = Climate(
        terrain,
        ClimateConfig(equator_y=0.0, equator_temperature=25.0, latitude_gradient=0.0),
    )
    plants = Plants(terrain, climate, Water.generate(terrain), PLANTS_CONFIG)
    for _ in range(settle_ticks):
        plants.grow()
    # These tests hand animals energy directly rather than letting them graze for it, so the
    # ledger has to already account for bodies the field never supplied (#21).
    plants.record_founding_stock(10_000.0)

    store = EntityStore(initial_capacity=8, n_drives=1, n_genes=len(GENE_NAMES))
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
    diet = Diet(
        GENE_REGISTRY,
        DietConfig(
            animal_derived_gene="diet_animal_derived", frontier_exponent=frontier_exponent
        ),
    )
    carrion = Carrion(terrain, plants, CarrionConfig(decay_rate=0.1))
    feeding = Feeding(
        store,
        plants,
        carrion,
        genetics,
        ecology,
        diet,
        GENE_REGISTRY,
        FeedingConfig(
            intake_rate=intake_rate, assimilation_max=assimilation_max, size_gene="size"
        ),
    )
    return store, species, plants, ecology, feeding


def graze_at(store, species, positions, size=1.0, animal_derived=0.0, energy=100.0):
    """Allocate one entity per (x, y), with the given body size and diet allocation."""
    n = len(positions)
    species_id = species.register(GENE_NAMES)
    genes = np.zeros((n, len(GENE_NAMES)), dtype=np.float32)
    genes[:, GENE_NAMES.index("size")] = size
    # The diet column is stored raw and read through the logistic squash, so the stored value that
    # expresses to `animal_derived` is its inverse. -20 and +20 saturate to a pure specialist.
    genes[:, GENE_NAMES.index("diet_animal_derived")] = animal_derived
    ids = store.allocate(
        n,
        x=np.array([p[0] for p in positions], dtype=np.float32),
        y=np.array([p[1] for p in positions], dtype=np.float32),
        energy=np.full(n, energy, dtype=np.float32),
        species_id=np.full(n, species_id, dtype=np.int32),
        genes=genes,
    )
    rows = [store._id_to_row[i] for i in np.asarray(ids).tolist()]
    return Selection.from_indices(np.array(rows, dtype=np.int64), capacity=store.capacity)


class TestConfigValidation:
    @pytest.mark.parametrize("rate", [0.0, -1.0])
    def test_rejects_a_non_positive_intake_rate(self, rate):
        with pytest.raises(ValueError, match="intake_rate"):
            FeedingConfig(intake_rate=rate, assimilation_max=0.5, size_gene="size")

    @pytest.mark.parametrize("assimilation", [0.0, -0.1, 1.5])
    def test_rejects_an_assimilation_ceiling_outside_zero_to_one(self, assimilation):
        """Above 1 is energy created out of grass, which §6 forbids outright; at or below 0 nothing
        can ever eat and the world is dead on arrival rather than visibly misconfigured."""
        with pytest.raises(ValueError, match="assimilation_max"):
            FeedingConfig(intake_rate=1.0, assimilation_max=assimilation, size_gene="size")


class TestGrazing:
    def test_a_grazer_gains_energy_and_the_cell_loses_biomass(self):
        store, species, plants, ecology, feeding = make_world()
        grazers = graze_at(store, species, [(5.0, 5.0)])
        standing = plants.biomass[5, 5]

        feeding.feed(grazers)

        assert plants.biomass[5, 5] < standing
        assert ecology.energy(grazers)[0] > 100.0

    def test_gained_energy_is_the_harvest_times_the_conversion(self):
        store, species, plants, ecology, feeding = make_world(
            intake_rate=2.0, assimilation_max=0.6
        )
        grazers = graze_at(store, species, [(5.0, 5.0)], size=1.5, animal_derived=-20.0)
        standing = plants.biomass[5, 5]

        feeding.feed(grazers)

        # size 1.5 x intake 2.0 = 3.0 demanded; the cell holds more than that, so the harvest is
        # the full demand. A saturated plant allocation converts at the assimilation ceiling.
        harvested = standing - plants.biomass[5, 5]
        assert harvested == pytest.approx(3.0)
        assert ecology.energy(grazers)[0] - 100.0 == pytest.approx(3.0 * 0.6, rel=1e-5)

    def test_a_bigger_body_takes_a_bigger_mouthful(self):
        store, species, plants, _, feeding = make_world()
        small = graze_at(store, species, [(2.0, 2.0)], size=1.0)
        large = graze_at(store, species, [(8.0, 8.0)], size=3.0)
        standing_small = plants.biomass[2, 2]
        standing_large = plants.biomass[8, 8]

        feeding.feed(Selection.from_mask(small.to_mask() | large.to_mask()))

        assert standing_large - plants.biomass[8, 8] > standing_small - plants.biomass[2, 2]

    def test_bare_ground_yields_nothing_and_is_not_an_error(self):
        store, species, plants, ecology, feeding = make_world()
        plants.biomass[5, 5] = 0.0
        grazers = graze_at(store, species, [(5.0, 5.0)])

        feeding.feed(grazers)

        assert ecology.energy(grazers)[0] == pytest.approx(100.0)

    def test_an_empty_selection_is_a_no_op(self):
        store, species, plants, _, feeding = make_world()
        opening = plants.total_nutrients()

        feeding.feed(Selection.from_mask(np.zeros(store.capacity, dtype=bool)))

        assert plants.total_nutrients() == pytest.approx(opening, rel=1e-9)


class TestTheYieldCeiling:
    """§6: energy is never created. A creature cannot gain more than was invested in building what
    it ate, so the conversion is bounded by 1 before the assimilation ceiling even applies."""

    def test_gain_never_exceeds_the_energy_content_of_the_harvest(self):
        store, species, plants, ecology, feeding = make_world(assimilation_max=1.0)
        grazers = graze_at(
            store, species, [(x, 5.0) for x in (1.0, 3.0, 5.0, 7.0, 9.0)], animal_derived=-20.0
        )
        standing = plants.biomass.sum()
        opening = ecology.energy(grazers).sum()

        feeding.feed(grazers)

        harvested = standing - plants.biomass.sum()
        assert ecology.energy(grazers).sum() - opening <= harvested + 1e-4

    def test_a_pure_carnivore_gains_nothing_from_grass(self):
        store, species, _, ecology, feeding = make_world()
        carnivores = graze_at(store, species, [(5.0, 5.0)], animal_derived=20.0)

        feeding.feed(carnivores)

        assert ecology.energy(carnivores)[0] == pytest.approx(100.0)

    def test_a_generalist_converts_worse_than_a_specialist(self):
        """#102's convex frontier, observed through the transfer rather than in the arithmetic."""
        store, species, _, ecology, feeding = make_world()
        specialist = graze_at(store, species, [(2.0, 2.0)], animal_derived=-20.0)
        generalist = graze_at(store, species, [(8.0, 8.0)], animal_derived=0.0)

        feeding.feed(Selection.from_mask(specialist.to_mask() | generalist.to_mask()))

        assert ecology.energy(specialist)[0] > ecology.energy(generalist)[0]


class TestNutrientConservation:
    """§2.5's closed loop, across the one step that moves nutrients out of the field and into an
    animal. What is not assimilated must land back in the soil rather than vanishing."""

    def test_total_nutrients_are_conserved_through_a_feed(self):
        store, species, plants, _, feeding = make_world()
        grazers = graze_at(store, species, [(3.0, 3.0), (7.0, 7.0)])
        opening = plants.total_nutrients()

        feeding.feed(grazers)

        assert plants.total_nutrients() == pytest.approx(opening, rel=1e-9)

    def test_the_unassimilated_fraction_fertilises_the_cell_it_was_eaten_in(self):
        store, species, plants, _, feeding = make_world(assimilation_max=0.5)
        grazers = graze_at(store, species, [(5.0, 5.0)], animal_derived=-20.0)
        before = plants.soil_nutrients[5, 5]

        feeding.feed(grazers)

        assert plants.soil_nutrients[5, 5] > before

    def test_a_perfect_gut_leaves_no_faeces(self):
        store, species, plants, _, feeding = make_world(assimilation_max=1.0)
        grazers = graze_at(store, species, [(5.0, 5.0)], animal_derived=-20.0)
        before = plants.soil_nutrients[5, 5]

        feeding.feed(grazers)

        assert plants.soil_nutrients[5, 5] == pytest.approx(before)

    def test_a_poor_digester_fertilises_more_than_a_good_one(self):
        """§2.5's consequence, and the reason faeces is not a separate mechanic: the animal that
        wastes most of its food is the one enriching the ground it grazes."""
        store, species, plants, _, feeding = make_world(assimilation_max=0.9)
        good = graze_at(store, species, [(2.0, 2.0)], animal_derived=-20.0)
        poor = graze_at(store, species, [(8.0, 8.0)], animal_derived=0.0)
        before_good = plants.soil_nutrients[2, 2]
        before_poor = plants.soil_nutrients[8, 8]

        feeding.feed(Selection.from_mask(good.to_mask() | poor.to_mask()))

        assert (plants.soil_nutrients[8, 8] - before_poor) > (
            plants.soil_nutrients[2, 2] - before_good
        )


class TestContention:
    def test_grazers_sharing_a_cell_split_it(self):
        """Delegated to `Plants.graze` rather than re-derived here; this pins that feeding actually
        goes through it, since a per-animal harvest would let n grazers take n times the crop."""
        store, species, plants, _, feeding = make_world(intake_rate=1000.0)
        standing = plants.biomass[5, 5]
        crowd = graze_at(store, species, [(5.0, 5.0)] * 4)

        feeding.feed(crowd)

        assert plants.biomass[5, 5] == pytest.approx(0.0, abs=1e-9)
        assert plants.total_nutrients() == pytest.approx(
            plants.total_nutrients(), rel=1e-9
        )
        assert standing > 0.0
