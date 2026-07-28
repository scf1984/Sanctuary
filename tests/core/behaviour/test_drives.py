import numpy as np
import pytest

from core.behaviour.drives import (
    Fatigue,
    FatigueConfig,
    Hunger,
    HungerConfig,
    Lust,
    LustConfig,
    Thirst,
    ThirstConfig,
)
from core.behaviour.service import Behaviour
from core.ecology.metabolism import Metabolism, MetabolismConfig
from core.ecology.plants import Plants, PlantsConfig
from core.ecology.service import Ecology
from core.entities.store import EntityStore
from core.genetics.service import Genetics
from core.genetics.species import SpeciesRegistry
from core.genetics.vocabulary import GeneVocabulary
from core.selection import Selection
from core.services import ColumnRegistry
from core.world.climate import Climate, ClimateConfig
from core.world.terrain import Terrain
from core.world.water import Water

GENE_NAMES = ("size", "speed", "sight", "insulation")

METABOLISM_CONFIG = MetabolismConfig(
    gene_costs={"size": 2.0, "speed": 3.0, "sight": 1.0, "insulation": 1.0},
    basal_rate=1.0,
    thermoregulation_rate=0.5,
    neutral_temperature=20.0,
    insulation_gene="insulation",
)

PLANTS_CONFIG = PlantsConfig(
    solar_constant=1.0,
    latitude_tilt=0.0,
    min_growth_temperature=0.0,
    optimal_growth_temperature=20.0,
    max_growth_temperature=40.0,
    nutrient_per_biomass=1.0,
    initial_soil_nutrients=100.0,
    senescence_rate=0.01,
    saturation_accumulation=10.0,
    max_rooting_depth=1.0,
)

HUNGER_CONFIG = HungerConfig(
    weight=1.0, satiation_energy=100.0, forage_reluctance=5.0, sight_gene="sight"
)


class World:
    """A flat, uniform-temperature world with every service #22's drives read from.

    Flat terrain with a zero latitude gradient makes `temperature_at` return `temperature`
    everywhere, so a test that is not about climate never has to care where it put its entities.
    """

    def __init__(self, capacity=8, temperature=20.0, grid=9, cell_size=1.0):
        self.store = EntityStore(initial_capacity=capacity, n_drives=4, n_genes=len(GENE_NAMES))
        self.columns = ColumnRegistry()
        self.vocabulary = GeneVocabulary(GENE_NAMES)
        self.species = SpeciesRegistry(self.vocabulary)
        self.genetics = Genetics(self.store, self.columns, self.species)
        self.terrain = Terrain(np.zeros((grid, grid), dtype=np.float32), cell_size=cell_size)
        self.climate = Climate(
            self.terrain,
            ClimateConfig(
                equator_y=0.0, equator_temperature=temperature, latitude_gradient=0.0
            ),
        )
        self.water = Water(
            depth=np.zeros((grid, grid), dtype=np.float32),
            flow_direction=np.full((grid, grid), -1, dtype=np.int8),
            flow_accumulation=np.ones((grid, grid), dtype=np.float32),
            cell_size=cell_size,
        )
        self.plants = Plants(self.terrain, self.climate, self.water, PLANTS_CONFIG)
        self.ecology = Ecology(
            self.store,
            self.columns,
            self.genetics,
            self.climate,
            Metabolism(self.vocabulary, METABOLISM_CONFIG),
        )
        self.behaviour = Behaviour(self.store, self.columns)
        self.species_id = self.species.register(GENE_NAMES)

    def spawn(self, n, **columns):
        """Allocate `n` entities of the single registered species and return their Selection."""
        columns.setdefault("species_id", np.full(n, self.species_id, dtype=np.int32))
        ids = self.store.allocate(n, **columns)
        rows = [self.store._id_to_row[i] for i in ids.tolist()]
        return Selection.from_indices(np.array(rows, dtype=np.int64), self.store.capacity)


def gene_rows(*rows):
    """Stack gene rows given as {name: value} dicts into a (n, n_genes) float32 matrix."""
    matrix = np.zeros((len(rows), len(GENE_NAMES)), dtype=np.float32)
    for i, genes in enumerate(rows):
        for name, value in genes.items():
            matrix[i, GENE_NAMES.index(name)] = value
    return matrix


class TestHungerScore:
    def test_a_full_pool_wants_nothing_and_an_empty_one_wants_everything(self):
        world = World()
        selection = world.spawn(3, energy=np.array([100.0, 50.0, 0.0], dtype=np.float32))
        hunger = Hunger(
            world.store, world.ecology, world.genetics, world.plants, world.vocabulary,
            HUNGER_CONFIG,
        )

        assert hunger.score(selection) == pytest.approx([0.0, 0.5, 1.0])

    def test_energy_above_satiation_does_not_produce_negative_hunger(self):
        """A drive scoring below zero would still lose every contest, but it would also make
        "zero means no pull" untrue and break the no-active-drive reading in the service.
        """
        world = World()
        selection = world.spawn(1, energy=np.array([500.0], dtype=np.float32))
        hunger = Hunger(
            world.store, world.ecology, world.genetics, world.plants, world.vocabulary,
            HUNGER_CONFIG,
        )

        assert hunger.score(selection) == pytest.approx([0.0])

    def test_weight_scales_the_whole_drive(self):
        world = World()
        selection = world.spawn(1, energy=np.array([0.0], dtype=np.float32))
        config = HungerConfig(
            weight=3.0, satiation_energy=100.0, forage_reluctance=5.0, sight_gene="sight"
        )
        hunger = Hunger(
            world.store, world.ecology, world.genetics, world.plants, world.vocabulary, config
        )

        assert hunger.score(selection) == pytest.approx([3.0])


class TestForageTarget:
    """The rule CLAUDE.md §2.5 settles: argmax of biomass / (1 + distance / forage_reluctance)."""

    def test_a_grazer_walks_to_the_only_patch_it_can_see(self):
        world = World()
        selection = world.spawn(
            1, x=np.array([4.0], dtype=np.float32), y=np.array([4.0], dtype=np.float32)
        )
        world.genetics.set_genes(selection, gene_rows({"sight": 3.0}))
        world.plants.biomass[:] = 0.0
        world.plants.biomass[4, 6] = 50.0
        hunger = Hunger(
            world.store, world.ecology, world.genetics, world.plants, world.vocabulary,
            HUNGER_CONFIG,
        )

        target_x, target_y = hunger.forage_target(selection)

        assert (target_x, target_y) == (pytest.approx([6.0]), pytest.approx([4.0]))

    def test_a_near_patch_beats_a_richer_far_one_when_reluctance_is_low(self):
        """Small forage_reluctance keeps grazers local and strips ground bare before they move —
        which is the local grazing pressure the field model of #18 exists to express.
        """
        world = World()
        selection = world.spawn(
            1, x=np.array([4.0], dtype=np.float32), y=np.array([4.0], dtype=np.float32)
        )
        world.genetics.set_genes(selection, gene_rows({"sight": 4.0}))
        world.plants.biomass[:] = 0.0
        world.plants.biomass[4, 5] = 10.0  # 1 unit away
        world.plants.biomass[4, 8] = 25.0  # 4 units away, 2.5x the standing crop
        config = HungerConfig(
            weight=1.0, satiation_energy=100.0, forage_reluctance=0.5, sight_gene="sight"
        )
        hunger = Hunger(
            world.store, world.ecology, world.genetics, world.plants, world.vocabulary, config
        )

        target_x, _ = hunger.forage_target(selection)

        # 10/(1 + 1/0.5) = 3.33 against 25/(1 + 4/0.5) = 2.78 — the near cell wins on discount
        # alone, not on a tie-break.
        assert target_x == pytest.approx([5.0])

    def test_a_high_reluctance_grazer_crosses_to_the_richer_patch(self):
        world = World()
        selection = world.spawn(
            1, x=np.array([4.0], dtype=np.float32), y=np.array([4.0], dtype=np.float32)
        )
        world.genetics.set_genes(selection, gene_rows({"sight": 4.0}))
        world.plants.biomass[:] = 0.0
        world.plants.biomass[4, 5] = 10.0
        world.plants.biomass[4, 8] = 25.0
        config = HungerConfig(
            weight=1.0, satiation_energy=100.0, forage_reluctance=100.0, sight_gene="sight"
        )
        hunger = Hunger(
            world.store, world.ecology, world.genetics, world.plants, world.vocabulary, config
        )

        target_x, _ = hunger.forage_target(selection)

        assert target_x == pytest.approx([8.0])

    def test_sight_range_gates_what_a_forager_can_find(self):
        """Perception a forager could not pay for would leave sight range charged by the metabolic
        budget while buying nothing (CLAUDE.md §2.5). Two identical animals, different sight.
        """
        world = World()
        selection = world.spawn(
            2,
            x=np.array([4.0, 4.0], dtype=np.float32),
            y=np.array([4.0, 4.0], dtype=np.float32),
        )
        world.genetics.set_genes(selection, gene_rows({"sight": 1.0}, {"sight": 4.0}))
        world.plants.biomass[:] = 0.0
        world.plants.biomass[4, 7] = 50.0
        hunger = Hunger(
            world.store, world.ecology, world.genetics, world.plants, world.vocabulary,
            HUNGER_CONFIG,
        )

        target_x, target_y = hunger.forage_target(selection)

        # The short-sighted animal cannot see the patch and stays put; the far-sighted one goes.
        assert (target_x[0], target_y[0]) == (pytest.approx(4.0), pytest.approx(4.0))
        assert (target_x[1], target_y[1]) == (pytest.approx(7.0), pytest.approx(4.0))

    def test_an_animal_that_can_see_no_food_stays_where_it_is(self):
        world = World()
        selection = world.spawn(
            1, x=np.array([4.0], dtype=np.float32), y=np.array([4.0], dtype=np.float32)
        )
        world.genetics.set_genes(selection, gene_rows({"sight": 3.0}))
        world.plants.biomass[:] = 0.0
        hunger = Hunger(
            world.store, world.ecology, world.genetics, world.plants, world.vocabulary,
            HUNGER_CONFIG,
        )

        target_x, target_y = hunger.forage_target(selection)

        assert (target_x, target_y) == (pytest.approx([4.0]), pytest.approx([4.0]))

    def test_an_unexpressed_sight_gene_leaves_a_forager_grazing_underfoot(self):
        """Expression, not genotype, is what a forager sees with — the same rule that makes an
        unexpressed gene cost nothing (#17). `perceive` always reports the cell underfoot, so a
        blind animal still finds food it is standing on rather than starving on top of it.
        """
        world = World()
        blind = world.species.register(("size", "speed", "insulation"))
        selection = world.spawn(
            1,
            x=np.array([4.0], dtype=np.float32),
            y=np.array([4.0], dtype=np.float32),
            species_id=np.array([blind], dtype=np.int32),
        )
        world.genetics.set_genes(selection, gene_rows({"sight": 4.0}))
        world.plants.biomass[:] = 0.0
        world.plants.biomass[4, 4] = 5.0
        world.plants.biomass[4, 8] = 90.0
        hunger = Hunger(
            world.store, world.ecology, world.genetics, world.plants, world.vocabulary,
            HUNGER_CONFIG,
        )

        target_x, target_y = hunger.forage_target(selection)

        assert (target_x, target_y) == (pytest.approx([4.0]), pytest.approx([4.0]))

    def test_a_returned_target_can_be_grazed_without_a_bounds_check(self):
        """#93 guarantees perceived positions are real in-world cell centres. A forager at the map
        edge is where that would break, so the target it produces is fed straight to `graze`.
        """
        world = World()
        selection = world.spawn(
            1, x=np.array([0.0], dtype=np.float32), y=np.array([0.0], dtype=np.float32)
        )
        world.genetics.set_genes(selection, gene_rows({"sight": 3.0}))
        world.plants.biomass[:] = 0.0
        world.plants.biomass[0, 2] = 20.0
        hunger = Hunger(
            world.store, world.ecology, world.genetics, world.plants, world.vocabulary,
            HUNGER_CONFIG,
        )

        target_x, target_y = hunger.forage_target(selection)
        harvested = world.plants.graze(target_x, target_y, np.array([5.0]))

        assert harvested == pytest.approx([5.0])

    def test_an_empty_selection_produces_empty_targets(self):
        world = World()
        hunger = Hunger(
            world.store, world.ecology, world.genetics, world.plants, world.vocabulary,
            HUNGER_CONFIG,
        )

        target_x, target_y = hunger.forage_target(Selection.none(world.store.capacity))

        assert target_x.shape == (0,) and target_y.shape == (0,)


class TestThirst:
    def test_thirst_rises_with_ambient_heat(self):
        config = ThirstConfig(weight=1.0, onset_temperature=20.0, saturation_temperature=40.0)

        scores = []
        for temperature in (15.0, 30.0, 50.0):
            world = World(temperature=temperature)
            selection = world.spawn(1)
            scores.append(Thirst(world.store, world.climate, config).score(selection)[0])

        # Below onset, halfway up the span, and clamped at saturation.
        assert scores == pytest.approx([0.0, 0.5, 1.0])

    def test_thirst_is_read_at_each_entitys_own_position(self):
        """The climate field is what makes hot ground push animals toward water, so two members of
        one species in different places must want different things.
        """
        world = World(grid=21, cell_size=1.0)
        world.climate = Climate(
            world.terrain,
            ClimateConfig(equator_y=0.0, equator_temperature=40.0, latitude_gradient=1.0),
        )
        selection = world.spawn(
            2,
            x=np.array([0.0, 0.0], dtype=np.float32),
            y=np.array([0.0, 20.0], dtype=np.float32),
        )
        config = ThirstConfig(weight=1.0, onset_temperature=20.0, saturation_temperature=40.0)

        scores = Thirst(world.store, world.climate, config).score(selection)

        # y=0 sits at 40 degC (saturated); y=20 sits at 20 degC, exactly at onset.
        assert scores == pytest.approx([1.0, 0.0])

    def test_saturation_must_exceed_onset(self):
        with pytest.raises(ValueError):
            ThirstConfig(weight=1.0, onset_temperature=30.0, saturation_temperature=30.0)


class TestLust:
    def test_an_immature_animal_wants_no_mate_however_fat(self):
        world = World()
        selection = world.spawn(1, energy=np.array([100.0], dtype=np.float32))
        world.store.age[selection.to_indices()] = 5
        config = LustConfig(
            weight=1.0, maturity_age=100, breeding_energy=20.0, abundant_energy=70.0
        )

        assert Lust(world.store, world.ecology, config).score(selection) == pytest.approx([0.0])

    def test_a_mature_animal_below_breeding_energy_wants_no_mate(self):
        """Gestation charges upkeep like any other trait (§2.5); wanting what you cannot afford
        would select for breeding yourself to death.
        """
        world = World()
        selection = world.spawn(1, energy=np.array([10.0], dtype=np.float32))
        world.store.age[selection.to_indices()] = 200
        config = LustConfig(
            weight=1.0, maturity_age=100, breeding_energy=20.0, abundant_energy=70.0
        )

        assert Lust(world.store, world.ecology, config).score(selection) == pytest.approx([0.0])

    def test_lust_rises_with_energy_above_the_breeding_floor(self):
        world = World()
        selection = world.spawn(3, energy=np.array([20.0, 45.0, 90.0], dtype=np.float32))
        world.store.age[selection.to_indices()] = 200
        config = LustConfig(
            weight=1.0, maturity_age=100, breeding_energy=20.0, abundant_energy=70.0
        )

        scores = Lust(world.store, world.ecology, config).score(selection)

        # At the floor, halfway to abundance, and clamped above it.
        assert scores == pytest.approx([0.0, 0.5, 1.0])

    def test_maturity_is_counted_in_ticks(self):
        """The tick counter is the only clock (CLAUDE.md §2.1) — maturity is a row of `age`."""
        world = World()
        selection = world.spawn(2, energy=np.array([70.0, 70.0], dtype=np.float32))
        world.store.age[selection.to_indices()] = [99, 100]
        config = LustConfig(
            weight=1.0, maturity_age=100, breeding_energy=20.0, abundant_energy=70.0
        )

        assert Lust(world.store, world.ecology, config).score(selection) == pytest.approx(
            [0.0, 1.0]
        )


class TestFatigue:
    def test_fatigue_is_the_health_deficit(self):
        world = World()
        selection = world.spawn(3, health=np.array([1.0, 0.25, 0.0], dtype=np.float32))

        scores = Fatigue(world.store, FatigueConfig(weight=1.0)).score(selection)

        assert scores == pytest.approx([0.0, 0.75, 1.0])

    def test_a_negative_weight_is_rejected(self):
        """A negative weight inverts a drive: the worse the injury, the less it wants to rest."""
        with pytest.raises(ValueError):
            FatigueConfig(weight=-1.0)


class TestDrivesCompeting:
    def test_a_starving_animal_forages_and_a_fed_injured_one_rests(self):
        """The whole point of #22 on a synthetic population: the same registered set of drives
        resolves two animals to different actions from their state alone.
        """
        world = World()
        selection = world.spawn(
            2,
            energy=np.array([0.0, 100.0], dtype=np.float32),
            health=np.array([1.0, 0.2], dtype=np.float32),
            x=np.array([4.0, 4.0], dtype=np.float32),
            y=np.array([4.0, 4.0], dtype=np.float32),
        )
        world.store.age[selection.to_indices()] = 0
        world.behaviour.register(
            Hunger(
                world.store, world.ecology, world.genetics, world.plants, world.vocabulary,
                HUNGER_CONFIG,
            )
        )
        world.behaviour.register(
            Thirst(
                world.store,
                world.climate,
                ThirstConfig(weight=1.0, onset_temperature=25.0, saturation_temperature=40.0),
            )
        )
        world.behaviour.register(
            Lust(
                world.store,
                world.ecology,
                LustConfig(
                    weight=1.0, maturity_age=100, breeding_energy=20.0, abundant_energy=70.0
                ),
            )
        )
        world.behaviour.register(Fatigue(world.store, FatigueConfig(weight=1.0)))

        world.behaviour.score(selection)

        starving = Selection.from_indices(selection.to_indices()[:1], world.store.capacity)
        injured = Selection.from_indices(selection.to_indices()[1:], world.store.capacity)
        assert world.behaviour.driven_by("hunger", selection) == starving
        assert world.behaviour.driven_by("fatigue", selection) == injured

    def test_the_breakdown_explains_the_winner(self):
        """"It rested because fatigue outscored hunger" has to be recoverable from the store, not
        told as a story about it (CLAUDE.md §2.5, §3.3).
        """
        world = World()
        selection = world.spawn(
            1,
            energy=np.array([80.0], dtype=np.float32),
            health=np.array([0.1], dtype=np.float32),
        )
        world.behaviour.register(
            Hunger(
                world.store, world.ecology, world.genetics, world.plants, world.vocabulary,
                HUNGER_CONFIG,
            )
        )
        world.behaviour.register(Fatigue(world.store, FatigueConfig(weight=1.0)))

        world.behaviour.score(selection)
        breakdown = world.behaviour.breakdown(selection)

        assert breakdown["hunger"] == pytest.approx([0.2])
        assert breakdown["fatigue"] == pytest.approx([0.9])
        assert world.behaviour.driven_by("fatigue", selection) == selection
