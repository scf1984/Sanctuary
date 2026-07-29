import numpy as np
import pytest

from core.behaviour.drives import (
    Fatigue,
    FatigueConfig,
    Fear,
    FearConfig,
    Hunger,
    HungerConfig,
    Lust,
    LustConfig,
    Thirst,
    ThirstConfig,
)
from core.behaviour.service import Behaviour
from core.ecology.cues import CueField, CueFieldConfig, Scent, ScentGenes
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

# Three cue channels rather than the eight CLAUDE.md §2.5 settles on: the algebra is identical and
# a test that spells out eight signature components per creature stops being readable.
CHANNELS = 3
SIGNATURE_GENES = tuple(f"signature_{i}" for i in range(CHANNELS))
AVERSION_GENES = tuple(
    tuple(f"aversion{d}_{i}" for i in range(CHANNELS)) for d in range(2)
)
FLAT_AVERSION = tuple(name for block in AVERSION_GENES for name in block)
GENE_NAMES = (
    "size",
    "speed",
    "sight",
    "insulation",
    "scent_emission",
    "scent_acuity",
    *SIGNATURE_GENES,
    *FLAT_AVERSION,
)
SCENT_GENES = ScentGenes(emission_gene="scent_emission", signature_genes=SIGNATURE_GENES)

METABOLISM_CONFIG = MetabolismConfig(
    gene_costs={
        "size": 2.0,
        "speed": 3.0,
        "sight": 1.0,
        "insulation": 1.0,
        # Emission is free pending #20: charging it would make silence both cheaper and safer,
        # driving it to zero in every lineage (CLAUDE.md §2.5).
        "scent_emission": 0.0,
        "scent_acuity": 0.5,
        # Signature and aversion are free: smelling of something and minding something cost
        # nothing to carry. §2.5 requires every gene declare a cost, zero included.
        **{name: 0.0 for name in (*SIGNATURE_GENES, *FLAT_AVERSION)},
    },
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
        self.store = EntityStore(initial_capacity=capacity, n_drives=5, n_genes=len(GENE_NAMES))
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



FEAR_CONFIG = FearConfig(
    weight=1.0,
    scent_acuity_gene="scent_acuity",
    aversion_genes=AVERSION_GENES,
    detection_threshold=0.01,
    saturation=1.0,
)


class FearWorld(World):
    """A World plus the cue field and scent binder the fear drive reads.

    Nothing here distinguishes predator from prey by species. A "predator" is a creature whose
    *signature* sits in channel 0, and a frightened creature is one whose *aversion* points there.
    Both are ordinary genes, which is the whole point (CLAUDE.md §2.5).
    """

    def __init__(self, diffusion_range=3.0, **kwargs):
        super().__init__(**kwargs)
        self.cue_field = CueField(
            self.terrain, CHANNELS, CueFieldConfig(diffusion_range=diffusion_range)
        )
        self.scent = Scent(
            self.store, self.genetics, self.cue_field, self.vocabulary, SCENT_GENES
        )

    def fear(self, config=FEAR_CONFIG):
        return Fear(self.store, self.genetics, self.scent, self.vocabulary, config)

    def spawn_as(self, species_id, n, **columns):
        columns["species_id"] = np.full(n, species_id, dtype=np.int32)
        ids = self.store.allocate(n, **columns)
        rows = [self.store._id_to_row[i] for i in ids.tolist()]
        return Selection.from_indices(np.array(rows, dtype=np.int64), self.store.capacity)


def dangerous(emission=1.0):
    """Genes for a creature that smells of channel 0 and fears nothing."""
    return {"scent_emission": emission, "signature_0": 1.0}


def timid(acuity=50.0):
    """Genes for a creature that fears channel 0 and broadcasts nothing."""
    return {"scent_acuity": acuity, "aversion0_0": 1.0}


class TestFearScore:
    def test_a_creature_alone_in_the_world_fears_nothing(self):
        world = FearWorld(grid=21)
        prey = world.spawn_as(0, 1, x=np.float32([10.0]), y=np.float32([10.0]))
        world.genetics.set_genes(prey, gene_rows(timid()))
        world.scent.rebuild(prey)

        assert world.fear().score(prey) == pytest.approx([0.0])

    def test_a_nearby_source_of_a_feared_signature_is_feared(self):
        world = FearWorld(grid=21)
        prey = world.spawn_as(0, 1, x=np.float32([10.0]), y=np.float32([10.0]))
        predator = world.spawn_as(0, 1, x=np.float32([11.0]), y=np.float32([10.0]))
        world.genetics.set_genes(prey, gene_rows(timid()))
        world.genetics.set_genes(predator, gene_rows(dangerous()))
        world.scent.rebuild(prey | predator)

        assert world.fear().score(prey)[0] > 0.0

    def test_fear_is_not_symmetric_and_needs_no_matrix_to_say_so(self):
        """Prey fear predators far more than predators fear prey. With aversion genetic, that
        asymmetry is just two creatures carrying different aversion vectors — there is no table
        anywhere that could be accidentally symmetrized.
        """
        world = FearWorld(grid=21)
        prey = world.spawn_as(0, 1, x=np.float32([10.0]), y=np.float32([10.0]))
        predator = world.spawn_as(0, 1, x=np.float32([11.0]), y=np.float32([10.0]))
        world.genetics.set_genes(prey, gene_rows(timid()))
        world.genetics.set_genes(predator, gene_rows(dangerous()))
        world.scent.rebuild(prey | predator)

        assert world.fear().score(prey)[0] > 0.0
        assert world.fear().score(predator) == pytest.approx([0.0])

    def test_aversion_discriminates_between_signatures(self):
        """The reason cue space has more than one dimension: fearing wolves must not mean fearing
        every animal that happens to smell of something.
        """
        world = FearWorld(grid=21, capacity=16)
        prey = world.spawn_as(0, 1, x=np.float32([10.0]), y=np.float32([10.0]))
        harmless = world.spawn_as(0, 1, x=np.float32([11.0]), y=np.float32([10.0]))
        world.genetics.set_genes(prey, gene_rows(timid()))
        # Loud, but in a channel this creature does not care about.
        world.genetics.set_genes(
            harmless, gene_rows({"scent_emission": 10.0, "signature_1": 1.0})
        )
        world.scent.rebuild(prey | harmless)

        assert world.fear().score(prey) == pytest.approx([0.0])

    def test_fear_falls_off_with_distance_from_the_source(self):
        world = FearWorld(grid=41)
        near = world.spawn_as(0, 1, x=np.float32([21.0]), y=np.float32([20.0]))
        far = world.spawn_as(0, 1, x=np.float32([27.0]), y=np.float32([20.0]))
        predator = world.spawn_as(0, 1, x=np.float32([20.0]), y=np.float32([20.0]))
        prey = near | far
        world.genetics.set_genes(prey, gene_rows(timid(), timid()))
        world.genetics.set_genes(predator, gene_rows(dangerous()))
        world.scent.rebuild(prey | predator)

        scores = world.fear().score(prey)

        assert scores[0] > scores[1]

    def test_a_pack_is_more_frightening_than_a_lone_predator(self):
        """Concentration, not nearest distance: several sources nearby are worse than one."""
        scores = []
        for pack_size in (1, 4):
            world = FearWorld(grid=41, capacity=16)
            prey = world.spawn_as(0, 1, x=np.float32([20.0]), y=np.float32([20.0]))
            pack = world.spawn_as(
                0,
                pack_size,
                x=np.float32([23.0] * pack_size),
                y=np.float32([20.0] * pack_size),
            )
            world.genetics.set_genes(prey, gene_rows(timid(20.0)))
            world.genetics.set_genes(pack, gene_rows(*[dangerous()] * pack_size))
            world.scent.rebuild(prey | pack)
            scores.append(world.fear().score(prey)[0])

        assert scores[1] > scores[0]

    def test_a_louder_predator_is_detected_from_further_away(self):
        """Emission is under selection too: a stealthy predator is one whose lineage drove its
        own broadcast down.
        """
        scores = []
        for emission in (0.2, 5.0):
            world = FearWorld(grid=41)
            prey = world.spawn_as(0, 1, x=np.float32([26.0]), y=np.float32([20.0]))
            predator = world.spawn_as(0, 1, x=np.float32([20.0]), y=np.float32([20.0]))
            world.genetics.set_genes(prey, gene_rows(timid()))
            world.genetics.set_genes(predator, gene_rows(dangerous(emission)))
            world.scent.rebuild(prey | predator)
            scores.append(world.fear().score(prey)[0])

        assert scores[1] > scores[0]

    def test_stronger_aversion_means_more_fear_of_the_same_thing(self):
        world = FearWorld(grid=21, capacity=16)
        timid_one = world.spawn_as(0, 1, x=np.float32([12.0]), y=np.float32([10.0]))
        bold = world.spawn_as(0, 1, x=np.float32([12.0]), y=np.float32([11.0]))
        predator = world.spawn_as(0, 1, x=np.float32([10.0]), y=np.float32([10.0]))
        prey = timid_one | bold
        world.genetics.set_genes(
            prey,
            gene_rows(
                {"scent_acuity": 20.0, "aversion0_0": 1.0},
                {"scent_acuity": 20.0, "aversion0_0": 0.1},
            ),
        )
        world.genetics.set_genes(predator, gene_rows(dangerous()))
        world.scent.rebuild(prey | predator)

        scores = world.fear().score(prey)

        assert scores[0] > scores[1]

    def test_weight_scales_the_whole_drive(self):
        world = FearWorld(grid=21)
        prey = world.spawn_as(0, 1, x=np.float32([10.0]), y=np.float32([10.0]))
        predator = world.spawn_as(0, 1, x=np.float32([10.0]), y=np.float32([10.0]))
        world.genetics.set_genes(prey, gene_rows(timid(1000.0)))
        world.genetics.set_genes(predator, gene_rows(dangerous()))
        world.scent.rebuild(prey | predator)

        loud = FearConfig(
            weight=3.0,
            scent_acuity_gene="scent_acuity",
            aversion_genes=AVERSION_GENES,
            detection_threshold=0.01,
            saturation=1.0,
        )
        assert world.fear(loud).score(prey) == pytest.approx(
            3.0 * world.fear().score(prey), rel=1e-5
        )


class TestEmergentBehaviour:
    """Behaviours nobody implemented, which fall out of signature and aversion being genes.

    These are the return on the whole encoding (CLAUDE.md §2.5) — if any of them stops holding,
    the cue space has been reduced to an authored threat table wearing different names.
    """

    def test_a_cannibal_fears_its_own_kind(self):
        """No diagonal, no special case: a lineage whose aversion overlaps its own signature."""
        world = FearWorld(grid=21)
        alone = world.spawn_as(0, 1, x=np.float32([10.0]), y=np.float32([10.0]))
        neighbour = world.spawn_as(0, 1, x=np.float32([11.0]), y=np.float32([10.0]))
        both = alone | neighbour
        cannibal = {
            "scent_emission": 1.0,
            "signature_0": 1.0,
            "scent_acuity": 50.0,
            "aversion0_0": 1.0,
        }
        world.genetics.set_genes(both, gene_rows(cannibal, cannibal))
        world.scent.rebuild(both)

        assert world.fear().score(alone)[0] > 0.0

    def test_a_cannibal_standing_alone_is_not_afraid_of_itself(self):
        """The self-exclusion this depends on. Without it, cannibalism would be indistinguishable
        from a permanently panicking animal, and the emergent version could never be trusted.
        """
        world = FearWorld(grid=21)
        alone = world.spawn_as(0, 1, x=np.float32([10.0]), y=np.float32([10.0]))
        world.genetics.set_genes(
            alone,
            gene_rows(
                {
                    "scent_emission": 5.0,
                    "signature_0": 1.0,
                    "scent_acuity": 500.0,
                    "aversion0_0": 1.0,
                }
            ),
        )
        world.scent.rebuild(alone)

        assert world.fear().score(alone) == pytest.approx([0.0])

    def test_a_mimic_is_avoided_without_being_dangerous(self):
        """Batesian mimicry, unauthored: a harmless lineage whose signature has drifted toward a
        feared one is feared, because nothing anywhere records which species is actually a threat.
        """
        world = FearWorld(grid=21)
        prey = world.spawn_as(0, 1, x=np.float32([10.0]), y=np.float32([10.0]))
        mimic = world.spawn_as(0, 1, x=np.float32([11.0]), y=np.float32([10.0]))
        world.genetics.set_genes(prey, gene_rows(timid()))
        world.genetics.set_genes(mimic, gene_rows(dangerous()))
        world.scent.rebuild(prey | mimic)

        assert world.fear().score(prey)[0] > 0.0

    def test_a_predator_that_drifts_out_of_signature_stops_being_feared(self):
        """The predator's half of the arms race: selection on its own signature makes it stealthy
        without any change to what its prey inherited.
        """
        scores = []
        for signature_gene in ("signature_0", "signature_2"):
            world = FearWorld(grid=21)
            prey = world.spawn_as(0, 1, x=np.float32([10.0]), y=np.float32([10.0]))
            predator = world.spawn_as(0, 1, x=np.float32([11.0]), y=np.float32([10.0]))
            world.genetics.set_genes(prey, gene_rows(timid()))
            world.genetics.set_genes(
                predator, gene_rows({"scent_emission": 1.0, signature_gene: 1.0})
            )
            world.scent.rebuild(prey | predator)
            scores.append(world.fear().score(prey)[0])

        assert scores[0] > 0.0
        assert scores[1] == pytest.approx(0.0)

    def test_speciation_costs_fear_nothing(self):
        """A daughter species inherits signature and aversion like any other trait, and there is
        no per-species table to extend — so CLAUDE.md §2.3's "speciation is a species-id write
        plus a new mask row" stays literally true.
        """
        world = FearWorld(grid=21)
        daughter = world.species.derive(world.species_id)
        prey = world.spawn_as(daughter, 1, x=np.float32([10.0]), y=np.float32([10.0]))
        predator = world.spawn_as(daughter, 1, x=np.float32([11.0]), y=np.float32([10.0]))
        world.genetics.set_genes(prey, gene_rows(timid()))
        world.genetics.set_genes(predator, gene_rows(dangerous()))
        world.scent.rebuild(prey | predator)

        assert world.fear().score(prey)[0] > 0.0


class TestScentAcuity:
    def test_a_keener_nose_detects_the_same_threat_from_further_away(self):
        """CLAUDE.md §2.5: for a plume, sensitivity and range are the same parameter. The dull
        creature must not merely be *less* afraid — it must not detect the threat at all.
        """
        world = FearWorld(grid=41, capacity=16)
        dull = world.spawn_as(0, 1, x=np.float32([28.0]), y=np.float32([20.0]))
        keen = world.spawn_as(0, 1, x=np.float32([28.0]), y=np.float32([21.0]))
        predator = world.spawn_as(0, 1, x=np.float32([20.0]), y=np.float32([20.0]))
        prey = dull | keen
        world.genetics.set_genes(prey, gene_rows(timid(1.0), timid(500.0)))
        world.genetics.set_genes(predator, gene_rows(dangerous()))
        world.scent.rebuild(prey | predator)

        scores = world.fear().score(prey)

        assert scores[0] == pytest.approx(0.0)
        assert scores[1] > 0.0

    def test_an_unexpressed_aversion_gene_leaves_a_creature_fearless(self):
        """Expression, not genotype — the same rule that makes an unexpressed gene cost nothing
        (#17). A species that does not express an aversion carries it, and can express it again
        generations later, but pays nothing and gains nothing now.
        """
        world = FearWorld(grid=21)
        fearless = world.species.register(
            ("size", "speed", "sight", "insulation", "scent_emission", "scent_acuity")
        )
        prey = world.spawn_as(fearless, 1, x=np.float32([10.0]), y=np.float32([10.0]))
        predator = world.spawn_as(0, 1, x=np.float32([10.0]), y=np.float32([10.0]))
        world.genetics.set_genes(prey, gene_rows(timid(500.0)))
        world.genetics.set_genes(predator, gene_rows(dangerous()))
        world.scent.rebuild(prey | predator)

        assert world.fear().score(prey) == pytest.approx([0.0])

    def test_detection_below_the_threshold_is_nothing_at_all(self):
        """Without a threshold every creature detects every trace from anywhere and acuity
        collapses into a panic multiplier.
        """
        world = FearWorld(grid=41)
        prey = world.spawn_as(0, 1, x=np.float32([32.0]), y=np.float32([20.0]))
        predator = world.spawn_as(0, 1, x=np.float32([20.0]), y=np.float32([20.0]))
        world.genetics.set_genes(prey, gene_rows(timid(1.0)))
        world.genetics.set_genes(predator, gene_rows(dangerous()))
        world.scent.rebuild(prey | predator)

        assert world.fear().score(prey) == pytest.approx([0.0])

    def test_detection_saturates_at_certainty(self):
        world = FearWorld(grid=21, capacity=16)
        prey = world.spawn_as(0, 1, x=np.float32([10.0]), y=np.float32([10.0]))
        predator = world.spawn_as(0, 8, x=np.float32([10.0] * 8), y=np.float32([10.0] * 8))
        world.genetics.set_genes(prey, gene_rows(timid(10_000.0)))
        world.genetics.set_genes(predator, gene_rows(*[dangerous()] * 8))
        world.scent.rebuild(prey | predator)

        assert world.fear().score(prey) == pytest.approx([1.0])


class TestNoisyOr:
    def test_a_single_channel_passes_through_unchanged(self):
        """With one channel the product collapses to that probability, so today's scores are the
        scent channel exactly — which is what makes #24's addition legible as a change.
        """
        world = FearWorld(grid=21)
        prey = world.spawn_as(0, 1, x=np.float32([10.0]), y=np.float32([10.0]))
        predator = world.spawn_as(0, 1, x=np.float32([11.0]), y=np.float32([10.0]))
        world.genetics.set_genes(prey, gene_rows(timid(30.0)))
        world.genetics.set_genes(predator, gene_rows(dangerous()))
        world.scent.rebuild(prey | predator)
        fear = world.fear()

        assert fear.score(prey) == pytest.approx(fear._channels(prey)[0], rel=1e-6)

    def test_channels_corroborate_without_exceeding_certainty(self):
        """The property that lets #24 add sight without retuning every other drive's weight
        (CLAUDE.md §2.5): two channels are scarier than one, and never run past 1.
        """
        world = FearWorld(grid=21)
        prey = world.spawn_as(0, 1, x=np.float32([10.0]), y=np.float32([10.0]))
        predator = world.spawn_as(0, 1, x=np.float32([11.0]), y=np.float32([10.0]))
        world.genetics.set_genes(prey, gene_rows(timid(30.0)))
        world.genetics.set_genes(predator, gene_rows(dangerous()))
        world.scent.rebuild(prey | predator)
        fear = world.fear()
        scent_only = fear.score(prey)[0]

        # Stand in for #24's sight channel until it exists, to pin the composition rule now.
        smelled = fear._channels(prey)
        sight = np.full(len(prey), 0.5, dtype=np.float32)
        fear._channels = lambda selection: [*smelled, sight]
        both = fear.score(prey)[0]

        assert both > scent_only
        assert both <= 1.0
        assert both == pytest.approx(1.0 - (1.0 - scent_only) * 0.5, rel=1e-5)


class TestFearConfig:
    def test_rejects_a_zero_detection_threshold(self):
        with pytest.raises(ValueError):
            FearConfig(
                weight=1.0,
                scent_acuity_gene="scent_acuity",
                aversion_genes=AVERSION_GENES,
                detection_threshold=0.0,
                saturation=1.0,
            )

    def test_rejects_saturation_at_or_below_the_threshold(self):
        with pytest.raises(ValueError):
            FearConfig(
                weight=1.0,
                scent_acuity_gene="scent_acuity",
                aversion_genes=AVERSION_GENES,
                detection_threshold=0.5,
                saturation=0.5,
            )

    def test_rejects_an_aversion_vector_that_does_not_match_the_cue_field(self):
        """Aversion and signature index the same space. A mismatch would silently weight channel
        k by aversion k+1, which no test would catch by accident.
        """
        world = FearWorld(grid=21)
        mismatched = FearConfig(
            weight=1.0,
            scent_acuity_gene="scent_acuity",
            aversion_genes=(AVERSION_GENES[0][:1],),
            detection_threshold=0.01,
            saturation=1.0,
        )

        with pytest.raises(ValueError):
            world.fear(mismatched)


class TestAllFiveDrivesCompeting:
    """#22's "done when": the full authored set resolves synthetic creatures to different actions
    from their state alone, and the reason is recoverable afterwards.
    """

    def register_all(self, world):
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
        world.behaviour.register(world.fear())
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

    def test_a_hungry_creature_next_to_a_predator_flees_instead_of_feeding(self):
        """Fear outscoring hunger in a creature that is *also* hungry is the case the whole
        utility contest exists for — a fixed priority order could not express it.

        Deliberately hungry rather than starving: at zero energy hunger saturates at exactly the
        same 1.0 fear does, and the tie falls to registration order. That is correct behaviour for
        equal weights, not a defect — it is the tuning table's job to decide whether a starving
        animal risks a predator, which is what makes these weights genes in #23.
        """
        world = FearWorld(grid=21, capacity=16)
        prey = world.spawn_as(
            0,
            1,
            x=np.float32([10.0]),
            y=np.float32([10.0]),
            energy=np.float32([30.0]),
            health=np.float32([1.0]),
        )
        predator = world.spawn_as(0, 3, x=np.float32([10.0] * 3), y=np.float32([10.0] * 3))
        world.genetics.set_genes(prey, gene_rows(timid(500.0)))
        world.genetics.set_genes(predator, gene_rows(*[dangerous()] * 3))
        world.scent.rebuild(prey | predator)
        self.register_all(world)

        world.behaviour.score(prey)

        assert world.behaviour.drive_names == ("hunger", "thirst", "fear", "lust", "fatigue")
        assert world.behaviour.driven_by("fear", prey) == prey
        breakdown = world.behaviour.breakdown(prey)
        assert breakdown["fear"][0] > breakdown["hunger"][0]

    def test_the_same_creature_feeds_once_the_predator_is_gone(self):
        """Same genes, same energy, same everything but the threat — so the change in action is
        attributable to the world rather than to the animal.
        """
        world = FearWorld(grid=21, capacity=16)
        prey = world.spawn_as(
            0,
            1,
            x=np.float32([10.0]),
            y=np.float32([10.0]),
            energy=np.float32([30.0]),
            health=np.float32([1.0]),
        )
        world.genetics.set_genes(prey, gene_rows(timid(500.0)))
        world.scent.rebuild(prey)
        self.register_all(world)

        world.behaviour.score(prey)

        assert world.behaviour.driven_by("hunger", prey) == prey


class TestTwoAversionDirections:
    """Why a creature carries more than one aversion direction (CLAUDE.md §2.5).

    A single direction pointed at two unrelated threats also fires at anything whose signature is
    a *blend* of them — a harmless creature smelling halfway between a wolf and an eagle. Two
    directions fear the two independently, so the blend is only half of each.
    """

    WOLF = {"signature_0": 1.0}
    BLEND = {"signature_0": 0.5, "signature_2": 0.5}

    def fear_of(self, threat_signature, prey_aversion, config, acuity=5.0):
        world = FearWorld(grid=21)
        prey = world.spawn_as(0, 1, x=np.float32([10.0]), y=np.float32([10.0]))
        threat = world.spawn_as(0, 1, x=np.float32([11.0]), y=np.float32([10.0]))
        world.genetics.set_genes(
            prey, gene_rows({"scent_acuity": acuity, **prey_aversion})
        )
        world.genetics.set_genes(
            threat, gene_rows({"scent_emission": 1.0, **threat_signature})
        )
        world.scent.rebuild(prey | threat)
        return world.fear(config).score(prey)[0]

    def test_one_direction_cannot_tell_a_blend_from_the_real_thing(self):
        """The limitation the second direction exists to remove."""
        single = FearConfig(
            weight=1.0,
            scent_acuity_gene="scent_acuity",
            aversion_genes=(AVERSION_GENES[0],),
            detection_threshold=0.01,
            saturation=1.0,
        )
        # One direction pointed at both wolf (channel 0) and eagle (channel 2).
        pointed_at_both = {"aversion0_0": 1.0, "aversion0_2": 1.0}

        wolf = self.fear_of(self.WOLF, pointed_at_both, single)
        blend = self.fear_of(self.BLEND, pointed_at_both, single)

        assert wolf > 0.0
        assert blend == pytest.approx(wolf, rel=1e-5)

    def test_two_directions_rank_a_blend_below_the_real_thing(self):
        """Wolf on one direction, eagle on the other. A creature that is half of each trips each
        direction halfway, which noisy-OR combines to less than either threat outright.
        """
        pointed_separately = {"aversion0_0": 1.0, "aversion1_2": 1.0}

        wolf = self.fear_of(self.WOLF, pointed_separately, FEAR_CONFIG)
        blend = self.fear_of(self.BLEND, pointed_separately, FEAR_CONFIG)

        assert wolf > 0.0
        assert blend < wolf

    def test_an_unused_second_direction_contributes_nothing(self):
        """A creature that only ever fears one thing leaves the second direction near zero, and it
        must then cost it nothing — otherwise carrying the capacity would itself be a hazard.
        """
        one_thing = {"aversion0_0": 1.0}
        both_named = {"aversion0_0": 1.0, "aversion1_1": 0.0}

        assert self.fear_of(self.WOLF, one_thing, FEAR_CONFIG) == pytest.approx(
            self.fear_of(self.WOLF, both_named, FEAR_CONFIG)
        )
