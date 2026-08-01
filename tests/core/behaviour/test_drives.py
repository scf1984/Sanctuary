from dataclasses import replace

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
from core.behaviour.exertion import Exertion, ExertionConfig
from core.behaviour.service import Behaviour, BehaviourConfig
from core.ecology.cues import CueField, CueFieldConfig, Scent, ScentGenes
from core.ecology.metabolism import Metabolism, MetabolismConfig
from core.ecology.plants import Plants, PlantsConfig
from core.world.diffusion import DiffusionConfig
from core.ecology.service import Ecology
from core.entities.store import EntityStore
from core.genetics.expression import GeneticsConfig
from core.genetics.service import Genetics
from core.genetics.species import SpeciesRegistry
from core.selection import Selection
from core.services import ColumnRegistry
from core.world.climate import Climate, ClimateConfig
from core.world.terrain import Terrain
from core.world.water import Water

from tests.support.genes import gene_registry
from tests.support.plants import plant_field

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
    "mutability",
    "choice_temperature",
    "commitment",
    "maturity_age",
    "hunger_weight",
    "thirst_weight",
    "fear_weight",
    "lust_weight",
    "fatigue_weight",
)
GENE_REGISTRY = gene_registry(GENE_NAMES, {"size": 2.0, "speed": 3.0, "sight": 1.0, "insulation": 1.0, "scent_acuity": 0.5})
SCENT_GENES = ScentGenes(emission_gene="scent_emission", signature_genes=SIGNATURE_GENES)

# Fatigue grades travelling options rather than vetoing them (#207). These are the shipped world's
# figures; nothing below asserts on them, and `TestFatigueGradesTravel` states its own.
TRAVEL_EFFORT = 0.25
CLIMB_TOLERANCE = 4.0


# Cue space is signed: a signature is a position in it and an aversion is a direction through it, so
# the sign is information in both (#104). Everything else here is a quantity.
CUE_GENES = (*SIGNATURE_GENES, *FLAT_AVERSION)
GENETICS_CONFIG = GeneticsConfig(
    mutability_gene="mutability",
    drift_margin=2.0,
)

METABOLISM_CONFIG = MetabolismConfig(
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
    forage_diffusion=DiffusionConfig(range=4.0, climb_penalty=0.5),
)

HUNGER_CONFIG = HungerConfig(
    weight_gene="hunger_weight",
    satiation_energy=100.0,
    # Low enough that an ordinary sight gene notices an ordinary meadow, high enough that a
    # near-blind animal does not. `test_sight_gates_what_a_forager_notices` is what pins it.
    detection_threshold=1.0,
    sight_gene="sight",
)


class World:
    """A flat, uniform-temperature world with every service #22's drives read from.

    Flat terrain with a zero latitude gradient makes `temperature_at` return `temperature`
    everywhere, so a test that is not about climate never has to care where it put its entities.
    """

    def __init__(
        self,
        capacity=8,
        temperature=20.0,
        grid=9,
        cell_size=1.0,
        heights=None,
        forage_range=4.0,
        climb_penalty=0.5,
        diffusion_range=3.0,
    ):
        self.store = EntityStore(initial_capacity=capacity, n_drives=5, n_genes=len(GENE_NAMES))
        self.columns = ColumnRegistry()
        self.genes = GENE_REGISTRY
        self.species = SpeciesRegistry(self.genes.vocabulary)
        self.genetics = Genetics(
            self.store, self.columns, self.species, self.genes, GENETICS_CONFIG
        )
        if heights is None:
            heights = np.zeros((grid, grid), dtype=np.float32)
        self.terrain = Terrain(heights, cell_size=cell_size)
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
        self.plants = Plants(
            self.terrain,
            self.climate,
            self.water,
            replace(
                PLANTS_CONFIG,
                forage_diffusion=DiffusionConfig(
                    range=forage_range, climb_penalty=climb_penalty
                ),
            ),
        )
        self.ecology = Ecology(
            self.store,
            self.columns,
            self.genetics,
            self.climate,
            Metabolism(self.genes, METABOLISM_CONFIG),
            plant_field(self.terrain, self.climate),
        )
        self.behaviour = Behaviour(
            self.store,
            self.columns,
            self.genetics,
            GENE_REGISTRY,
            self.terrain,
            BehaviourConfig(
                n_candidates=8,
                look_ahead=2.0,
                commitment_gene="commitment",
                choice_temperature_gene="choice_temperature",
            ),
        )
        # Fatigue reads exertion as well as health (#107). These tests exercise the health term and
        # the drive contest, so nothing below moves; test_exertion.py covers the other term.
        self.exertion = Exertion(self.store, self.columns, ExertionConfig(recovery_rate=0.5))
        self.cue_field = CueField(
            self.terrain, CHANNELS, CueFieldConfig(diffusion_range=diffusion_range)
        )
        self.scent = Scent(
            self.store, self.genetics, self.cue_field, self.genes, SCENT_GENES
        )
        self.species_id = self.species.register(GENE_NAMES)

    def spawn(self, n, **columns):
        """Allocate `n` entities of the single registered species and return their Selection.

        Drive weights are genes now (#23), so an entity spawned with a zeroed gene row wants
        nothing at all. These fixtures were written against a scalar weight of 1.0, so that is what
        an unspecified weight gene means here.
        """
        columns.setdefault("species_id", np.full(n, self.species_id, dtype=np.int32))
        if "genes" not in columns:
            columns["genes"] = gene_rows(*[{}] * n)
        ids = self.store.allocate(n, **columns)
        rows = [self.store._id_to_row[i] for i in ids.tolist()]
        return Selection.from_indices(np.array(rows, dtype=np.int64), self.store.capacity)


def gene_rows(*rows):
    """Stack gene rows given as {name: value} dicts into a (n, n_genes) float32 matrix.

    Drive weights default to 1.0 rather than 0: they are genes now (#23) and a zero weight is an
    animal that wants nothing, which is not the neutral starting point these fixtures assume.
    """
    matrix = np.zeros((len(rows), len(GENE_NAMES)), dtype=np.float32)
    for drive in ("hunger", "thirst", "fear", "lust", "fatigue"):
        matrix[:, GENE_NAMES.index(f"{drive}_weight")] = 1.0
    for i, genes in enumerate(rows):
        for name, value in genes.items():
            matrix[i, GENE_NAMES.index(name)] = value
    return matrix


class TestHungerScore:
    def test_a_full_pool_wants_nothing_and_an_empty_one_wants_everything(self):
        world = World()
        selection = world.spawn(3, energy=np.array([100.0, 50.0, 0.0], dtype=np.float32))
        hunger = Hunger(
            world.store, world.ecology, world.genetics, world.plants, world.genes,
            HUNGER_CONFIG,
        )

        assert hunger.urgency(selection) == pytest.approx([0.0, 0.5, 1.0])

    def test_energy_above_satiation_does_not_produce_negative_hunger(self):
        """A drive scoring below zero would still lose every contest, but it would also make
        "zero means no pull" untrue and break the no-active-drive reading in the service.
        """
        world = World()
        selection = world.spawn(1, energy=np.array([500.0], dtype=np.float32))
        hunger = Hunger(
            world.store, world.ecology, world.genetics, world.plants, world.genes,
            HUNGER_CONFIG,
        )

        assert hunger.urgency(selection) == pytest.approx([0.0])

    def test_weight_scales_the_whole_drive_and_is_per_entity(self):
        """The weight is a gene now (#23), so two equally starving animals can want food to
        different degrees — which is the whole mechanism by which temperament evolves."""
        world = World()
        selection = world.spawn(2, energy=np.array([0.0, 0.0], dtype=np.float32))
        world.genetics.set_genes(
            selection, gene_rows({"hunger_weight": 3.0}, {"hunger_weight": 1.0})
        )
        config = HungerConfig(
            weight_gene="hunger_weight",
            satiation_energy=100.0,
            detection_threshold=1.0,
            sight_gene="sight",
        )
        hunger = Hunger(
            world.store, world.ecology, world.genetics, world.plants, world.genes, config
        )

        assert hunger.urgency(selection) == pytest.approx([3.0, 1.0])


EAST, WEST, NULL = 0, 1, 2


def options_at(world, selection, reach=2.0):
    """(x, y) for an east option, a west option and the null option, each (n, 3) float64.

    `Behaviour` samples candidate positions once per entity and hands the same pair to every drive
    (#114), so a drive test that builds them itself is exercising the real contract rather than a
    stand-in for one. Clipped into the world exactly as `candidate_positions` clips them, because a
    heading near the edge points out of it and the fields a drive samples raise outside their bounds.
    """
    mask = selection.to_mask()
    x = world.store.x[mask].astype(np.float64)[:, None]
    y = world.store.y[mask].astype(np.float64)[:, None]
    offsets = np.array([reach, -reach, 0.0])
    return (
        np.clip(x + offsets, 0.0, world.terrain.world_width),
        np.clip(y + np.zeros(3), 0.0, world.terrain.world_height),
    )


class TestHungerAppeal:
    """The rule #93 settles, asked of each candidate option rather than of a gradient (#114).

    `appeal` reads the diffused plant field at the point each option would take the animal toward,
    so these assert which of two opposed options a grazer rates higher. That is the same claim the
    gradient made — which way is better from here — asked at the resolution the option set has, and
    it is still a claim no argmax over candidate patches could make, since a distance discount
    alone cannot see the ground in between.

    Every test here offers three options: east, west, and staying put, in that column order.
    """

    def _hunger(self, world, config=None):
        # The forage field is tick state with one writer now (#170), so a test that plants biomass
        # by hand has to run the step before anything can smell it. That is the point of the
        # change: nothing invalidates the field, so the tick order is what keeps it fresh.
        world.plants.rebuild_forage()
        return Hunger(
            world.store,
            world.ecology,
            world.genetics,
            world.plants,
            world.genes,
            config or HUNGER_CONFIG,
        )

    def _appeal(self, world, selection, reach=2.0, config=None):
        """(n, 3) appeal for an east option, a west option and the null option, at `reach` units."""
        x, y = options_at(world, selection, reach)
        return self._hunger(world, config).appeal(selection, x, y)

    def test_a_grazer_rates_the_option_toward_the_only_food_there_is_highest(self):
        world = World()
        selection = world.spawn(
            1, x=np.array([4.0], dtype=np.float32), y=np.array([4.0], dtype=np.float32)
        )
        world.genetics.set_genes(selection, gene_rows({"sight": 3.0}))
        world.plants.biomass[:] = 0.0
        world.plants.biomass[4, 6] = 50.0

        appeal = self._appeal(world, selection)

        assert appeal[0, EAST] > appeal[0, WEST]
        assert appeal[0, EAST] > appeal[0, NULL]

    def test_a_near_patch_beats_a_richer_far_one(self):
        """Keeping grazers local is what strips ground bare before they move on, which is the local
        grazing pressure the field model of #18 exists to express. The diffusion range sets it now,
        so the discount lives in one coefficient instead of two — a range well short of the gap
        between the patches is what "local" means."""
        world = World(forage_range=1.0)
        selection = world.spawn(
            1, x=np.array([4.0], dtype=np.float32), y=np.array([4.0], dtype=np.float32)
        )
        world.genetics.set_genes(selection, gene_rows({"sight": 4.0}))
        world.plants.biomass[:] = 0.0
        world.plants.biomass[4, 3] = 10.0  # 1 unit west
        world.plants.biomass[4, 8] = 25.0  # 4 units east, 2.5x the standing crop

        appeal = self._appeal(world, selection)

        assert appeal[0, WEST] > appeal[0, EAST]

    def test_a_wider_ranging_field_sends_a_grazer_to_the_richer_patch(self):
        """The same two patches in a world whose forage field carries further: the richer one now
        outweighs the discount. This is `forage_reluctance`'s old job, done by the field's range."""
        world = World(forage_range=4.0)
        selection = world.spawn(
            1, x=np.array([4.0], dtype=np.float32), y=np.array([4.0], dtype=np.float32)
        )
        world.genetics.set_genes(selection, gene_rows({"sight": 4.0}))
        world.plants.biomass[:] = 0.0
        world.plants.biomass[4, 3] = 10.0
        world.plants.biomass[4, 8] = 25.0

        appeal = self._appeal(world, selection)

        assert appeal[0, EAST] > appeal[0, WEST]

    def test_food_behind_a_ridge_loses_to_food_on_open_ground(self):
        """What the discrete contract could not express (#93): two equal meadows, equally distant,
        one across a climb. Sight range alone cannot tell them apart — the cost of the walk can.

        Probed at one world unit rather than at the two the rest of this class uses, because the
        crest is two units east and a candidate landing *on or beyond* a barrier reads the far
        side's abundance without paying for the climb to it. That is the same failure #113 fixed
        for movement — a stride priced by its endpoints nets a descent against a climb — and it is
        filed as #169 rather than papered over here. The field itself is correct: along this row it
        reads 0.946 one unit west of the animal against 0.819 one unit east, so the mechanism #93
        settled still points away from the ridge.
        """
        heights = np.zeros((9, 9), dtype=np.float32)
        heights[:, 6] = 40.0
        world = World(heights=heights, climb_penalty=1.0)
        selection = world.spawn(
            1, x=np.array([4.0], dtype=np.float32), y=np.array([4.0], dtype=np.float32)
        )
        world.genetics.set_genes(selection, gene_rows({"sight": 8.0}))
        world.plants.biomass[:] = 0.0
        world.plants.biomass[4, 2] = 40.0  # open ground, 2 west
        world.plants.biomass[4, 8] = 40.0  # 2 east of the ridge, the same distance away

        appeal = self._appeal(world, selection, reach=1.0)

        assert appeal[0, WEST] > appeal[0, EAST], "the grazer preferred the meadow behind the ridge"

    def test_sight_gates_what_a_forager_notices(self):
        """Perception a forager could not pay for would leave sight range charged by the metabolic
        budget while buying nothing (§2.5). One field serves everyone, so acuity is a threshold on
        what is sampled rather than a radius — the same rule scent already uses.
        """
        world = World()
        selection = world.spawn(
            2,
            x=np.array([4.0, 4.0], dtype=np.float32),
            y=np.array([4.0, 4.0], dtype=np.float32),
        )
        world.genetics.set_genes(selection, gene_rows({"sight": 0.02}, {"sight": 40.0}))
        world.plants.biomass[:] = 0.0
        world.plants.biomass[4, 7] = 50.0

        appeal = self._appeal(world, selection)

        assert appeal[0] == pytest.approx(np.zeros(3))
        assert appeal[1, EAST] > appeal[1, WEST]

    def test_an_animal_that_can_detect_no_food_is_indifferent_to_every_option(self):
        """All zeros, not a flat non-zero preference — the distinction #114 turns on.

        A drive contributing zero to every option drops out of the sum entirely, so the null option
        is free to win on fatigue alone and the animal rests. A flat *non-zero* appeal would instead
        add a constant to every candidate and to the null option alike, which changes no ranking but
        also never lets hunger stop steering. That the two look alike in a single-drive contest is
        exactly why this asserts the values rather than the choice.
        """
        world = World()
        selection = world.spawn(
            1, x=np.array([4.0], dtype=np.float32), y=np.array([4.0], dtype=np.float32)
        )
        world.genetics.set_genes(selection, gene_rows({"sight": 3.0}))
        world.plants.biomass[:] = 0.0

        assert self._appeal(world, selection) == pytest.approx(np.zeros((1, 3)))

    def test_an_unexpressed_sight_gene_leaves_a_forager_grazing_underfoot(self):
        """Expression, not genotype, is what a forager sees with — the same rule that makes an
        unexpressed gene cost nothing (#17). A blind animal detects nothing and stays put, which
        still lets `graze` feed it where it stands.
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
        world.plants.biomass[4, 8] = 90.0

        assert self._appeal(world, selection) == pytest.approx(np.zeros((1, 3)))

    def test_appeal_is_normalised_to_the_animals_own_best_option(self):
        """Every drive's appeal is summed against every other's, so they must share a scale.

        Normalising by the animal's own best option is what puts hunger on the same [0, 1] range
        fatigue and fear already occupy, without any drive knowing what the others return.
        """
        world = World()
        selection = world.spawn(
            1, x=np.array([4.0], dtype=np.float32), y=np.array([4.0], dtype=np.float32)
        )
        world.genetics.set_genes(selection, gene_rows({"sight": 3.0}))
        world.plants.biomass[:] = 0.0
        world.plants.biomass[4, 6] = 50.0

        appeal = self._appeal(world, selection)

        assert appeal.max() == pytest.approx(1.0)
        assert (appeal >= 0.0).all()

    def test_a_forager_on_the_world_edge_is_answered_without_running_off_the_grid(self):
        """`Movement._landing` guarantees animals land exactly on the boundary, so this is a
        position that certainly occurs rather than an edge case, and a candidate heading outward
        from it is clipped back onto the boundary by `Behaviour.candidate_positions`.

        It asserts only that the answer is finite and in range. It deliberately does *not* assert
        that a cornered forager prefers nearby food: within about one diffusion range of the
        boundary the field reads rich, because a walk starting at a corner is confined and revisits
        nearby source more often than an interior walk does. That is recorded as a known limitation
        of the operator rather than papered over here.
        """
        world = World(forage_range=1.0)
        selection = world.spawn(
            1, x=np.array([0.0], dtype=np.float32), y=np.array([0.0], dtype=np.float32)
        )
        world.genetics.set_genes(selection, gene_rows({"sight": 3.0}))
        world.plants.biomass[:] = 0.0
        world.plants.biomass[0, 2] = 20.0

        appeal = self._appeal(world, selection)

        assert np.isfinite(appeal).all()
        assert ((appeal >= 0.0) & (appeal <= 1.0)).all()

    def test_every_forager_is_answered_in_one_call(self):
        world = World(capacity=16)
        selection = world.spawn(
            6,
            x=np.full(6, 4.0, dtype=np.float32),
            y=np.linspace(2.0, 6.0, 6).astype(np.float32),
        )
        world.genetics.set_genes(selection, gene_rows(*[{"sight": 5.0}] * 6))
        world.plants.biomass[:] = 0.0
        world.plants.biomass[:, 7] = 30.0

        appeal = self._appeal(world, selection)

        assert appeal.shape == (6, 3)
        assert (appeal[:, EAST] > appeal[:, WEST]).all()


class TestThirst:
    def test_thirst_rises_with_ambient_heat(self):
        config = ThirstConfig(weight_gene="thirst_weight", onset_temperature=20.0, saturation_temperature=40.0)

        scores = []
        for temperature in (15.0, 30.0, 50.0):
            world = World(temperature=temperature)
            selection = world.spawn(1)
            scores.append(Thirst(world.store, world.climate, world.genetics, world.genes, config).urgency(selection)[0])

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
        config = ThirstConfig(weight_gene="thirst_weight", onset_temperature=20.0, saturation_temperature=40.0)

        scores = Thirst(world.store, world.climate, world.genetics, world.genes, config).urgency(selection)

        # y=0 sits at 40 degC (saturated); y=20 sits at 20 degC, exactly at onset.
        assert scores == pytest.approx([1.0, 0.0])

    def test_saturation_must_exceed_onset(self):
        with pytest.raises(ValueError):
            ThirstConfig(weight_gene="thirst_weight", onset_temperature=30.0, saturation_temperature=30.0)

    def test_thirst_rates_every_option_alike_because_nothing_can_find_water(self):
        """A drive with no perception is *indifferent*, which is the whole of #126's fix.

        Before #114 a thirsty animal in a world with no drinking mechanic won the argmax and then
        stood still, and nothing said so — the first assembled world had all forty founders wanting
        water and not one moved for the entire run. Now thirst still registers its appetite in the
        breakdown, but a flat appeal shifts no ranking, so whichever drive *can* perceive something
        decides. #156 replaces this with a reading of `Water.depth` at each candidate.
        """
        world = World(temperature=30.0)
        selection = world.spawn(2, x=np.float32([4.0, 4.0]), y=np.float32([4.0, 4.0]))
        thirst = Thirst(
            world.store,
            world.climate,
            world.genetics,
            world.genes,
            ThirstConfig(weight_gene="thirst_weight", onset_temperature=20.0, saturation_temperature=40.0),
        )
        x, y = options_at(world, selection)

        appeal = thirst.appeal(selection, x, y)

        assert appeal.shape == (2, 3)
        assert appeal == pytest.approx(np.ones((2, 3)))
        # Thirsty — so the flatness is a statement about direction, not about the appetite.
        assert thirst.urgency(selection) == pytest.approx([0.5, 0.5])


class TestLust:
    def test_an_immature_animal_wants_no_mate_however_fat(self):
        world = World()
        selection = world.spawn(1, energy=np.array([100.0], dtype=np.float32))
        world.genetics.set_genes(selection, gene_rows({"maturity_age": 100.0}))
        world.store.age[selection.to_indices()] = 5
        config = LustConfig(
            weight_gene="lust_weight",
            maturity_gene="maturity_age",
            scent_acuity_gene="scent_acuity",
            detection_threshold=1e-4,
            breeding_energy=20.0,
            abundant_energy=70.0,
        )

        assert Lust(world.store, world.ecology, world.genetics, world.scent, world.genes, config).urgency(selection) == pytest.approx([0.0])

    def test_a_mature_animal_below_breeding_energy_wants_no_mate(self):
        """Gestation charges upkeep like any other trait (§2.5); wanting what you cannot afford
        would select for breeding yourself to death.
        """
        world = World()
        selection = world.spawn(1, energy=np.array([10.0], dtype=np.float32))
        world.genetics.set_genes(selection, gene_rows({"maturity_age": 100.0}))
        world.store.age[selection.to_indices()] = 200
        config = LustConfig(
            weight_gene="lust_weight",
            maturity_gene="maturity_age",
            scent_acuity_gene="scent_acuity",
            detection_threshold=1e-4,
            breeding_energy=20.0,
            abundant_energy=70.0,
        )

        assert Lust(world.store, world.ecology, world.genetics, world.scent, world.genes, config).urgency(selection) == pytest.approx([0.0])

    def test_lust_rises_with_energy_above_the_breeding_floor(self):
        world = World()
        selection = world.spawn(3, energy=np.array([20.0, 45.0, 90.0], dtype=np.float32))
        world.genetics.set_genes(selection, gene_rows(*[{"maturity_age": 100.0}] * 3))
        world.store.age[selection.to_indices()] = 200
        config = LustConfig(
            weight_gene="lust_weight",
            maturity_gene="maturity_age",
            scent_acuity_gene="scent_acuity",
            detection_threshold=1e-4,
            breeding_energy=20.0,
            abundant_energy=70.0,
        )

        scores = Lust(world.store, world.ecology, world.genetics, world.scent, world.genes, config).urgency(selection)

        # At the floor, halfway to abundance, and clamped above it.
        assert scores == pytest.approx([0.0, 0.5, 1.0])

    def test_maturity_is_counted_in_ticks(self):
        """The tick counter is the only clock (CLAUDE.md §2.1) — maturity is a row of `age`."""
        world = World()
        selection = world.spawn(2, energy=np.array([70.0, 70.0], dtype=np.float32))
        world.genetics.set_genes(selection, gene_rows(*[{"maturity_age": 100.0}] * 2))
        world.store.age[selection.to_indices()] = [99, 100]
        config = LustConfig(
            weight_gene="lust_weight",
            maturity_gene="maturity_age",
            scent_acuity_gene="scent_acuity",
            detection_threshold=1e-4,
            breeding_energy=20.0,
            abundant_energy=70.0,
        )

        assert Lust(world.store, world.ecology, world.genetics, world.scent, world.genes, config).urgency(selection) == pytest.approx(
            [0.0, 1.0]
        )

    def lust(self, world, **overrides):
        params = dict(
            weight_gene="lust_weight",
            maturity_gene="maturity_age",
            scent_acuity_gene="scent_acuity",
            detection_threshold=1e-4,
            breeding_energy=20.0,
            abundant_energy=70.0,
        )
        params.update(overrides)
        return Lust(
            world.store, world.ecology, world.genetics, world.scent, world.genes, LustConfig(**params)
        )

    def ready(self, world, n, **columns):
        selection = world.spawn(n, energy=np.full(n, 70.0, dtype=np.float32), **columns)
        world.store.age[selection.to_indices()] = 200
        return selection

    def test_an_animal_alone_in_the_world_prefers_nowhere(self):
        """All zeros rather than a flat score, so lust drops out of the utility sum entirely and
        the null option can win — the same reason `Hunger.appeal` normalises. An animal that senses
        nobody rests instead of marching whichever way the numerical noise leaned.
        """
        world = World()
        selection = self.ready(world, 1, x=np.float32([4.0]), y=np.float32([4.0]))
        world.genetics.set_genes(selection, gene_rows({"scent_emission": 1.0, "signature_0": 1.0,
                                                       "scent_acuity": 1.0}))
        world.scent.rebuild(selection)
        x, y = options_at(world, selection)

        assert self.lust(world).appeal(selection, x, y) == pytest.approx(np.zeros((1, 3)))

    def test_it_does_not_smell_itself_and_stay_put(self):
        """Lust's vector is the animal's *own* signature (§2.5), so its own plume is the strongest
        match to it anywhere in the world. Without exclusion at candidates this would be a rule
        that says never move (#188).
        """
        world = World()
        selection = self.ready(world, 1, x=np.float32([4.0]), y=np.float32([4.0]))
        world.genetics.set_genes(selection, gene_rows({"scent_emission": 5.0, "signature_0": 1.0,
                                                       "scent_acuity": 1.0}))
        world.scent.rebuild(selection)
        x, y = options_at(world, selection)

        # Its own cell is among the candidates and must not be the winner — nothing is.
        assert self.lust(world).appeal(selection, x, y).max() == pytest.approx(0.0)

    def test_it_is_drawn_toward_something_that_smells_like_it(self):
        world = World(grid=17)
        pair = self.ready(
            world, 2, x=np.float32([4.0, 10.0]), y=np.float32([8.0, 8.0])
        )
        world.genetics.set_genes(
            pair,
            gene_rows(
                {"scent_emission": 4.0, "signature_0": 1.0, "scent_acuity": 1.0},
                {"scent_emission": 4.0, "signature_0": 1.0, "scent_acuity": 1.0},
            ),
        )
        world.scent.rebuild(pair)
        seeker = Selection.from_indices(pair.to_indices()[:1], world.store.capacity)

        # Two candidates for the first animal: one step toward its partner, one step away.
        toward = np.float32([[5.0, 3.0]])
        same_y = np.float32([[8.0, 8.0]])
        appeal = self.lust(world).appeal(seeker, toward, same_y)

        assert appeal[0, 0] > appeal[0, 1]

    def test_a_stranger_smelling_of_something_else_is_no_draw(self):
        """The vector is the searcher's own signature, so attraction tracks similarity — which is
        what makes it follow speciation for free, with nothing told that a split happened."""
        world = World(grid=17)
        pair = self.ready(
            world, 2, x=np.float32([4.0, 10.0]), y=np.float32([8.0, 8.0])
        )
        world.genetics.set_genes(
            pair,
            gene_rows(
                {"scent_emission": 4.0, "signature_0": 1.0, "scent_acuity": 1.0},
                {"scent_emission": 4.0, "signature_1": 1.0, "scent_acuity": 1.0},
            ),
        )
        world.scent.rebuild(pair)
        seeker = Selection.from_indices(pair.to_indices()[:1], world.store.capacity)

        appeal = self.lust(world).appeal(
            seeker, np.float32([[5.0, 3.0]]), np.float32([[8.0, 8.0]])
        )

        assert appeal == pytest.approx(np.zeros((1, 2)))

    def test_urgency_is_unchanged_by_any_of_this(self):
        """Appeal says which way; urgency says how badly. They stay separate."""
        world = World()
        selection = self.ready(world, 1)

        assert self.lust(world).urgency(selection) == pytest.approx([1.0])


class TestFatigue:
    def test_fatigue_is_the_health_deficit(self):
        world = World()
        selection = world.spawn(3, health=np.array([1.0, 0.25, 0.0], dtype=np.float32))

        scores = Fatigue(world.store, world.exertion, world.genetics, world.terrain, world.genes, FatigueConfig(
                weight_gene="fatigue_weight",
                exertion_saturation=1.0,
                travel_effort=TRAVEL_EFFORT,
                climb_tolerance=CLIMB_TOLERANCE,
            )).urgency(selection)

        assert scores == pytest.approx([0.0, 0.75, 1.0])

    def test_a_negative_weight_gene_cannot_invert_the_drive(self):
        """A negative weight would invert a drive — the worse the injury, the less it wants rest.

        `LustConfig` and friends used to reject one at construction. They no longer can, because a
        weight is a gene now and genes drift freely (#23). The check is *deleted* rather than moved:
        the weight is read as a magnitude, so a lineage whose stored value has drifted below zero
        expresses a positive weight and the inversion is unrepresentable (§8.7, the same move #111
        made for gene costs).
        """
        world = World()
        selection = world.spawn(2, health=np.float32([0.5, 0.5]))
        world.genetics.set_genes(
            selection, gene_rows({"fatigue_weight": -2.0}, {"fatigue_weight": 2.0})
        )
        fatigue = Fatigue(
            world.store,
            world.exertion,
            world.genetics,
            world.terrain,
            world.genes,
            FatigueConfig(
                weight_gene="fatigue_weight",
                exertion_saturation=1.0,
                travel_effort=TRAVEL_EFFORT,
                climb_tolerance=CLIMB_TOLERANCE,
            ),
        )

        scores = fatigue.urgency(selection)
        assert (scores >= 0.0).all()
        assert scores[0] == pytest.approx(scores[1])

    def test_fatigue_prefers_resting_without_vetoing_travel(self):
        """Rest needs no mode, no flag and no state column — it is an option in the same contest.

        An animal that picks the null option proposes no displacement, so `Movement.step` prices a
        step of zero and it pays nothing. That is what makes exertion recover (#107) with nothing
        anywhere branching on "is resting".

        **Travelling scores `1 − travel_effort`, not zero, and that is #207.** Scoring it at zero
        made this drive's spread across options equal to its entire urgency — the largest voice in
        the contest, spent on one bit of information — so hunger, which ranks the food correctly
        0.998 of the time, never once decided a direction. Resting still wins here; it just no
        longer wins by the whole width of the scale.
        """
        world = World()
        # Mid-world, so both travelling candidates are real: `options_at` clips into the world
        # exactly as `candidate_positions` does, and from the default corner the westward option
        # clips onto the animal's own position — which is a stay-put option and reads as one.
        selection = world.spawn(
            2, health=np.float32([0.5, 1.0]), x=np.float32([4.0, 4.0]), y=np.float32([4.0, 4.0])
        )
        fatigue = Fatigue(
            world.store,
            world.exertion,
            world.genetics,
            world.terrain,
            world.genes,
            FatigueConfig(
                weight_gene="fatigue_weight",
                exertion_saturation=1.0,
                travel_effort=TRAVEL_EFFORT,
                climb_tolerance=CLIMB_TOLERANCE,
            ),
        )
        x, y = options_at(world, selection)

        appeal = fatigue.appeal(selection, x, y)

        assert appeal[:, NULL] == pytest.approx([1.0, 1.0])
        assert appeal[:, :NULL] == pytest.approx(np.full((2, 2), 1.0 - TRAVEL_EFFORT))


# Cold enough that sampling is effectively the argmax: exp(-4) is about 0.018, so a utility gap of
# 0.3 becomes 16 in scaled units and swamps the Gumbel noise the softmax adds. Temperature is a gene
# precisely so a world can hold both this animal and an exploratory one (#114).
DECISIVE = {"sight": 5.0, "choice_temperature": -4.0}


def register_four(world):
    """Hunger, thirst, lust and fatigue against one world, at equal weight."""
    world.behaviour.register(
        Hunger(
            world.store, world.ecology, world.genetics, world.plants, world.genes, HUNGER_CONFIG
        )
    )
    world.behaviour.register(
        Thirst(
            world.store,
            world.climate,
            world.genetics,
            world.genes,
            ThirstConfig(weight_gene="thirst_weight", onset_temperature=25.0, saturation_temperature=40.0),
        )
    )
    world.behaviour.register(
        Lust(
            world.store,
            world.ecology,
            world.genetics,
            world.scent,
            world.genes,
            LustConfig(weight_gene="lust_weight",
            maturity_gene="maturity_age",
            scent_acuity_gene="scent_acuity",
            detection_threshold=1e-4,
            breeding_energy=20.0,
            abundant_energy=70.0,),
        )
    )
    world.behaviour.register(
        Fatigue(world.store, world.exertion, world.genetics, world.terrain, world.genes, FatigueConfig(
                weight_gene="fatigue_weight",
                exertion_saturation=1.0,
                travel_effort=TRAVEL_EFFORT,
                climb_tolerance=CLIMB_TOLERANCE,
            ))
    )


class TestDrivesCompeting:
    def test_a_starving_animal_forages_and_a_fed_injured_one_rests(self):
        """The whole point of #22 on a synthetic population, now readable as an *action*: the same
        registered set of drives resolves two animals to different behaviour from their state alone.

        Under #22 this could only be asserted as "which drive won". It is now asserted as what the
        animals actually do — one heads for the meadow, the other stays put — which is the claim
        that was always meant and that a winning-drive column could only stand in for.
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
        world.genetics.set_genes(selection, gene_rows(DECISIVE, DECISIVE))
        world.plants.biomass[:] = 0.0
        world.plants.biomass[4, 7] = 80.0
        register_four(world)

        world.behaviour.choose(selection, np.random.default_rng(0))

        moving = world.store.choice_moving[selection.to_indices()]
        assert moving[0], "the starving animal did not set off for the only meadow in the world"
        assert not moving[1], "the fed, injured animal did not rest"
        # And it went the right way: the meadow is due east, so the heading is within a quadrant
        # of it. The jitter means the exact angle is not reproducible, which is the point of it.
        assert np.cos(world.store.choice_heading[selection.to_indices()][0]) > 0.0

    def test_the_breakdown_explains_the_chosen_option(self):
        """"It rested because fatigue outweighed hunger" has to be recoverable from the store, not
        told as a story about it (CLAUDE.md §2.5, §3.3).

        The breakdown is each drive's contribution *to the option actually taken*, which is strictly
        more than #22's winner name: hunger reads 0 here not because the animal is fed — it is not,
        it wants food at 0.2 — but because there is nothing edible in any direction, so hunger has
        nothing to say about where to go. A winning-drive column could not express the difference.
        """
        world = World()
        selection = world.spawn(
            1,
            energy=np.array([80.0], dtype=np.float32),
            health=np.array([0.1], dtype=np.float32),
            # Mid-world: from the corner, half the candidate headings clip onto the animal's own
            # position, and a stay-put option that `Behaviour` records as moving would confuse
            # what this test is about.
            x=np.array([4.0], dtype=np.float32),
            y=np.array([4.0], dtype=np.float32),
        )
        world.genetics.set_genes(selection, gene_rows(DECISIVE))
        world.plants.biomass[:] = 0.0
        world.behaviour.register(
            Hunger(
                world.store, world.ecology, world.genetics, world.plants, world.genes,
                HUNGER_CONFIG,
            )
        )
        world.behaviour.register(Fatigue(world.store, world.exertion, world.genetics, world.terrain, world.genes, FatigueConfig(
                weight_gene="fatigue_weight",
                exertion_saturation=1.0,
                travel_effort=TRAVEL_EFFORT,
                climb_tolerance=CLIMB_TOLERANCE,
            )))

        world.behaviour.choose(selection, np.random.default_rng(0))
        breakdown = world.behaviour.breakdown(selection)

        # Fatigue no longer vetoes travel (#207), so the decisive animal may take either — what
        # matters is that the breakdown reports its contribution *to whatever it took*.
        rested = not world.store.choice_moving[selection.to_indices()][0]
        assert breakdown["fatigue"] == pytest.approx([0.9 if rested else 0.9 * (1.0 - TRAVEL_EFFORT)])
        assert breakdown["hunger"] == pytest.approx([0.0])



FEAR_CONFIG = FearConfig(
    weight_gene="fear_weight",
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

    def fear(self, config=FEAR_CONFIG):
        return Fear(self.store, self.genetics, self.scent, self.genes, config)

    def spawn_as(self, species_id, n, **columns):
        columns["species_id"] = np.full(n, species_id, dtype=np.int32)
        ids = self.store.allocate(n, **columns)
        rows = [self.store._id_to_row[i] for i in ids.tolist()]
        return Selection.from_indices(np.array(rows, dtype=np.int64), self.store.capacity)


def dangerous(emission=1.0):
    """Genes for a creature that smells of channel 0 and fears nothing."""
    return {"scent_emission": emission, "signature_0": 1.0}


def timid(acuity=50.0, **extra):
    """Genes for a creature that fears channel 0 and broadcasts nothing."""
    return {"scent_acuity": acuity, "aversion0_0": 1.0, **extra}


class TestFearScore:
    def test_a_creature_alone_in_the_world_fears_nothing(self):
        world = FearWorld(grid=21)
        prey = world.spawn_as(0, 1, x=np.float32([10.0]), y=np.float32([10.0]))
        world.genetics.set_genes(prey, gene_rows(timid()))
        world.scent.rebuild(prey)

        assert world.fear().urgency(prey) == pytest.approx([0.0])

    def test_a_nearby_source_of_a_feared_signature_is_feared(self):
        world = FearWorld(grid=21)
        prey = world.spawn_as(0, 1, x=np.float32([10.0]), y=np.float32([10.0]))
        predator = world.spawn_as(0, 1, x=np.float32([11.0]), y=np.float32([10.0]))
        world.genetics.set_genes(prey, gene_rows(timid()))
        world.genetics.set_genes(predator, gene_rows(dangerous()))
        world.scent.rebuild(prey | predator)

        assert world.fear().urgency(prey)[0] > 0.0

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

        assert world.fear().urgency(prey)[0] > 0.0
        assert world.fear().urgency(predator) == pytest.approx([0.0])

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

        assert world.fear().urgency(prey) == pytest.approx([0.0])

    def test_fear_falls_off_with_distance_from_the_source(self):
        world = FearWorld(grid=41)
        near = world.spawn_as(0, 1, x=np.float32([21.0]), y=np.float32([20.0]))
        far = world.spawn_as(0, 1, x=np.float32([27.0]), y=np.float32([20.0]))
        predator = world.spawn_as(0, 1, x=np.float32([20.0]), y=np.float32([20.0]))
        prey = near | far
        world.genetics.set_genes(prey, gene_rows(timid(), timid()))
        world.genetics.set_genes(predator, gene_rows(dangerous()))
        world.scent.rebuild(prey | predator)

        scores = world.fear().urgency(prey)

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
            scores.append(world.fear().urgency(prey)[0])

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
            scores.append(world.fear().urgency(prey)[0])

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

        scores = world.fear().urgency(prey)

        assert scores[0] > scores[1]

    def test_weight_scales_the_whole_drive(self):
        world = FearWorld(grid=21)
        prey = world.spawn_as(0, 1, x=np.float32([10.0]), y=np.float32([10.0]))
        predator = world.spawn_as(0, 1, x=np.float32([10.0]), y=np.float32([10.0]))
        world.genetics.set_genes(prey, gene_rows(timid(1000.0)))
        world.genetics.set_genes(predator, gene_rows(dangerous()))
        world.scent.rebuild(prey | predator)

        baseline = world.fear().urgency(prey)
        # Weight is a gene now (#23), so scaling it means changing the animal, not the world.
        world.genetics.set_genes(prey, gene_rows(timid(1000.0, fear_weight=3.0)))

        assert world.fear().urgency(prey) == pytest.approx(3.0 * baseline, rel=1e-5)


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

        assert world.fear().urgency(alone)[0] > 0.0

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

        assert world.fear().urgency(alone) == pytest.approx([0.0])

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

        assert world.fear().urgency(prey)[0] > 0.0

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
            scores.append(world.fear().urgency(prey)[0])

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

        assert world.fear().urgency(prey)[0] > 0.0


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

        scores = world.fear().urgency(prey)

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

        assert world.fear().urgency(prey) == pytest.approx([0.0])

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

        assert world.fear().urgency(prey) == pytest.approx([0.0])

    def test_detection_saturates_at_certainty(self):
        world = FearWorld(grid=21, capacity=16)
        prey = world.spawn_as(0, 1, x=np.float32([10.0]), y=np.float32([10.0]))
        predator = world.spawn_as(0, 8, x=np.float32([10.0] * 8), y=np.float32([10.0] * 8))
        world.genetics.set_genes(prey, gene_rows(timid(10_000.0)))
        world.genetics.set_genes(predator, gene_rows(*[dangerous()] * 8))
        world.scent.rebuild(prey | predator)

        assert world.fear().urgency(prey) == pytest.approx([1.0])


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

        assert fear.urgency(prey) == pytest.approx(fear._channels(prey)[0], rel=1e-6)

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
        scent_only = fear.urgency(prey)[0]

        # Stand in for #24's sight channel until it exists, to pin the composition rule now.
        smelled = fear._channels(prey)
        sight = np.full(len(prey), 0.5, dtype=np.float32)
        fear._channels = lambda selection: [*smelled, sight]
        both = fear.urgency(prey)[0]

        assert both > scent_only
        assert both <= 1.0
        assert both == pytest.approx(1.0 - (1.0 - scent_only) * 0.5, rel=1e-5)


class TestFearConfig:
    def test_rejects_a_zero_detection_threshold(self):
        with pytest.raises(ValueError):
            FearConfig(
                weight_gene="fear_weight",
                scent_acuity_gene="scent_acuity",
                aversion_genes=AVERSION_GENES,
                detection_threshold=0.0,
                saturation=1.0,
            )

    def test_rejects_saturation_at_or_below_the_threshold(self):
        with pytest.raises(ValueError):
            FearConfig(
                weight_gene="fear_weight",
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
            weight_gene="fear_weight",
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
                world.store, world.ecology, world.genetics, world.plants, world.genes,
                HUNGER_CONFIG,
            )
        )
        world.behaviour.register(
            Thirst(
                world.store,
                world.climate,
                world.genetics,
                world.genes,
                ThirstConfig(weight_gene="thirst_weight", onset_temperature=25.0, saturation_temperature=40.0),
            )
        )
        world.behaviour.register(world.fear())
        world.behaviour.register(
            Lust(
                world.store,
                world.ecology,
                world.genetics,
                world.scent,
                world.genes,
                LustConfig(
                    weight_gene="lust_weight",
            maturity_gene="maturity_age",
            scent_acuity_gene="scent_acuity",
            detection_threshold=1e-4,
            breeding_energy=20.0,
            abundant_energy=70.0,
                ),
            )
        )
        world.behaviour.register(Fatigue(world.store, world.exertion, world.genetics, world.terrain, world.genes, FatigueConfig(
                weight_gene="fatigue_weight",
                exertion_saturation=1.0,
                travel_effort=TRAVEL_EFFORT,
                climb_tolerance=CLIMB_TOLERANCE,
            )))

    def _terrified_and_hungry(self, with_predator):
        world = FearWorld(grid=21, capacity=16)
        prey = world.spawn_as(
            0,
            1,
            x=np.float32([10.0]),
            y=np.float32([10.0]),
            energy=np.float32([30.0]),
            health=np.float32([1.0]),
        )
        genes = dict(timid(500.0))
        genes.update(DECISIVE)
        world.genetics.set_genes(prey, gene_rows(genes))
        smellers = prey
        if with_predator:
            predator = world.spawn_as(0, 3, x=np.float32([10.0] * 3), y=np.float32([10.0] * 3))
            world.genetics.set_genes(predator, gene_rows(*[dangerous()] * 3))
            smellers = prey | predator
        world.scent.rebuild(smellers)
        world.plants.biomass[:] = 0.0
        world.plants.biomass[10, 14] = 80.0  # the only meadow, due east
        # The forage field is tick state with one writer (#170), so planting a meadow by hand
        # means running the step before anything can smell it.
        world.plants.rebuild_forage()
        self.register_all(world)
        return world, prey

    def test_a_hungry_creature_next_to_a_predator_is_more_afraid_than_hungry(self):
        """Fear outweighing hunger in a creature that is *also* hungry is the case the whole
        utility contest exists for — a fixed priority order could not express it.

        What it can no longer be asserted as is "it fled": fear has no direction yet, because
        reading the cue field at a candidate needs `sample_excluding_self` generalised to arbitrary
        points (see `Fear.appeal`). So the contest is visible in the breakdown rather than in the
        heading, and it is the breakdown that #114 makes the load-bearing surface.
        """
        world, prey = self._terrified_and_hungry(with_predator=True)

        world.behaviour.choose(prey, np.random.default_rng(0))

        assert world.behaviour.drive_names == ("hunger", "thirst", "fear", "lust", "fatigue")
        breakdown = world.behaviour.breakdown(prey)
        assert breakdown["fear"][0] > breakdown["hunger"][0]

    def test_terror_no_longer_freezes_an_animal_that_can_see_food(self):
        """The defect #114 exists to remove (#126), asserted directly.

        Under #22 fear won the argmax, movement acted only for hunger, and a frightened animal
        therefore stood still — a drive without a mechanic behind it could paralyse one that had
        one. Now fear contributes to every option equally, so it cannot shift a ranking: the same
        terrified animal still walks to the meadow it can see. Nothing about fear changed to make
        this true; it falls out of scoring options rather than entities.
        """
        world, prey = self._terrified_and_hungry(with_predator=True)

        world.behaviour.choose(prey, np.random.default_rng(0))

        row = prey.to_indices()[0]
        assert world.behaviour.breakdown(prey)["fear"][0] > 0.0, "the prey was not actually afraid"
        assert world.store.choice_moving[row]
        assert np.cos(world.store.choice_heading[row]) > 0.0

    def test_the_same_creature_is_unafraid_once_the_predator_is_gone(self):
        """Same genes, same energy, same everything but the threat — so the change is attributable
        to the world rather than to the animal.
        """
        world, prey = self._terrified_and_hungry(with_predator=False)

        world.behaviour.choose(prey, np.random.default_rng(0))

        breakdown = world.behaviour.breakdown(prey)
        assert breakdown["fear"] == pytest.approx([0.0])
        assert breakdown["hunger"][0] > 0.0


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
        return world.fear(config).urgency(prey)[0]

    def test_one_direction_cannot_tell_a_blend_from_the_real_thing(self):
        """The limitation the second direction exists to remove."""
        single = FearConfig(
            weight_gene="fear_weight",
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


def eastward_ramp(gain_per_unit, grid=9):
    """Ground rising steadily along +x, so the east option climbs and the west one descends."""
    x = np.arange(grid, dtype=np.float32) * gain_per_unit
    return np.broadcast_to(x, (grid, grid)).astype(np.float32)


class TestFatigueGradesTravelByEffort:
    """Issue #207: fatigue prefers rest without vetoing movement, and prefers an *easy* direction.

    The defect it replaces was structural rather than a mis-set weight. Scoring 1 on the null
    option and 0 on every travelling one made this drive's spread across options equal to its
    entire urgency — measured at 0.921 against hunger's 0.210 — so the loudest voice in the contest
    was spent on one bit of information, and hunger, which ranks the food correctly 0.998 of the
    time, never once decided a direction.
    """

    def fatigue(self, world, travel_effort=TRAVEL_EFFORT, climb_tolerance=CLIMB_TOLERANCE):
        return Fatigue(
            world.store,
            world.exertion,
            world.genetics,
            world.terrain,
            world.genes,
            FatigueConfig(
                weight_gene="fatigue_weight",
                exertion_saturation=1.0,
                travel_effort=travel_effort,
                climb_tolerance=climb_tolerance,
            ),
        )

    def scored(self, world, **config):
        """One mid-world animal's appeal over (east, west, null).

        Away from the edge deliberately: `options_at` clips into the world exactly as
        `candidate_positions` does, so from the fixture's default corner the west option lands on
        the animal's own position and is a stay-put option rather than a travelling one.
        """
        selection = world.spawn(1, x=np.float32([4.0]), y=np.float32([4.0]))
        return self.fatigue(world, **config).appeal(selection, *options_at(world, selection))[0]

    def test_staying_put_is_worth_more_than_travelling_but_not_everything(self):
        appeal = self.scored(World())

        assert appeal[NULL] == pytest.approx(1.0)
        assert appeal[EAST] == pytest.approx(1.0 - TRAVEL_EFFORT)

    def test_its_spread_across_options_is_the_travel_effort_on_level_ground(self):
        """The quantity this whole issue is about, asserted directly. How far a drive's appeal
        varies between options is what decides whether it can move a ranking at all — urgency only
        decides how much it contributes, and a constant added to every option changes nothing.
        """
        appeal = self.scored(World())

        assert appeal.max() - appeal.min() == pytest.approx(TRAVEL_EFFORT)

    def test_travel_effort_of_one_is_the_veto_it_replaced(self):
        """The degenerate case stays reachable, which is why the interval is closed at the top: a
        world that wants a tired animal to refuse to move at all can still say so, and the old
        behaviour is recovered exactly rather than approximately."""
        appeal = self.scored(World(), travel_effort=1.0)

        assert appeal[NULL] == pytest.approx(1.0)
        assert appeal[:NULL] == pytest.approx(np.zeros(NULL))

    def test_climbing_is_less_restful_than_descending(self):
        """The direction half. Distance cannot discriminate — every candidate sits one look-ahead
        away, so it is identical across the options — and relief is the one thing that differs."""
        appeal = self.scored(World(heights=eastward_ramp(0.5)))

        assert appeal[EAST] < appeal[WEST]

    def test_descending_is_no_more_restful_than_level_ground(self):
        """Only *gain* counts, which is the identical rule §2.5 settles for what a step costs
        (#113): descent is charged its horizontal distance and no more. Fatigue calling a downhill
        option restful while `Movement` charges it as level ground would be two readings of one
        physical fact drifting apart — the shape of defect #112 was.
        """
        downhill = self.scored(World(heights=eastward_ramp(0.5)))[WEST]

        assert downhill == pytest.approx(1.0 - TRAVEL_EFFORT)

    def test_a_steeper_climb_is_less_restful_than_a_gentle_one(self):
        gentle = self.scored(World(heights=eastward_ramp(0.5)))
        steep = self.scored(World(heights=eastward_ramp(2.0)))

        assert steep[EAST] < gentle[EAST]

    def test_a_bigger_climb_tolerance_forgives_the_same_slope(self):
        """The knob does what it says, and it is what keeps this drive from *steering* harder than
        it should: measured at 1.0 — comparable to the largest rise between an animal and a
        candidate — fatigue pulled downhill firmly enough to cost condition."""
        strict = self.scored(World(heights=eastward_ramp(0.5)), climb_tolerance=1.0)
        forgiving = self.scored(World(heights=eastward_ramp(0.5)), climb_tolerance=8.0)

        assert strict[EAST] < forgiving[EAST] < 1.0 - TRAVEL_EFFORT

    def test_an_option_clipped_onto_the_animal_reads_as_staying_put(self):
        """At a world corner half the candidate headings clip onto the animal's own position. An
        option proposing no displacement *is* rest, and identifying the null option by zero
        displacement rather than by its column index is what makes that fall out rather than
        needing a case of its own.
        """
        world = World()
        cornered = world.spawn(1)  # the fixture default is the origin
        appeal = self.fatigue(world).appeal(cornered, *options_at(world, cornered))[0]

        assert appeal[WEST] == pytest.approx(1.0)
        assert appeal[EAST] == pytest.approx(1.0 - TRAVEL_EFFORT)


class TestFatigueConfigRejectsWhatWouldBreakTheContest:
    def base(self, **overrides):
        params = dict(
            weight_gene="fatigue_weight",
            exertion_saturation=1.0,
            travel_effort=TRAVEL_EFFORT,
            climb_tolerance=CLIMB_TOLERANCE,
        )
        params.update(overrides)
        return FatigueConfig(**params)

    @pytest.mark.parametrize("effort", [0.0, -0.1, 1.5])
    def test_travel_effort_outside_the_unit_interval_is_rejected(self, effort):
        """At zero, walking is as restful as lying down: a tired animal never stops, exertion never
        sheds, and the drive that cannot act comes back (#126). Above one, travelling would be
        *negatively* restful — a reason to move, from the drive that exists to want rest."""
        with pytest.raises(ValueError, match="travel_effort"):
            self.base(travel_effort=effort)

    def test_travel_effort_of_exactly_one_is_allowed(self):
        assert self.base(travel_effort=1.0).travel_effort == 1.0

    @pytest.mark.parametrize("tolerance", [0.0, -1.0])
    def test_a_non_positive_climb_tolerance_is_rejected(self, tolerance):
        """At zero any ascent whatsoever makes an option maximally tiring, which turns a graded
        preference back into the veto this replaced — one applied to terrain instead of to
        movement."""
        with pytest.raises(ValueError, match="climb_tolerance"):
            self.base(climb_tolerance=tolerance)
