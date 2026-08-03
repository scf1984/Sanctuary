"""Animals eating animals: who strikes, how hard, and what is left on the ground (#179).

The claims worth pinning are the ones that were wrong first. A strike is **not** a mouthful — the
first build made them the same number and carnivory was measurably impossible in every world
(`docs/spikes/predation_viability.py`) — and damage follows the **frontier** rather than the raw
allocation, without which every half-hearted grazer mauls its neighbours and the population bleeds
into the carrion field faster than anything can eat it.
"""

import numpy as np
import pytest

from core.ecology.carrion import Carrion, CarrionConfig
from core.ecology.contact import pair_by_contact
from core.ecology.diet import Diet, DietConfig
from core.ecology.feeding import FeedingConfig
from core.ecology.metabolism import Metabolism, MetabolismConfig
from core.ecology.plants import Plants, PlantsConfig
from core.ecology.predation import Predation, PredationConfig
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

GENE_NAMES = ("size", "diet_animal_derived", "mutability", "insulation")
GENE_REGISTRY = gene_registry(GENE_NAMES, {"insulation": 1.0})
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
FEEDING_CONFIG = FeedingConfig(intake_rate=4.0, assimilation_max=0.5, size_gene="size")
GENETICS_CONFIG = GeneticsConfig(mutability_gene="mutability", drift_margin=2.0)
# Nothing here charges upkeep: these tests are about what a strike moves, and a basal drain would
# put a second withdrawal in every assertion for no gain.
METABOLISM_CONFIG = MetabolismConfig(
    basal_rate=0.0,
    thermoregulation_rate=0.0,
    neutral_temperature=25.0,
    insulation_gene="insulation",
)


class World:
    """A flat world with predation wired and nothing else running."""

    def __init__(self, strike_power=50.0, strike_range=1.0, frontier_exponent=2.0):
        self.terrain = Terrain(np.zeros((11, 11), dtype=np.float32), cell_size=1.0)
        self.climate = Climate(
            self.terrain,
            ClimateConfig(equator_y=0.0, equator_temperature=25.0, latitude_gradient=0.0),
        )
        self.plants = Plants(
            self.terrain, self.climate, Water.generate(self.terrain), PLANTS_CONFIG
        )
        self.plants.record_founding_stock(10_000.0)
        self.store = EntityStore(initial_capacity=32, n_drives=5, n_genes=len(GENE_NAMES))
        self.columns = ColumnRegistry()
        self.species = SpeciesRegistry(GENE_REGISTRY)
        self.species_id = self.species.register(GENE_NAMES)
        self.genetics = Genetics(
            self.store, self.columns, self.species, GENE_REGISTRY, GENETICS_CONFIG
        )
        self.ecology = Ecology(
            self.store,
            self.columns,
            self.genetics,
            self.climate,
            Metabolism(GENE_REGISTRY, METABOLISM_CONFIG),
            self.plants,
        )
        self.carrion = Carrion(self.terrain, self.plants, CarrionConfig(decay_rate=0.1))
        self.diet = Diet(
            GENE_REGISTRY,
            DietConfig(
                animal_derived_gene="diet_animal_derived",
                frontier_exponent=frontier_exponent,
            ),
        )
        self.predation = Predation(
            self.store,
            self.ecology,
            self.genetics,
            self.carrion,
            self.diet,
            GENE_REGISTRY,
            FEEDING_CONFIG,
            PredationConfig(strike_range=strike_range, strike_power=strike_power),
        )

    def spawn(self, *animals):
        """One entity per (x, y, flesh, size, energy) dict; returns their Selection."""
        n = len(animals)
        genes = np.zeros((n, len(GENE_NAMES)), dtype=np.float32)
        for i, a in enumerate(animals):
            genes[i, GENE_NAMES.index("size")] = a.get("size", 1.0)
            # The allocation is read on [0, 1] through a logistic, which never reaches its ends, so
            # "pure herbivore" is a large negative stored value rather than a zero.
            genes[i, GENE_NAMES.index("diet_animal_derived")] = a.get("flesh", -20.0)
        ids = self.store.allocate(
            n,
            x=np.array([a["x"] for a in animals], dtype=np.float32),
            y=np.array([a["y"] for a in animals], dtype=np.float32),
            z=np.zeros(n, dtype=np.float32),
            energy=np.array(
                [a.get("energy", 100.0) for a in animals], dtype=np.float32
            ),
            species_id=np.full(n, self.species_id, dtype=np.int32),
            genes=genes,
        )
        rows = [self.store._id_to_row[i] for i in ids.tolist()]
        return Selection.from_indices(np.array(rows, dtype=np.int64), self.store.capacity)

    def strike(self, selection, seed=1):
        self.predation.strike(selection, np.random.default_rng(seed))


# A stored gene of +20 expresses as a flesh allocation of essentially 1: a specialist carnivore.
CARNIVORE = 20.0


class TestAStrikeIsNotAMouthful:
    """The defect the first build shipped, and the whole reason carrion has a mass source."""

    def test_a_carnivore_takes_far_more_than_it_could_swallow(self):
        """Killing is bounded by force, eating by a gut. With them equal, a predator drained its
        prey four energy units at a time while a body held a hundred, so a kill took a dozen ticks
        of contact two moving animals never have — and carnivory could not pay in any world."""
        world = World(strike_power=50.0)
        pair = world.spawn(
            {"x": 5.0, "y": 5.0, "flesh": CARNIVORE}, {"x": 5.2, "y": 5.0}
        )

        world.strike(pair)

        taken = 100.0 - world.store.energy[pair.to_indices()[1]]
        assert taken == pytest.approx(50.0, rel=1e-3)
        assert taken > FEEDING_CONFIG.intake_rate * 10

    def test_the_whole_wound_lands_on_the_ground(self):
        """A strike feeds the attacker nothing directly. It eats by standing on the carcass next
        tick, which is what gives #100's commitment gene a payoff in food and makes scavenging
        exist without a mechanic."""
        world = World(strike_power=50.0)
        pair = world.spawn(
            {"x": 5.0, "y": 5.0, "flesh": CARNIVORE}, {"x": 5.2, "y": 5.0}
        )

        world.strike(pair)

        assert world.carrion.mass.sum() == pytest.approx(50.0, rel=1e-3)
        assert world.store.energy[pair.to_indices()[0]] == pytest.approx(100.0)

    def test_a_body_cannot_yield_more_than_it_is(self):
        """The cap is the kill rule and the multi-tick kill at once: a strike bigger than the whole
        pool empties it, and nothing here decides whether that is death — `Ecology.starving` reads
        an empty pool and `Death` frees the row, both already in the tick."""
        world = World(strike_power=500.0)
        pair = world.spawn(
            {"x": 5.0, "y": 5.0, "flesh": CARNIVORE},
            {"x": 5.2, "y": 5.0, "energy": 30.0},
        )

        world.strike(pair)

        assert world.store.energy[pair.to_indices()[1]] == pytest.approx(0.0)
        assert world.carrion.mass.sum() == pytest.approx(30.0, rel=1e-3)


class TestOnlyASpecialistIsDangerous:
    def test_a_herbivore_pair_barely_scratches_each_other(self):
        """The measured pathology: at a *linear* reading every grazer 7% allocated toward meat
        wounded whichever neighbour it stood beside, and 35,000 energy units of meat were standing
        at 2,000 ticks. The convex frontier is what makes a half-hearted predator harmless."""
        world = World(strike_power=50.0)
        pair = world.spawn({"x": 5.0, "y": 5.0}, {"x": 5.2, "y": 5.0})

        world.strike(pair)

        assert world.carrion.mass.sum() < 1e-3

    def test_damage_follows_the_frontier_rather_than_the_allocation(self):
        """A half-allocated omnivore does a *quarter* of a specialist's damage at `p = 2`, not
        half — the same convex curve #102 already prices the gut on."""
        specialist = World(strike_power=50.0)
        specialist.strike(
            specialist.spawn({"x": 5.0, "y": 5.0, "flesh": CARNIVORE}, {"x": 5.2, "y": 5.0})
        )
        omnivore = World(strike_power=50.0)
        omnivore.strike(
            omnivore.spawn({"x": 5.0, "y": 5.0, "flesh": 0.0}, {"x": 5.2, "y": 5.0})
        )

        assert omnivore.carrion.mass.sum() == pytest.approx(
            specialist.carrion.mass.sum() * 0.25, rel=0.02
        )


class TestSizeIsADefence:
    def test_the_same_jaws_do_less_to_a_bigger_body(self):
        """Free physics rather than an authored penalty, and it gives `size` its first benefit
        beyond a larger mouthful — until now it only ever charged upkeep, locomotion and inertia."""
        small = World(strike_power=50.0)
        small.strike(
            small.spawn(
                {"x": 5.0, "y": 5.0, "flesh": CARNIVORE}, {"x": 5.2, "y": 5.0, "size": 1.0}
            )
        )
        large = World(strike_power=50.0)
        large.strike(
            large.spawn(
                {"x": 5.0, "y": 5.0, "flesh": CARNIVORE}, {"x": 5.2, "y": 5.0, "size": 4.0}
            )
        )

        assert large.carrion.mass.sum() == pytest.approx(
            small.carrion.mass.sum() * 0.25, rel=0.02
        )


class TestContactIsWhatDecides:
    def test_animals_out_of_reach_are_untouched(self):
        """There is no chase resolution here and no success probability. Getting into contact is a
        pursuit fought out in `core.behaviour.movement`, where velocity is state (§2.5, #204)."""
        world = World(strike_range=1.0)
        pair = world.spawn(
            {"x": 2.0, "y": 5.0, "flesh": CARNIVORE}, {"x": 9.0, "y": 5.0}
        )

        world.strike(pair)

        assert world.carrion.mass.sum() == pytest.approx(0.0)

    def test_a_lone_animal_strikes_nothing(self):
        world = World()
        world.strike(world.spawn({"x": 5.0, "y": 5.0, "flesh": CARNIVORE}))

        assert world.carrion.mass.sum() == pytest.approx(0.0)

    def test_both_directions_resolve_from_the_same_pre_strike_pools(self):
        """Pairing is symmetric — what makes an animal an attacker is its own allocation and
        nothing else — so resolving in sequence would make the outcome depend on which of two
        identical animals happened to sort first, which is the grid leaking into the ecology."""
        world = World(strike_power=40.0)
        pair = world.spawn(
            {"x": 5.0, "y": 5.0, "flesh": CARNIVORE},
            {"x": 5.2, "y": 5.0, "flesh": CARNIVORE},
        )

        world.strike(pair)

        rows = pair.to_indices()
        assert world.store.energy[rows[0]] == pytest.approx(world.store.energy[rows[1]])
        assert world.carrion.mass.sum() == pytest.approx(80.0, rel=1e-3)


class TestNutrientsSurviveAKill:
    def test_the_total_does_not_move(self):
        """A kill relocates nutrient from a pool to the ground; it creates and destroys none (§6).
        `Ecology.kill` deliberately excretes nothing, because the flesh is not respired — it is
        lying there, and `Carrion.decompose` is what finally pays the ledger back."""
        world = World(strike_power=50.0)
        pair = world.spawn(
            {"x": 5.0, "y": 5.0, "flesh": CARNIVORE}, {"x": 5.2, "y": 5.0}
        )
        before = world.plants.total_nutrients()

        world.strike(pair)

        assert world.plants.total_nutrients() == pytest.approx(before)


class TestTheConfigRefusesTheDegenerate:
    @pytest.mark.parametrize("value", [0.0, -1.0])
    def test_a_strike_range_that_reaches_nothing_is_refused(self, value):
        with pytest.raises(ValueError, match="strike_range"):
            PredationConfig(strike_range=value, strike_power=10.0)

    @pytest.mark.parametrize("value", [0.0, -1.0])
    def test_a_strike_that_cannot_harm_is_refused(self, value):
        """A world where no strike ever harms anything is one where the animal half of every diet
        buys nothing, which is exactly the state this issue exists to end (§8.7)."""
        with pytest.raises(ValueError, match="strike_power"):
            PredationConfig(strike_range=1.0, strike_power=value)


class TestPairingByContact:
    """`core.ecology.contact`, shared with mating (#20). Extracted at the second caller rather than
    the third because the algorithm is subtle in ways a duplicate would not survive."""

    def positions(self, *points):
        x = np.array([p[0] for p in points], dtype=np.float32)
        y = np.array([p[1] for p in points], dtype=np.float32)
        return x, y

    def test_a_shared_cell_is_not_enough_when_the_gap_is_too_wide(self):
        """A shared bucket puts two entities within `range × √2`, so the true distance is checked
        rather than assumed — the bucket finds candidates and the distance decides."""
        x, y = self.positions((0.05, 0.05), (0.95, 0.95))

        first, second = pair_by_contact(
            x, y, np.array([0, 1]), 1.0, np.random.default_rng(1)
        )

        assert first.shape[0] == 0

    def test_touching_entities_pair(self):
        x, y = self.positions((0.4, 0.4), (0.6, 0.4))

        first, second = pair_by_contact(
            x, y, np.array([0, 1]), 1.0, np.random.default_rng(1)
        )

        assert sorted(first.tolist() + second.tolist()) == [0, 1]

    def test_each_row_appears_in_at_most_one_pair(self):
        """What `Ecology.kill` relies on to read every pool exactly once: three animals in one cell
        make one pair and one bystander, never a chain."""
        x, y = self.positions((0.4, 0.4), (0.5, 0.4), (0.6, 0.4))

        first, second = pair_by_contact(
            x, y, np.array([0, 1, 2]), 1.0, np.random.default_rng(1)
        )

        paired = first.tolist() + second.tolist()
        assert len(paired) == len(set(paired)) == 2

    def test_entities_in_distant_cells_never_pair(self):
        x, y = self.positions((0.5, 0.5), (8.5, 8.5))

        first, _ = pair_by_contact(
            x, y, np.array([0, 1]), 1.0, np.random.default_rng(1)
        )

        assert first.shape[0] == 0
