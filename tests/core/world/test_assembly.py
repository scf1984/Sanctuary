"""World assembly: every service running alongside the others, in the settled order (issue #115).

These are the first tests in the repository where one service reads what another wrote. Everything
before them exercised a service against its own fixture, so a world that assembles and advances is
itself the assertion — the rest of this file pins the properties that would otherwise regress
silently.

Nothing here asserts a population figure or an energy level. Those are ecological outcomes, not
contracts (§8.1), and several of the systems that would produce them do not exist yet.
"""

import numpy as np
import pytest

from core.behaviour.drives import (
    FatigueConfig,
    FearConfig,
    HungerConfig,
    LustConfig,
    ThirstConfig,
)
from core.behaviour.exertion import ExertionConfig
from core.behaviour.service import BehaviourConfig
from core.behaviour.movement import MovementConfig
from core.ecology.conception import ConceptionConfig
from core.entities.growth import GrowthConfig
from core.ecology.cues import CueFieldConfig, ScentGenes
from core.ecology.diet import DietConfig
from core.ecology.carrion import CarrionConfig
from core.ecology.feeding import FeedingConfig
from core.ecology.predation import PredationConfig
from core.ecology.metabolism import MetabolismConfig
from core.ecology.plants import PlantsConfig
from core.genetics.expression import GeneticsConfig
from core.world.diffusion import DiffusionConfig
from core.selection import Selection
from core.world.assembly import (
    TICK_ORDER,
    SystemOrderError,
    World,
    WorldConfig,
    _ordered,
    build_world,
)
from core.world.climate import ClimateConfig
from core.world.terrain import TerrainConfig

from tests.support.genes import gene_registry

# Eight cue channels, per CLAUDE.md §2.5 — the settled floor, not a number this test picked.
SIGNATURE_GENES = tuple(f"signature_{i}" for i in range(8))
AVERSION_GENES = (
    tuple(f"aversion0_{i}" for i in range(8)),
    tuple(f"aversion1_{i}" for i in range(8)),
)
GENE_NAMES = (
    "size",
    "speed",
    "agility",
    "haste",
    "insulation",
    "sight",
    "scent_emission",
    "scent_acuity",
    *SIGNATURE_GENES,
    *AVERSION_GENES[0],
    *AVERSION_GENES[1],
    "mutability",
    "choice_temperature",
    "commitment",
    "diet_animal_derived",
    "maturity_age",
    "hunger_weight",
    "thirst_weight",
    "fear_weight",
    "lust_weight",
    "fatigue_weight",
    "gestation_length",
)
# Cue space is signed — a signature is a position in it, an aversion a direction through it — and
# everything else here is a quantity that cannot go negative (#104).
CUE_GENES = (*SIGNATURE_GENES, *AVERSION_GENES[0], *AVERSION_GENES[1])
# The genes §2.5 costs at zero: their counterweight is a selective consequence rather than upkeep.
FREE_GENES = (*CUE_GENES, "mutability", "choice_temperature", "commitment", "haste")

GENE_REGISTRY = gene_registry(
    GENE_NAMES,
    # Every other quantity charges the same token rate. A cost on a `SIGNED` gene is now rejected
    # outright by the registry (#136), so this fixture can no longer express the defect it once
    # made reachable.
    {name: 0.01 for name in GENE_NAMES if name not in FREE_GENES},
)

GRID = 24
CELL_SIZE = 1.0
# Relief a tenth of extent, the ratio TerrainConfig asks callers to choose (#112).
RELIEF = (GRID - 1) * CELL_SIZE / 10.0


def world_config(**overrides):
    params = dict(
        terrain=TerrainConfig(
            width=GRID,
            height=GRID,
            min_elevation=0.0,
            max_elevation=RELIEF,
            cell_size=CELL_SIZE,
            seed=7,
        ),
        climate=ClimateConfig(equator_y=(GRID - 1) * CELL_SIZE / 2),
        plants=PlantsConfig(
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
        ),
        growth=GrowthConfig(
            # Grow when free rows fall below a tenth of the rows in use. Measured in
            # docs/spikes/conception-and-capacity.md: at the steepest growth a world managed about
            # 0.43% of occupancy allocated per tick, so a tenth is roughly twenty ticks of runway
            # — and since `grow` doubles, it is reached rarely.
            reserve_fraction=0.1,
        ),
        conception=ConceptionConfig(
            # World units. A contact distance, not a search radius: finding each other is the lust
            # drive's business (#188), and by the time two animals are this close they have walked.
            contact_range=2.0,
            # Energy units, moved out of the parents rather than charged and burned. Under a
            # founder's 180 so a healthy adult can breed more than once before feeding back up —
            # the breeding interval is emergent from that rather than being a constant.
            offspring_energy=60.0,
            maturity_gene="maturity_age",
            gestation_gene="gestation_length",
            # Genetic distance at which interbreeding reaches zero; #16 reads the same number.
            speciation_threshold=8.0,
        ),
        diet=DietConfig(animal_derived_gene="diet_animal_derived", frontier_exponent=2.0),
        feeding=FeedingConfig(intake_rate=0.6, assimilation_max=0.5, size_gene="size"),
        predation=PredationConfig(strike_range=1.0, strike_power=20.0),
        carrion=CarrionConfig(decay_rate=0.1),
        cue_field=CueFieldConfig(diffusion_range=3.0),
        metabolism=MetabolismConfig(
            basal_rate=0.05,
            thermoregulation_rate=0.01,
            neutral_temperature=20.0,
            insulation_gene="insulation",
        ),
        genetics=GeneticsConfig(
            mutability_gene="mutability",
            drift_margin=2.0,
        ),
        movement=MovementConfig(
            speed_gene="speed",
            size_gene="size",
            agility_gene="agility",
            haste_gene="haste",
            transport_cost=0.5,
            exertion_premium=2.0,
            climb_cost=1.0,
            walking_pace=0.4,
        ),
        exertion=ExertionConfig(recovery_rate=0.2),
        hunger=HungerConfig(
            weight_gene="hunger_weight", satiation_energy=200.0, detection_threshold=0.5, sight_gene="sight"
        ),
        # Thirst is deliberately the quietest drive here, and the reason is a finding rather than
        # a preference: a drive that *wins* with no mechanic behind it leaves the animal standing
        # still, and hunger is the only drive that can act today (see the assembly's docstring).
        # At equal weights thirst outscored hunger 0.30 to 0.10 in this world's climate and nothing
        # in it ever moved. Filed as #126; until then a world has to be tuned around it.
        thirst=ThirstConfig(weight_gene="thirst_weight", onset_temperature=25.0, saturation_temperature=40.0),
        fear=FearConfig(
            weight_gene="fear_weight",
            scent_acuity_gene="scent_acuity",
            aversion_genes=AVERSION_GENES,
            detection_threshold=0.05,
            saturation=1.0,
        ),
        lust=LustConfig(
            weight_gene="lust_weight",
            maturity_gene="maturity_age",
            scent_acuity_gene="scent_acuity",
            detection_threshold=1e-4,
            breeding_energy=120.0,
            abundant_energy=250.0,
        ),
        fatigue=FatigueConfig(
            weight_gene="fatigue_weight",
            exertion_saturation=20.0,
            travel_effort=0.25,
            climb_tolerance=4.0,
        ),
        behaviour=BehaviourConfig(
            # Eight headings is enough that a forager can follow a gradient without the walk
            # visibly staircasing, and the per-entity jitter makes the effective resolution
            # continuous across the population.
            n_candidates=8,
            # One diffusion range: the distance over which the forage field carries information,
            # so it is the furthest a candidate reading can vouch for.
            look_ahead=4.0,
            commitment_gene="commitment",
            choice_temperature_gene="choice_temperature",
        ),
        scent_genes=ScentGenes(
            emission_gene="scent_emission", signature_genes=SIGNATURE_GENES
        ),
        genes=GENE_REGISTRY.specs,
        founder_gene_ranges={
            "size": (0.8, 1.2),
            "speed": (1.0, 3.0),
            # World units per tick per tick, against those speeds: a founder needs a few ticks to
            # reach its own top speed and as long again to reverse (#204).
            "agility": (0.3, 0.9),
            # Read through `exp`, so haste founds between 1 and about 4 — the band across which
            # the gene visibly changes a pace at the drive advantages this world produces (#203).
            "haste": (0.0, 1.4),
            "insulation": (0.0, 0.5),
            "sight": (2.0, 6.0),
            "scent_emission": (0.5, 1.5),
            "scent_acuity": (0.5, 1.5),
            # Naive: a uniform draw over the whole cue space rather than a chosen point, so nothing
            # here writes down what any lineage smells like or what frightens it (§2.5, #101).
            **{name: (0.0, 1.0) for name in SIGNATURE_GENES},
            **{name: (-1.0, 1.0) for name in AVERSION_GENES[0] + AVERSION_GENES[1]},
            # Around the 0.02 the drift spike measured as a working mutation-drift balance, drawn
            # rather than fixed so a founder population varies in its evolvability like anything
            # else (docs/spikes/speciation-drift.md).
            "mutability": (0.01, 0.03),
            "choice_temperature": (-0.3, 0.3),
            "commitment": (0.05, 0.25),
            # Drive weights, drawn around 1 so founders differ in temperament from the first
            # generation and selection has something to act on (§2.5, #23).
            "hunger_weight": (0.6, 1.4),
            "thirst_weight": (0.6, 1.4),
            "fear_weight": (0.6, 1.4),
            "lust_weight": (0.6, 1.4),
            "fatigue_weight": (0.6, 1.4),
            # Spread across zero, which the squash reads as allocations either side of an even
            # split: founders are undecided about what they eat rather than declared herbivores.
            "diet_animal_derived": (-1.0, 1.0),
            "maturity_age": (5.0, 15.0),
            # Ticks. Short against a lifetime, and drawn so founders differ from generation one.
            "gestation_length": (20.0, 60.0),
        },
        n_founders=40,
        founder_energy=180.0,
    )
    params.update(overrides)
    return WorldConfig(**params)


class TestTheOrderIsDeclaredData:
    def test_the_loop_runs_the_declared_order(self):
        """The tuple sequences the loop rather than describing it (§4: a rule declared as data must
        be consulted by the code it governs)."""
        world = build_world(world_config(), seed=1)

        assert len(world.loop.systems) == len(TICK_ORDER)

    def test_a_system_built_but_not_placed_is_rejected(self):
        with pytest.raises(SystemOrderError, match="unplaced"):
            _ordered({name: (lambda: None) for name in TICK_ORDER + ("speciation",)})

    def test_an_order_naming_a_system_that_does_not_exist_is_rejected(self):
        """Otherwise a name in TICK_ORDER with nothing behind it is a silent gap in the tick."""
        with pytest.raises(SystemOrderError, match="unbuilt"):
            _ordered({name: (lambda: None) for name in TICK_ORDER[:-1]})

    def test_upkeep_follows_movement_which_follows_scoring(self):
        """The three hard dependencies among the systems that exist: an animal acts on this tick's
        decision, and pays for the ground it actually covered."""
        assert TICK_ORDER.index("drive_scoring") < TICK_ORDER.index("movement")
        assert TICK_ORDER.index("movement") < TICK_ORDER.index("metabolic_upkeep")
        assert TICK_ORDER.index("cue_field_rebuild") < TICK_ORDER.index("drive_scoring")

    def test_recovery_follows_movement(self):
        """#107's placement: the tick's effort is spent and then the tick's rest is taken, so
        nothing ever reads a raw exertion value."""
        assert TICK_ORDER.index("movement") < TICK_ORDER.index("exertion_recovery")


class TestABuiltWorldRuns:
    def test_a_world_advances_without_raising(self):
        world = build_world(world_config(), seed=1)

        world.loop.advance(20)

        assert world.loop.tick_count == 20

    def test_every_invariant_holds_over_a_run(self):
        """The harness is what makes "it ran" mean something: energy never negative, nothing
        outside the world, no live entity on a free row, nutrients conserved (§6)."""
        world = build_world(world_config(), seed=3, debug_checks=True)

        world.loop.advance(50)

        assert world.loop.tick_count == 50

    def test_animals_eat_what_grows(self):
        """The loop is closed rather than a drain (§2.5). Asserted separately from
        `test_every_invariant_holds_over_a_run` because nutrient conservation holds perfectly in a
        world where nothing eats at all — it would pass unchanged if feeding never fired.
        """
        world = build_world(world_config(), seed=5)
        population = Selection.from_mask(world.store.alive)
        opening = world.store.energy[world.store.alive].copy()
        upkeep = world.ecology.upkeep(population)

        world.loop.advance(1)

        # Biomass left the field. Asserted on the export ledger rather than on standing crop,
        # because `plant_growth` runs before `feeding` in the same tick (§2.1) and a young field
        # grows faster than a herd can eat — so the standing total rises even while it is grazed.
        assert world.plants.exported_nutrients > 0.0
        # At least one animal ends the tick holding more than upkeep alone would have left it,
        # which is only possible if something credited the pool.
        closing = world.store.energy[world.store.alive]
        assert np.any(closing > opening - upkeep)

    def test_animals_that_run_out_of_energy_leave_the_world(self):
        """The half of the loop this world has: a population can now fall. Nothing is born, so it
        can only fall — which is the honest shape until #20."""
        world = build_world(world_config(), seed=7)
        # An emptied pool is what death reads, so empty some rather than waiting for starvation to
        # arrive on its own schedule; what is under test is that the loop acts on it.
        doomed = world.store.alive.copy()
        doomed[np.cumsum(doomed) > 10] = False
        world.store.energy[doomed] = 0.0
        before = int(world.store.alive.sum())

        world.loop.advance(1)

        assert int(world.store.alive.sum()) == before - int(doomed.sum())
        assert world.store.available >= int(doomed.sum())

    def test_a_row_freed_by_death_is_handed_out_again(self):
        """§2.1 orders death before reproduction precisely so this holds within a tick. Nothing
        breeds yet, so the allocation stands in for what #20 will do."""
        world = build_world(world_config(), seed=8)
        world.store.energy[world.store.alive] = 0.0

        world.loop.advance(1)
        reused = world.store.allocate(1, energy=np.array([50.0], dtype=np.float32))

        assert world.store.alive.sum() == 1
        assert reused.shape == (1,)

    def test_a_gestating_row_does_not_act(self):
        """§2.1's rule stated as a condition rather than an ordering: an unborn animal is not
        half-simulated, it is simply not a participant. One exclusion in `living()` keeps it out of
        sensing, movement, feeding, upkeep, death and scent at once (#20)."""
        world = build_world(world_config(), seed=11)
        row = int(np.flatnonzero(world.store.alive)[0])
        world.store.age[row] = -5
        before = (world.store.x[row], world.store.y[row], world.store.energy[row])

        world.loop.advance(1)

        assert world.store.x[row] == before[0]
        assert world.store.y[row] == before[1]
        # Not fed, and not charged upkeep either — it is outside every system.
        assert world.store.energy[row] == before[2]
        # But it did age, because `Aging` is the gestation clock.
        assert world.store.age[row] == -4

    def test_a_gestating_row_is_born_when_its_term_is_up(self):
        world = build_world(world_config(), seed=12)
        row = int(np.flatnonzero(world.store.alive)[0])
        world.store.age[row] = -2

        world.loop.advance(2)

        assert world.store.age[row] == 0
        assert world.store.alive[row]

    def test_the_world_is_populated_and_aging(self):
        world = build_world(world_config(), seed=4)
        assert len(world.founders) == world.config.n_founders

        world.loop.advance(10)

        # The living, not every allocated row: predation frees rows and conception fills them, so
        # a world ten ticks old can already hold a gestating young at a negative age (#20). That is
        # the same `living()` distinction the tick loop itself draws.
        born = world.store.alive & (world.store.age >= 0)
        assert np.all(world.store.age[born] == 10)

    def test_animals_move(self):
        """Movement is wired to a real decision: `Behaviour` picks a winner and the foragers among
        them walk toward the patch `Hunger` chose. Nothing asserts *where* — that is ecology."""
        world = build_world(world_config(), seed=5)
        alive = world.store.alive.copy()
        before = world.store.x[alive].copy()

        world.loop.advance(15)

        assert np.any(world.store.x[: alive.shape[0]][alive] != before)

    def test_every_drive_scores(self):
        """All five are registered and all five run, which is the thing no test could check before
        an assembly existed."""
        world = build_world(world_config(), seed=6)

        world.loop.advance(1)

        breakdown = world.behaviour.breakdown(Selection.from_mask(world.store.alive))
        assert set(breakdown) == {"hunger", "thirst", "fear", "lust", "fatigue"}
        assert all(scores.shape == (world.config.n_founders,) for scores in breakdown.values())

    def test_entities_stay_on_the_surface(self):
        """Surface-locked (§2.6): z is the ground under (x, y) after every step, so an animal that
        walked uphill is standing on the hill and not inside it."""
        world = build_world(world_config(), seed=7)

        world.loop.advance(10)

        mask = world.store.alive
        ground = world.terrain.elevation_at(world.store.x[mask], world.store.y[mask])
        assert world.store.z[mask] == pytest.approx(ground, abs=1e-4)

    def test_animals_pay_for_living(self):
        """Upkeep runs, so a world where nothing eats is a world that spends down. Feeding is #19's;
        until it lands this direction is the whole energy story."""
        world = build_world(world_config(), seed=8)
        population = Selection.from_mask(world.store.alive)
        before = world.ecology.energy(population).sum()

        world.loop.advance(10)

        assert world.ecology.energy(Selection.from_mask(world.store.alive)).sum() < before


class TestBatchingDoesNotChangeTheWorld:
    """§2.4: the wake schedule decides when compute happens, never how fast the world moves.

    A fixed system order satisfies that automatically *provided no system reads anything outside the
    store between ticks*, which is worth asserting once rather than assuming — it is the property
    that makes offline catch-up the same simulation as a live run rather than a second code path.
    """

    def _state(self, world):
        mask = Selection.from_mask(world.store.alive).to_mask()
        return (
            world.store.x[mask].copy(),
            world.store.y[mask].copy(),
            world.store.z[mask].copy(),
            world.store.energy[mask].copy(),
            world.store.age[mask].copy(),
            world.store.exertion[mask].copy(),
            world.store.drive_scores[mask].copy(),
            world.plants.total_nutrients(),
        )

    def test_one_batch_of_n_equals_n_batches_of_one(self):
        batched = build_world(world_config(), seed=11)
        stepped = build_world(world_config(), seed=11)

        batched.loop.advance(30)
        for _ in range(30):
            stepped.loop.advance(1)

        for batched_column, stepped_column in zip(self._state(batched), self._state(stepped)):
            assert np.asarray(batched_column) == pytest.approx(np.asarray(stepped_column))

    def test_uneven_batches_agree_too(self):
        """A decaying wake schedule (§2.4) produces uneven batches, not uniform ones."""
        even = build_world(world_config(), seed=12)
        uneven = build_world(world_config(), seed=12)

        even.loop.advance(24)
        for batch in (1, 3, 8, 12):
            uneven.loop.advance(batch)

        for even_column, uneven_column in zip(self._state(even), self._state(uneven)):
            assert np.asarray(even_column) == pytest.approx(np.asarray(uneven_column))


class TestConfigRejectsWorldsThatCannotBeFounded:
    def test_a_gene_with_no_founding_range_is_rejected(self):
        """The same rule `MetabolismConfig` applies to costs: a gene left out would found the world
        at zero, which for a speed gene is a population that cannot move."""
        ranges = dict(world_config().founder_gene_ranges)
        del ranges["speed"]

        with pytest.raises(ValueError, match="speed"):
            world_config(founder_gene_ranges=ranges)

    def test_an_inverted_founding_range_is_rejected(self):
        ranges = dict(world_config().founder_gene_ranges)
        ranges["size"] = (1.2, 0.8)

        with pytest.raises(ValueError, match="inverted"):
            world_config(founder_gene_ranges=ranges)

    def test_a_world_with_no_founders_is_rejected(self):
        with pytest.raises(ValueError, match="founder"):
            world_config(n_founders=0)


class TestFoundersAreNaive:
    def test_the_same_seed_founds_the_same_world(self):
        """Generation is reproducible even though the simulation is not (§2.2): a starting state
        that cannot be rebuilt is one whose crash cannot be replayed."""
        first = build_world(world_config(), seed=99)
        second = build_world(world_config(), seed=99)

        assert np.array_equal(first.terrain.heights, second.terrain.heights)
        assert np.array_equal(
            first.genetics.genes(first.founders), second.genetics.genes(second.founders)
        )

    def test_different_seeds_found_different_worlds(self):
        first = build_world(world_config(), seed=1)
        second = build_world(world_config(), seed=2)

        assert not np.array_equal(
            first.genetics.genes(first.founders), second.genetics.genes(second.founders)
        )

    def test_founders_vary_within_the_population(self):
        """A founding population of identical creatures has nothing for selection to act on, and
        `inherit_genes` draws its spread from parental disagreement (#104)."""
        world = build_world(world_config(), seed=13)

        assert np.all(world.genetics.genes(world.founders).std(axis=0) > 0.0)

    def test_founders_start_on_the_surface_and_alive(self):
        world = build_world(world_config(), seed=14)
        mask = world.store.alive

        assert np.all(world.store.alive[mask])
        assert np.all(world.store.health[mask] == 1.0)
        assert world.store.z[mask] == pytest.approx(
            world.terrain.elevation_at(world.store.x[mask], world.store.y[mask]), abs=1e-4
        )


class TestOneWorldIsNotAnother:
    def test_two_worlds_share_no_state(self):
        """No singletons (§4). The prototype's metaclass made two worlds impossible and test
        isolation with them."""
        first = build_world(world_config(), seed=21)
        second = build_world(world_config(), seed=21)

        first.loop.advance(5)

        assert second.loop.tick_count == 0
        assert np.all(second.store.age[second.founders.to_mask()] == 0)
        assert first.columns is not second.columns


def test_the_world_object_exposes_only_wired_services():
    """`World` is the whole set deliberately: the services share one store and one ColumnRegistry,
    so handing back a subset would let a caller construct a second owner of a column."""
    world = build_world(world_config(), seed=31)

    assert isinstance(world, World)
    assert world.columns.owner_of("exertion") == "Exertion"
    assert world.columns.owner_of("drive_scores") == "Behaviour"
    assert world.columns.owner_of("age") == "Aging"
