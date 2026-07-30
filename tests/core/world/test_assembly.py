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
from core.behaviour.movement import MovementConfig
from core.ecology.cues import CueFieldConfig, ScentGenes
from core.ecology.metabolism import MetabolismConfig
from core.ecology.plants import PlantsConfig
from core.genetics.expression import ExpressionMode, GeneticsConfig
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

# Eight cue channels, per CLAUDE.md §2.5 — the settled floor, not a number this test picked.
SIGNATURE_GENES = tuple(f"signature_{i}" for i in range(8))
AVERSION_GENES = (
    tuple(f"aversion0_{i}" for i in range(8)),
    tuple(f"aversion1_{i}" for i in range(8)),
)
GENE_NAMES = (
    "size",
    "speed",
    "insulation",
    "sight",
    "scent_emission",
    "scent_acuity",
    *SIGNATURE_GENES,
    *AVERSION_GENES[0],
    *AVERSION_GENES[1],
    "mutability",
)
# Cue space is signed — a signature is a position in it, an aversion a direction through it — and
# everything else here is a quantity that cannot go negative (#104).
CUE_GENES = (*SIGNATURE_GENES, *AVERSION_GENES[0], *AVERSION_GENES[1])

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
        cue_field=CueFieldConfig(diffusion_range=3.0),
        metabolism=MetabolismConfig(
            gene_costs={
                **{name: 0.01 for name in GENE_NAMES},
                # Mutability charges nothing: high mutability already pays for itself in unfit
                # offspring, so a stable world selects it down with no energy price (#104).
                "mutability": 0.0,
            },
            basal_rate=0.05,
            thermoregulation_rate=0.01,
            neutral_temperature=20.0,
            insulation_gene="insulation",
        ),
        genetics=GeneticsConfig(
            expression_modes={
                name: ExpressionMode.SIGNED if name in CUE_GENES else ExpressionMode.MAGNITUDE
                for name in GENE_NAMES
            },
            mutability_gene="mutability",
            drift_margin=2.0,
        ),
        movement=MovementConfig(
            speed_gene="speed",
            size_gene="size",
            transport_cost=0.5,
            exertion_premium=2.0,
            climb_cost=1.0,
            walking_pace=0.4,
        ),
        exertion=ExertionConfig(recovery_rate=0.2),
        hunger=HungerConfig(
            weight=1.0, satiation_energy=200.0, detection_threshold=0.5, sight_gene="sight"
        ),
        # Thirst is deliberately the quietest drive here, and the reason is a finding rather than
        # a preference: a drive that *wins* with no mechanic behind it leaves the animal standing
        # still, and hunger is the only drive that can act today (see the assembly's docstring).
        # At equal weights thirst outscored hunger 0.30 to 0.10 in this world's climate and nothing
        # in it ever moved. Filed as #126; until then a world has to be tuned around it.
        thirst=ThirstConfig(weight=0.2, onset_temperature=25.0, saturation_temperature=40.0),
        fear=FearConfig(
            weight=1.0,
            scent_acuity_gene="scent_acuity",
            aversion_genes=AVERSION_GENES,
            detection_threshold=0.05,
            saturation=1.0,
        ),
        lust=LustConfig(
            weight=1.0, maturity_age=20, breeding_energy=120.0, abundant_energy=250.0
        ),
        fatigue=FatigueConfig(weight=1.0, exertion_saturation=20.0),
        scent_genes=ScentGenes(
            emission_gene="scent_emission", signature_genes=SIGNATURE_GENES
        ),
        gene_names=GENE_NAMES,
        founder_gene_ranges={
            "size": (0.8, 1.2),
            "speed": (1.0, 3.0),
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
            _ordered({name: (lambda: None) for name in TICK_ORDER + ("decomposition",)})

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

    def test_the_world_is_populated_and_aging(self):
        world = build_world(world_config(), seed=4)
        assert len(world.founders) == world.config.n_founders

        world.loop.advance(10)

        assert np.all(world.store.age[world.founders.to_mask()] == 10)

    def test_animals_move(self):
        """Movement is wired to a real decision: `Behaviour` picks a winner and the foragers among
        them walk toward the patch `Hunger` chose. Nothing asserts *where* — that is ecology."""
        world = build_world(world_config(), seed=5)
        before = world.store.x[world.founders.to_mask()].copy()

        world.loop.advance(15)

        assert np.any(world.store.x[world.founders.to_mask()] != before)

    def test_every_drive_scores(self):
        """All five are registered and all five run, which is the thing no test could check before
        an assembly existed."""
        world = build_world(world_config(), seed=6)

        world.loop.advance(1)

        breakdown = world.behaviour.breakdown(world.founders)
        assert set(breakdown) == {"hunger", "thirst", "fear", "lust", "fatigue"}
        assert all(scores.shape == (world.config.n_founders,) for scores in breakdown.values())

    def test_entities_stay_on_the_surface(self):
        """Surface-locked (§2.6): z is the ground under (x, y) after every step, so an animal that
        walked uphill is standing on the hill and not inside it."""
        world = build_world(world_config(), seed=7)

        world.loop.advance(10)

        mask = world.founders.to_mask()
        ground = world.terrain.elevation_at(world.store.x[mask], world.store.y[mask])
        assert world.store.z[mask] == pytest.approx(ground, abs=1e-4)

    def test_animals_pay_for_living(self):
        """Upkeep runs, so a world where nothing eats is a world that spends down. Feeding is #19's;
        until it lands this direction is the whole energy story."""
        world = build_world(world_config(), seed=8)
        before = world.ecology.energy(world.founders).sum()

        world.loop.advance(10)

        assert world.ecology.energy(world.founders).sum() < before


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
        mask = world.founders.to_mask()

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
