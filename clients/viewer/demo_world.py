"""The world the diagnostic viewer looks at: a real assembled world, not a scatter of markers.

**Separate from `app.py` because `app.py` imports pygame at module scope.** CI installs `.[dev]`
and never the viewer extra — deliberately, since a run that never installs pygame is a standing
check that the core is runnable headless (CLAUDE.md §3) — so nothing importing `app.py` is
collectable there. World-building sitting inside that module is precisely why its `EntityStore`
call could go a whole gene-matrix release without `n_genes` and without anyone noticing (#110):
the one entry point that runs the simulation as a program was the one module no test could load.
The rule `render.py` already states — pygame lives in `app.py` and nowhere else — earns its keep
only if everything worth testing stays on this side of it.

This module is now **config and nothing else**. `core.world.assembly.build_world` (#115) does the
wiring, so the viewer cannot drift into a second assembly disagreeing with the settled tick order —
which is what §7.2 exists to prevent, and what the previous version of this file was explicitly
waiting for.

**These numbers are one world, not defaults.** Nothing else should import them. A world tuned for
watching — small enough to fill a window, warm enough that plants grow, populated enough that
something is always happening — is not the world a competition (#41) or a shipped starting state
(#101) would want, and dressing it up as a shared default is how a viewing preference becomes an
ecological constant (§2.1: coefficients are per world, never constants in `core/`).
"""

from __future__ import annotations

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
from core.world.assembly import World, WorldConfig, build_world
from core.world.climate import ClimateConfig
from core.world.terrain import TerrainConfig

_GRID = 80
_CELL_SIZE = 1.0
_WORLD_EXTENT = (_GRID - 1) * _CELL_SIZE

# Cue space is eight-dimensional, per CLAUDE.md §2.5's settled floor. The signature genes and the
# two aversion directions are that reserved block, named here because the vocabulary is per world.
_SIGNATURE_GENES = tuple(f"signature_{channel}" for channel in range(8))
_AVERSION_GENES = (
    tuple(f"aversion0_{channel}" for channel in range(8)),
    tuple(f"aversion1_{channel}" for channel in range(8)),
)
_GENE_NAMES = (
    "size",
    "speed",
    "insulation",
    "sight",
    "scent_emission",
    "scent_acuity",
    *_SIGNATURE_GENES,
    *_AVERSION_GENES[0],
    *_AVERSION_GENES[1],
)


def demo_world_config(n_entities: int, seed: int) -> WorldConfig:
    """One world's coefficients: warm, green, and small enough to watch."""
    return WorldConfig(
        terrain=TerrainConfig(
            width=_GRID,
            height=_GRID,
            # Relief a tenth of extent, the ratio `TerrainConfig` asks callers to choose (#112).
            # Elevation shares x and y's unit, so this is directly a statement about how steep the
            # world looks — and about what crossing a ridge costs, since `climb_cost` below is
            # denominated in that same unit as `transport_cost`.
            min_elevation=0.0,
            max_elevation=_WORLD_EXTENT / 10.0,
            cell_size=_CELL_SIZE,
            seed=seed,
        ),
        climate=ClimateConfig(equator_y=_WORLD_EXTENT / 2.0),
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
        ),
        cue_field=CueFieldConfig(diffusion_range=3.0),
        metabolism=MetabolismConfig(
            gene_costs={name: 0.01 for name in _GENE_NAMES},
            basal_rate=0.05,
            thermoregulation_rate=0.01,
            neutral_temperature=20.0,
            insulation_gene="insulation",
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
            weight=1.0, satiation_energy=200.0, forage_reluctance=4.0, sight_gene="sight"
        ),
        # Thirst is held quiet on purpose, and this is a workaround rather than a tuning choice: a
        # drive that *wins* with no mechanic behind it leaves the animal standing still, and hunger
        # is the only drive that can act today. At equal weights thirst outscores hunger in this
        # climate and nothing in the world ever moves — which is #126.
        thirst=ThirstConfig(weight=0.2, onset_temperature=25.0, saturation_temperature=40.0),
        fear=FearConfig(
            weight=1.0,
            scent_acuity_gene="scent_acuity",
            aversion_genes=_AVERSION_GENES,
            detection_threshold=0.05,
            saturation=1.0,
        ),
        lust=LustConfig(
            weight=1.0, maturity_age=20, breeding_energy=120.0, abundant_energy=250.0
        ),
        fatigue=FatigueConfig(weight=1.0, exertion_saturation=20.0),
        scent_genes=ScentGenes(emission_gene="scent_emission", signature_genes=_SIGNATURE_GENES),
        gene_names=_GENE_NAMES,
        # Naive founders (§2.5, #101): a uniform draw over the whole cue space rather than a chosen
        # point, so nothing here writes down what a lineage smells like or what frightens it.
        founder_gene_ranges={
            "size": (0.8, 1.2),
            "speed": (1.0, 3.0),
            "insulation": (0.0, 0.5),
            "sight": (2.0, 6.0),
            "scent_emission": (0.5, 1.5),
            "scent_acuity": (0.5, 1.5),
            **{name: (0.0, 1.0) for name in _SIGNATURE_GENES},
            **{name: (-1.0, 1.0) for name in _AVERSION_GENES[0] + _AVERSION_GENES[1]},
        },
        n_founders=n_entities,
        founder_energy=180.0,
    )


def build_demo_world(seed: int, n_entities: int) -> World:
    """The assembled world the viewer runs, reproducible from `seed`.

    Generation is a pure function of `seed` (§2.2): the simulation itself is non-deterministic, but
    a world the viewer cannot rebuild is one whose crash cannot be replayed.
    """
    return build_world(demo_world_config(n_entities, seed), seed=seed)
