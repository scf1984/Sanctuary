"""A plant field for tests that need one only because `Ecology` requires it.

`Ecology.spend` excretes the nutrients an animal burns back into the cell it is standing in (#21),
so every service that charges energy now needs a field to excrete into — including a good many
tests that are about movement, drives or exertion and have no interest in plants at all.

Restating a `PlantsConfig` in each of those files would put the same eleven coefficients in six
places, which is exactly the drift `tests/support/genes.py` exists to remove. The coefficients here
are a plausible middle: nothing in this module's callers asserts on plant behaviour, and anything
that does builds its own field.
"""

from __future__ import annotations

from core.ecology.plants import Plants, PlantsConfig
from core.world.climate import Climate
from core.world.diffusion import DiffusionConfig
from core.world.terrain import Terrain
from core.world.water import Water

CONFIG = PlantsConfig(
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


def plant_field(
    terrain: Terrain,
    climate: Climate,
    founding_stock: float = 1.0e4,
    config: PlantsConfig = CONFIG,
) -> Plants:
    """A field over `terrain`, with the export ledger pre-seeded.

    `founding_stock` covers the endowment a caller hands its animals directly: energy the field
    never supplied still has to be on the export ledger, or the first `spend` excretes against
    nothing and `Plants.return_nutrients` rightly refuses (#21).

    The default is a working figure rather than an enormous one on purpose. A ledger inflated far
    past what a test actually burns would swamp `total_nutrients()`, and every conservation
    assertion measured against it in relative terms would go slack — so a caller that endows more
    than this passes its own number instead of the default being raised to fit.

    The assembled world seeds exactly its founders' energy (`core.world.assembly`), which is what
    makes the conservation invariant meaningful there.
    """
    plants = Plants(terrain, climate, Water.generate(terrain), config)
    plants.record_founding_stock(founding_stock)
    return plants
