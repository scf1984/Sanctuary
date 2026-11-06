"""What the derived weather field costs per tick (#159).

The issue measured ~3 ms/tick at 256x256 for its proposal before anything was built. This times the
shipped `Climate.advance_to` — static field + seasonal cycle + lattice noise, bilinearly upsampled —
and the extra per-tick work it causes in `Plants.grow` (re-multiplying the temperature response).

Throwaway spike code (CLAUDE.md §8.3), not imported by anything. Run from a repository root with
`PYTHONPATH=.`:

    python docs/spikes/weather_bench.py [GRID] [WEATHER_SCALE_CELLS]
"""

from __future__ import annotations

import sys
import time

import numpy as np

from core.ecology.plants import PlantsConfig, _temperature_response
from core.world.climate import Climate, ClimateConfig, SeasonsConfig
from core.world.diffusion import DiffusionConfig
from core.world.terrain import Terrain, TerrainConfig

REPEATS = 15
TICKS_PER_REPEAT = 50


def median_ms(fn) -> float:
    samples = []
    for repeat in range(REPEATS):
        start = time.perf_counter()
        for tick in range(TICKS_PER_REPEAT):
            fn(repeat * TICKS_PER_REPEAT + tick)
        samples.append((time.perf_counter() - start) * 1000.0 / TICKS_PER_REPEAT)
    return float(np.median(samples))


def main() -> None:
    grid = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    scale_cells = float(sys.argv[2]) if len(sys.argv) > 2 else 16.0
    terrain = Terrain.generate(
        TerrainConfig(width=grid, height=grid, min_elevation=0.0, max_elevation=grid / 10.0, seed=1)
    )

    def climate(amplitude: float, weather: float) -> Climate:
        return Climate(
            terrain,
            ClimateConfig(
                equator_y=terrain.world_height / 2,
                equator_temperature=30.0,
                seasons=SeasonsConfig(
                    year_length=720,
                    amplitude_fraction=amplitude,
                    weather_amplitude=weather,
                    weather_scale=scale_cells,
                    weather_timescale=30,
                ),
            ),
            seed=1,
        )

    seasons_only = climate(0.5, 0.0)
    full = climate(0.5, 0.5)
    plants_config = PlantsConfig(
        solar_constant=8.0,
        latitude_tilt=0.0,
        min_growth_temperature=0.0,
        optimal_growth_temperature=22.0,
        max_growth_temperature=45.0,
        nutrient_per_biomass=1.0,
        initial_soil_nutrients=400.0,
        senescence_rate=0.05,
        saturation_accumulation=20.0,
        max_rooting_depth=0.5,
        forage_diffusion=DiffusionConfig(range=4.0, climb_penalty=0.5),
    )

    print(f"grid {grid}x{grid}, lattice every {scale_cells:g} cells, median of {REPEATS}")
    print(f"  seasons only (no noise)      : {median_ms(seasons_only.advance_to):6.2f} ms/tick")
    print(f"  seasons + weather noise      : {median_ms(full.advance_to):6.2f} ms/tick")
    print(
        "  plant temperature response   : "
        f"{median_ms(lambda _t: _temperature_response(full, plants_config)):6.2f} ms/tick"
    )


if __name__ == "__main__":
    main()
