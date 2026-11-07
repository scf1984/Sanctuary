"""Where the temperate world's climate numbers come from (#261, docs/spikes/thermal-drive.md §5).

For a grid of gradient scales and equator offsets, the demo world's climate is rebuilt with both
gradients scaled together (#112's pair) and the equator set that far above neutral. Each row reports
the two conditions the temperate world is fitted to: the share of ground-and-year above neutral (aim:
about half) and the mean squared deviation from neutral (aim: the demo world's, because the shared
`thermoregulation_rate` was derived from it, #176). It also reports the summer and winter shares above
neutral, the warmest winter pole and the coolest summer-equator maximum across seeds.

Throwaway spike code (CLAUDE.md §8.3), not imported by anything. Run from a repository root with
`PYTHONPATH=.`:

    python docs/spikes/temperate_climate_fit.py
"""

from __future__ import annotations

import dataclasses

import numpy as np

from clients.viewer.demo_world import demo_world_config
from core.world.climate import Climate
from core.world.terrain import Terrain

SEEDS = range(1, 9)
SAMPLES_PER_YEAR = 24


def year_of_fields(climate: Climate, year: int) -> np.ndarray:
    ticks = range(0, year, year // SAMPLES_PER_YEAR)
    return np.stack([climate.field_at(tick) for tick in ticks])


def main() -> None:
    configs = {seed: demo_world_config(10, seed) for seed in SEEDS}
    terrains = {seed: Terrain.generate(config.terrain) for seed, config in configs.items()}
    neutral = configs[1].metabolism.neutral_temperature
    year = configs[1].climate.seasons.year_length

    demo = [
        float(np.mean((year_of_fields(Climate(terrains[s], configs[s].climate, seed=s), year)
                       - neutral) ** 2))
        for s in SEEDS
    ]
    print(f"demo world: mean squared deviation {np.mean(demo):.1f}")

    for scale in (9.0, 10.0, 11.0):
        for above in (16.0, 17.0, 18.0, 19.0, 20.0):
            msd, share, summer_share, winter_share, winter_pole, summer_equator = ([] for _ in range(6))
            for s in SEEDS:
                base = configs[s].climate
                climate_config = dataclasses.replace(
                    base,
                    equator_temperature=neutral + above,
                    latitude_gradient=scale * base.latitude_gradient,
                    lapse_rate=scale * base.lapse_rate,
                )
                climate = Climate(terrains[s], climate_config, seed=s)
                fields = year_of_fields(climate, year)
                msd.append(np.mean((fields - neutral) ** 2))
                share.append(np.mean(fields > neutral))
                summer, winter = climate.field_at(year // 4), climate.field_at(3 * year // 4)
                summer_share.append(np.mean(summer > neutral))
                winter_share.append(np.mean(winter > neutral))
                winter_pole.append(winter[[0, -1]].max())
                equator_row = int(round(base.equator_y / terrains[s].cell_size))
                summer_equator.append(summer[equator_row - 1 : equator_row + 1].max())
            print(
                f"scale={scale:g} equator=neutral+{above:g} msd={np.mean(msd):.1f} "
                f"above={np.mean(share):.2f} summer_above={np.mean(summer_share):.2f} "
                f"winter_above={np.mean(winter_share):.2f} warmest_winter_pole={max(winter_pole):.1f} "
                f"coolest_summer_equator_max={min(summer_equator):.1f}"
            )


if __name__ == "__main__":
    main()
