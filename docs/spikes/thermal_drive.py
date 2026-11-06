"""Does a thermal drive turn seasons into movement, or do winters still only kill (#249)?

Before #249 nothing read the temperature field to decide where to go: animals felt it only as
upkeep, so a bad season was differential mortality (range contraction) and never migration. This
runs the demo world with and without the thermal drive and asks:

- **Range.** Does the population's mean distance from the equator, and the mean temperature it
  stands in, oscillate with the year? Compared against the same world with the drive switched off,
  because seasons alone move the field under a stationary herd and that is not migration.
- **Seasonal mortality.** Deaths in the warm half of the year against the cold half.
- **Viability.** Population at the end.

"Off" keeps the gene vocabulary and every other number identical and makes the drive perceive
nothing (an unreachable detection threshold), so it contributes zero to every option. Weight genes
still drift, which is harmless when the drive can never point anywhere.

Throwaway spike code (CLAUDE.md §8.3), not imported by anything. Run from a repository root with
`PYTHONPATH=.`:

    python docs/spikes/thermal_drive.py SEED [TICKS] [on|off]

It prints one line per 20-tick sample and a summary line; docs/spikes/thermal-drive.md has the
results.
"""

from __future__ import annotations

import dataclasses
import sys

import numpy as np

from clients.viewer.demo_world import demo_world_config
from core.selection import Selection
from core.world.assembly import build_world

N_ENTITIES = 200
EVERY = 20


def config_for(seed: int, drive: bool):
    config = demo_world_config(N_ENTITIES, seed)
    if drive:
        return config
    return dataclasses.replace(
        config, thermal=dataclasses.replace(config.thermal, detection_threshold=1e30)
    )


def main() -> None:
    seed = int(sys.argv[1])
    ticks = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
    drive = (sys.argv[3] if len(sys.argv) > 3 else "on") == "on"
    config = config_for(seed, drive)
    world = build_world(config, seed=seed)
    year = config.climate.seasons.year_length
    equator = config.climate.equator_y
    previous: set[int] = set()
    deaths = {"warm": 0, "cold": 0}
    rows_out = []
    for tick in range(0, ticks + 1, EVERY):
        if tick:
            world.loop.advance(EVERY)
        alive = world.store.alive
        ids = set(world.store.row_ids()[alive].tolist())
        # The season over the interval just run: sin > 0 is the warm half (#159's phase convention).
        phase = np.sin(2.0 * np.pi * (tick - EVERY / 2) / year)
        if tick:
            deaths["warm" if phase > 0 else "cold"] += len(previous - ids)
        previous = ids
        born = Selection.from_mask(alive & (world.store.age >= 0))
        if not len(born):
            print(f"seed={seed} drive={drive} tick={tick} extinct", flush=True)
            break
        rows = born.to_indices()
        x, y = world.store.x[rows], world.store.y[rows]
        latitude = float(np.mean(np.abs(y - equator)))
        experienced = float(np.mean(world.climate.temperature_at(x, y)))
        rows_out.append((tick, len(rows), latitude, experienced, np.sin(2.0 * np.pi * tick / year)))
        print(
            f"seed={seed} drive={drive} tick={tick} n={len(rows)} lat={latitude:.2f} "
            f"T={experienced:.3f} season={rows_out[-1][4]:+.2f}",
            flush=True,
        )
    data = np.array(rows_out)
    # Skip the first year: founders are scattered uniformly and are still sorting themselves.
    settled = data[data[:, 0] >= year]
    if len(settled) > 3:
        season = settled[:, 4]
        lat_corr = float(np.corrcoef(season, settled[:, 2])[0, 1])
        temp_corr = float(np.corrcoef(season, settled[:, 3])[0, 1])
        # Peak-to-trough swing of occupied latitude, averaged over the settled years.
        warm, cold = settled[season > 0.5], settled[season < -0.5]
        lat_swing = float(warm[:, 2].mean() - cold[:, 2].mean()) if len(warm) and len(cold) else float("nan")
    else:
        lat_corr = temp_corr = lat_swing = float("nan")
    print(
        f"SUMMARY seed={seed} drive={drive} final_n={int(data[-1, 1])} "
        f"mean_lat={settled[:, 2].mean() if len(settled) else float('nan'):.2f} "
        f"lat_season_corr={lat_corr:+.3f} lat_swing_warm_minus_cold={lat_swing:+.2f} "
        f"T_season_corr={temp_corr:+.3f} deaths_warm={deaths['warm']} deaths_cold={deaths['cold']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
