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

The fourth argument picks the world: `demo` (the default, and what §§1-4 of the results ran on) or
`temperate`, the same world under a climate that straddles neutral (#261, §5 of the results).

Throwaway spike code (CLAUDE.md §8.3), not imported by anything. Run from a repository root with
`PYTHONPATH=.`:

    python docs/spikes/thermal_drive.py SEED [TICKS] [on|off] [demo|temperate]

It prints one line per 20-tick sample and a summary line; docs/spikes/thermal-drive.md has the
results.
"""

from __future__ import annotations

import dataclasses
import sys

import numpy as np

from clients.viewer.demo_world import demo_world_config, temperate_world_config
from core.selection import Selection
from core.world.assembly import build_world

N_ENTITIES = 200
EVERY = 20


WORLDS = {"demo": demo_world_config, "temperate": temperate_world_config}


def config_for(seed: int, drive: bool, world: str = "demo"):
    config = WORLDS[world](N_ENTITIES, seed)
    if drive:
        return config
    return dataclasses.replace(
        config, thermal=dataclasses.replace(config.thermal, detection_threshold=1e30)
    )


def main() -> None:
    seed = int(sys.argv[1])
    ticks = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
    drive = (sys.argv[3] if len(sys.argv) > 3 else "on") == "on"
    world_name = sys.argv[4] if len(sys.argv) > 4 else "demo"
    config = config_for(seed, drive, world_name)
    neutral = config.metabolism.neutral_temperature
    world = build_world(config, seed=seed)
    year = config.climate.seasons.year_length
    equator = config.climate.equator_y
    previous: set[int] = set()
    deaths = {"warm": 0, "cold": 0}
    rows_out = []
    # id -> latitude at the previous sample, for the living, so movement can be told from turnover.
    last_latitude: dict[int, float] = {}
    plant_rows = np.arange(world.plants.biomass.shape[0]) * world.terrain.cell_size
    plant_latitude = np.abs(plant_rows - equator)[:, None]
    for tick in range(0, ticks + 1, EVERY):
        if tick:
            world.loop.advance(EVERY)
        alive = world.store.alive
        row_ids = world.store.row_ids()
        ids = set(row_ids[alive].tolist())
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
        each_latitude = np.abs(y - equator)
        latitude = float(np.mean(each_latitude))
        # Mean latitude change of animals alive at both this sample and the last: movement alone,
        # with births and deaths excluded. A range shift made of this is migration; one made of
        # the difference between it and the population mean is demography.
        now = dict(zip(row_ids[rows].tolist(), each_latitude.tolist()))
        stayed = [now[i] - last_latitude[i] for i in now.keys() & last_latitude.keys()]
        moved = float(np.mean(stayed)) if stayed else 0.0
        last_latitude = now
        # Where the food is: biomass-weighted mean latitude of the standing crop.
        biomass = world.plants.biomass
        forage_latitude = float((biomass * plant_latitude).sum() / max(biomass.sum(), 1e-12))
        here = world.climate.temperature_at(x, y)
        experienced = float(np.mean(here))
        # How far from neutral the living stand: what the drive exists to reduce.
        off_neutral = float(np.mean(np.abs(here - neutral)))
        rows_out.append(
            (tick, len(rows), latitude, experienced, np.sin(2.0 * np.pi * tick / year),
             off_neutral, moved, forage_latitude)
        )
        print(
            f"seed={seed} world={world_name} drive={drive} tick={tick} n={len(rows)} "
            f"lat={latitude:.2f} moved={moved:+.3f} forage_lat={forage_latitude:.2f} "
            f"T={experienced:.3f} offN={off_neutral:.2f} season={rows_out[-1][4]:+.2f}",
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
        lat_amplitude, lat_lag = annual_harmonic(settled[:, 0], settled[:, 2], year)
        # Survivors' movement integrated into a latitude of its own: the range shift there would be
        # if nobody were born or died. Detrended, because a slow net drift is not a season.
        movement = np.cumsum(settled[:, 6])
        move_amplitude, move_lag = annual_harmonic(settled[:, 0], movement, year, detrend=True)
        forage_amplitude, forage_lag = annual_harmonic(settled[:, 0], settled[:, 7], year)
        off_neutral = float(settled[:, 5].mean())
    else:
        lat_corr = temp_corr = lat_swing = lat_amplitude = lat_lag = off_neutral = float("nan")
        move_amplitude = move_lag = forage_amplitude = forage_lag = float("nan")
    print(
        f"SUMMARY seed={seed} world={world_name} drive={drive} final_n={int(data[-1, 1])} "
        f"min_n={int(data[:, 1].min())} "
        f"mean_lat={settled[:, 2].mean() if len(settled) else float('nan'):.2f} "
        f"lat_season_corr={lat_corr:+.3f} lat_swing_warm_minus_cold={lat_swing:+.2f} "
        f"lat_harmonic={lat_amplitude:.2f} lat_lag={lat_lag:.0f} "
        f"move_harmonic={move_amplitude:.2f} move_lag={move_lag:.0f} "
        f"forage_harmonic={forage_amplitude:.2f} forage_lag={forage_lag:.0f} "
        f"off_neutral={off_neutral:.2f} "
        f"T_season_corr={temp_corr:+.3f} deaths_warm={deaths['warm']} deaths_cold={deaths['cold']}",
        flush=True,
    )


def annual_harmonic(ticks, series, year, detrend=False):
    """Amplitude and lag of a series' one-year cycle, fitted on sin and cos together.

    So a range shift that lags the season is counted at its full size rather than shrunk by a
    correlation against sin alone. Amplitude in the series' units; lag in ticks after sin peaks
    (the summer solstice). A lag near half a year is a cycle running the other way round.
    """
    angle = 2.0 * np.pi * ticks / year
    columns = [np.sin(angle), np.cos(angle), np.ones(len(ticks))]
    if detrend:
        columns.append(ticks / year)
    (b_sin, b_cos, *_), *_ = np.linalg.lstsq(np.column_stack(columns), series, rcond=None)
    return float(np.hypot(b_sin, b_cos)), float((np.arctan2(-b_cos, b_sin) / (2.0 * np.pi) * year) % year)

if __name__ == "__main__":
    main()
