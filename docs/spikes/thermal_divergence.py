"""Do hot and cold ground evolve different animals once tolerance is one gene per side (#176)?

The demo world's whole map sits on the hot side of its neutral temperature (26-29 against 20), so
it can only ever select heat tolerance up and cold tolerance down: there is no cold ground to
diverge on. This builds the demo world unchanged except for two numbers, so that it has both:

- `neutral_temperature` moves to the map's median cell temperature, putting about half the land
  on each side of it;
- `thermoregulation_rate` is re-derived so the mean thermal bill over the land is the same 0.08
  per tick the demo world charges its typical animal. The pressure is the same; only its sign now
  varies in space.

Then it asks whether animals standing on hot ground carry more heat tolerance than animals on cold
ground, and less cold tolerance. Under the symmetric `insulation` damper this could not happen by
construction: one gene, charged identically either side, cannot differ by side.

Since #159 the demo world has seasons and weather. The last argument chooses between the demo
world's seasonal climate (`seasonal`, the default) and the same world with `seasons=None`
(`static`, which reproduces #176's measurement). Either way "hot ground" means the *static* field
— where an animal stands on the annual-mean map — so the two modes classify ground identically and
differ only in what the animals live through.

Since #249 the demo world has a thermal drive, so every mode here runs with animals that can
sense the field and steer by it. thermal-divergence.md §5 has the comparison.

Throwaway spike code (CLAUDE.md §8.3), not imported by anything. Run from a repository root with
`PYTHONPATH=.`:

    python docs/spikes/thermal_divergence.py SEED [TICKS] [EVERY] [seasonal|static]
"""

from __future__ import annotations

import dataclasses
import sys

import numpy as np

from clients.viewer.demo_world import demo_world_config
from core.selection import Selection
from core.world.assembly import build_world
from core.world.climate import Climate
from core.world.terrain import Terrain

N_ENTITIES = 200
# The demo world's typical thermal bill: its linear 0.01 per temperature unit at the 8.0 units its mean
# cell sits from neutral, which the quadratic rate there was derived to preserve.
TYPICAL_BILL = 0.08


def split_world_config(seed, seasonal=True):
    config = demo_world_config(N_ENTITIES, seed)
    static_climate = dataclasses.replace(config.climate, seasons=None)
    if not seasonal:
        config = dataclasses.replace(config, climate=static_climate)
    terrain = Terrain.generate(config.terrain)
    static = Climate(terrain, static_climate)
    temperature = static.temperature
    neutral = float(np.median(temperature))
    mean_squared_deviation = float(np.mean((temperature - neutral) ** 2))
    metabolism = dataclasses.replace(
        config.metabolism,
        neutral_temperature=neutral,
        thermoregulation_rate=TYPICAL_BILL / mean_squared_deviation,
    )
    return dataclasses.replace(config, metabolism=metabolism), neutral, static


def sample(world, neutral, names, static):
    living = Selection.from_mask(world.store.alive & (world.store.age >= 0))
    if not len(living):
        return None
    rows = living.to_indices()
    temperature = static.temperature_at(world.store.x[rows], world.store.y[rows])
    hot = temperature > neutral
    phenotype = world.genetics.expressed(living)
    heat = phenotype[:, names.index("heat_tolerance")]
    cold = phenotype[:, names.index("cold_tolerance")]
    return {
        "n": len(rows),
        "n_hot": int(hot.sum()),
        "heat_on_hot": float(heat[hot].mean()) if hot.any() else float("nan"),
        "heat_on_cold": float(heat[~hot].mean()) if (~hot).any() else float("nan"),
        "cold_on_hot": float(cold[hot].mean()) if hot.any() else float("nan"),
        "cold_on_cold": float(cold[~hot].mean()) if (~hot).any() else float("nan"),
        # Positive when animals on warmer ground lean toward heat tolerance over cold tolerance.
        "corr": float(np.corrcoef(temperature - neutral, heat - cold)[0, 1]),
    }


def main():
    seed = int(sys.argv[1])
    ticks = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
    every = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    seasonal = (sys.argv[4] if len(sys.argv) > 4 else "seasonal") == "seasonal"
    config, neutral, static = split_world_config(seed, seasonal)
    world = build_world(config, seed=seed)
    names = list(world.genes.vocabulary.names)
    print(
        f"seed={seed} mode={'seasonal' if seasonal else 'static'} neutral={neutral:.3f} "
        f"rate={config.metabolism.thermoregulation_rate:.5f}",
        flush=True,
    )
    for tick in range(0, ticks + 1, every):
        if tick:
            world.loop.advance(every)
        row = sample(world, neutral, names, static)
        if row is None:
            print(f"seed={seed} tick={tick} extinct", flush=True)
            return
        print(
            f"seed={seed} tick={tick} n={row['n']} hot_share={row['n_hot'] / row['n']:.2f} "
            f"heat(hot/cold)={row['heat_on_hot']:.3f}/{row['heat_on_cold']:.3f} "
            f"cold(hot/cold)={row['cold_on_hot']:.3f}/{row['cold_on_cold']:.3f} "
            f"corr={row['corr']:+.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
