"""How close the forward model gets, against worlds actually run (#216).

`core.world.forecast` predicts a world's carrying capacity from its tunings without simulating.
This is the measurement that says how much to trust it — §8.5 forbids the estimate standing in for
the number, and a forecast whose error was never measured would be exactly that.

Run from a repository root with `PYTHONPATH=.`. Slow: it runs each world to settle.
"""

from __future__ import annotations

import dataclasses

from clients.viewer.demo_world import demo_world_config
from core.selection import Selection
from core.world.assembly import build_world
from core.world.forecast import forecast

# Long enough for the population to stop climbing steeply. Not "to equilibrium" — the corrected
# capacity spike (docs/spikes/capacity-growth.md) found no plateau inside 3,000 ticks, so this is
# a settled *slope*, not a settled level, and the comparison is honest about that.
TICKS = 700
SUNS = (4.0, 8.0, 16.0)
SEEDS = (1, 2)


def world_with(solar, seed, founders=200):
    config = demo_world_config(founders, seed)
    return build_world(
        dataclasses.replace(
            config, plants=dataclasses.replace(config.plants, solar_constant=solar)
        ),
        seed=seed,
    )


def main():
    print(f"{'solar':>7}{'seed':>6}{'forecast':>10}{'actual':>9}{'error':>9}")
    for solar in SUNS:
        for seed in SEEDS:
            built = world_with(solar, seed)
            predicted = forecast(built).carrying_capacity
            built.loop.advance(TICKS)
            actual = len(Selection.from_mask(built.store.alive & (built.store.age >= 0)))
            print(
                f"{solar:>7.1f}{seed:>6}{predicted:>10.0f}{actual:>9}"
                f"{(predicted - actual) / actual:>8.0%}"
            )


if __name__ == "__main__":
    main()
