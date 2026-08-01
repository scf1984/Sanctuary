"""What one tick of movement costs, at the population §2.1's budget is written against.

Run from a repository root (or worktree) with `PYTHONPATH=.`; it prints one line per population
size. Written to run **unchanged on master and on the branch**, because the point is the
comparison and a benchmark that only exists on one side of a change measures nothing (§8.5).
`Movement.step` took a scalar pace before #203 and a per-entity urge after it, so the call is
built from the config's own fields rather than hardcoded.

The population is synthetic and stationary in size — no births, no deaths — because momentum
changes where animals go and therefore how many survive, so timing two *worlds* would compare
populations rather than code.
"""

from __future__ import annotations

import time

import numpy as np

from clients.viewer.demo_world import build_demo_world
from core.selection import Selection


def timed(world, population, repeats):
    """Seconds per `Movement.step` over `population`, best of `repeats`."""
    target_x, target_y = world.behaviour.chosen_target(population)
    if hasattr(world.movement.config, "haste_gene"):
        # Post-#203: a per-entity urge, which the module converts to a pace itself.
        extra = world.behaviour.chosen_urge(population).astype(np.float64)
    else:
        extra = world.movement.config.walking_pace

    best = float("inf")
    for _ in range(repeats):
        started = time.perf_counter()
        world.movement.step(population, target_x, target_y, extra)
        best = min(best, time.perf_counter() - started)
    return best


def main():
    for n in (1_000, 10_000, 100_000):
        world = build_demo_world(seed=1, n_entities=n)
        population = Selection.from_mask(world.store.alive)
        # One scored tick, so every animal carries a real heading and a real urge rather than the
        # zeros `allocate` leaves — a population that all chose nothing would walk nowhere.
        world.behaviour.choose(population, np.random.default_rng(0))
        seconds = timed(world, population, repeats=5)
        print(f"{n:>7} entities   {seconds * 1e3:7.1f} ms/step   {n / seconds / 1e6:5.2f} M/s")


if __name__ == "__main__":
    main()
