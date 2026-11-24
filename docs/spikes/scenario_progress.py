"""Spike for #317: what the divergence reading shows a player, fenced and not, on the demo world.

Throwaway measurement code (§8.3). It answers what the scenario's progress bar and its briefing
rest on:

1. What does `Speciation.divergence` read for a species nobody has isolated? `propose` halves even
   one well-mixed population, so the baseline is not zero, and the panel has to say what it is.
2. After a player fences a group in, how does the reading move, and how long until the check
   splits it, in ticks, at the live rate and at the fastest rate the demo world offers?

The pen is what a first-time player is likely to draw: a square around where the animals are,
some way into the run. A pen is placed on the densest `PEN_SIDE` square of the map, since a fixed
square in the middle of the map came out empty or on water in three seeds of four.

Run: python -I docs/spikes/scenario_progress.py SEED fenced|open TICKS [PEN_SIDE]   (from the
repository root). Prints one line per speciation check, and the split tick if there was one.
"""

from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, ".")

from worlds.demo import DEMO_PACING, build_demo_world  # noqa: E402

FENCE_AT = 1_500  # ticks: the founding boom has settled into a population a player can read


def densest_square(x: np.ndarray, y: np.ndarray, side: int, extent: int) -> tuple[float, ...]:
    """The `side`-unit square, on whole world units, holding the most of the points given."""
    counts, _, _ = np.histogram2d(y, x, bins=extent, range=[[0, extent], [0, extent]])
    summed = np.zeros((extent + 1, extent + 1))
    summed[1:, 1:] = counts.cumsum(axis=0).cumsum(axis=1)
    window = (
        summed[side:, side:] - summed[:-side, side:] - summed[side:, :-side] + summed[:-side, :-side]
    )
    row, col = np.unravel_index(np.argmax(window), window.shape)
    return float(col), float(row), float(col + side), float(row + side)


def main(seed: int, fenced: bool, ticks: int, side: int) -> None:
    world = build_demo_world(seed=seed, n_entities=200)
    every = world.config.speciation.every_n_ticks
    store = world.store
    started = time.perf_counter()
    split_at = None
    pen = None
    while world.loop.tick_count < ticks and not world.loop.collapsed:
        world.loop.advance(every)
        tick = world.loop.tick_count
        living = store.alive & (store.age >= 0)
        if fenced and pen is None and tick >= FENCE_AT:
            extent = int(world.terrain.world_width)
            pen = densest_square(store.x[living], store.y[living], side, extent)
            world.barriers.enclose(*pen)
            print(f"fence at {tick}: {pen}", flush=True)
        if pen is not None:
            inside = (
                living
                & (store.x >= pen[0]) & (store.x < pen[2])
                & (store.y >= pen[1]) & (store.y < pen[3])
            )
            pen_now = int(inside.sum())
        else:
            pen_now = -1
        reading = world.speciation.divergence.get(0, float("nan"))
        n_species = world.species.n_species
        print(
            f"tick {tick} living {int(living.sum())} pen {pen_now} species {n_species} "
            f"divergence {reading:.3f}",
            flush=True,
        )
        if n_species > 1 and split_at is None:
            split_at = world.speciation.lineage.splits()[0].tick
            print(f"split at {split_at}", flush=True)
            break
    elapsed = time.perf_counter() - started
    hours = "-" if split_at is None else f"{split_at / DEMO_PACING.ticks_per_second / 3600:.2f}"
    fastest = (
        "-" if split_at is None else f"{split_at / DEMO_PACING.fastest_ticks_per_second / 3600:.2f}"
    )
    print(
        f"done seed {seed} fenced {fenced} at tick {world.loop.tick_count} in {elapsed:.0f} s; "
        f"split {split_at}; live hours to split {hours}; at fastest {fastest}",
        flush=True,
    )


if __name__ == "__main__":
    main(
        int(sys.argv[1]),
        sys.argv[2] == "fenced",
        int(sys.argv[3]),
        int(sys.argv[4]) if len(sys.argv) > 4 else 16,
    )
