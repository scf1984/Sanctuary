"""What a fence does to where penned animals stand, and what the region gate costs (#169).

Throwaway spike code (CLAUDE.md §8.3), not imported by anything. Run from a worktree root with
`PYTHONPATH=.`; it uses only what master already had before #169 (`build_demo_world`, `Fence`), so
the same file measures the tree before the gate and after it.

1. **Against the wire.** #27's fenced run (`tests/core/world/test_fencing.py`): the demo world grown
   for 200 ticks, then a pen fenced over the middle quarter of the map and the world advanced 400
   more. Every 10 ticks it records the share of penned animals standing in the ring of cells just
   inside the wire, and the share of outsiders in the ring just outside it, each divided by that
   ring's share of its region's area. 1.0 is no preference; above 1 is crowding against the wire.
   Also reports who is alive at the end, inside and out, as viability.

2. **`choose()` cost** at 100,000 animals, N = 8, on the 256 x 256 bench grid
   (`docs/spikes/option_sampling_bench.py`'s world), unfenced and with four pens up. Median of 7.

Usage:
    python docs/spikes/fence_perception.py wire [seeds...]
    python docs/spikes/fence_perception.py choose
"""

from __future__ import annotations

import statistics
import sys
import time
from dataclasses import replace

import numpy as np

from clients.viewer.demo_world import build_demo_world, demo_world_config
from core.world.assembly import build_world
from core.world.fence import Fence


def pen_bounds(world):
    width, height = world.terrain.world_width, world.terrain.world_height
    corners = (width * 0.25, height * 0.25, width * 0.75, height * 0.75)
    cell = world.terrain.cell_size
    low_col, high_col = (int(np.floor(corners[i] / cell + 0.5)) for i in (0, 2))
    low_row, high_row = (int(np.floor(corners[i] / cell + 0.5)) for i in (1, 3))
    return corners, (low_row, high_row, low_col, high_col)


def wire_run(seed: int) -> dict[str, float]:
    world = build_demo_world(seed=seed, n_entities=300)
    world.loop.advance(200)
    corners, (r0, r1, c0, c1) = pen_bounds(world)
    world.loop.interventions.balance = 1.0e6
    world.loop.interventions.request(Fence(world.barriers, *corners))
    world.loop.advance(1)

    rows_n, cols_n = world.terrain.heights.shape
    r, c = np.mgrid[0:rows_n, 0:cols_n]
    pen = (r >= r0) & (r < r1) & (c >= c0) & (c < c1)
    inner_ring = pen & ~((r >= r0 + 1) & (r < r1 - 1) & (c >= c0 + 1) & (c < c1 - 1))
    grown = (r >= r0 - 1) & (r < r1 + 1) & (c >= c0 - 1) & (c < c1 + 1)
    outer_ring = grown & ~pen
    inner_share = inner_ring.sum() / pen.sum()
    outer_share = outer_ring.sum() / (~pen).sum()

    inner, outer = [], []
    for _ in range(40):
        world.loop.advance(10)
        living = world.store.alive & (world.store.age >= 0)
        rr, cc = world.terrain.cell_indices(world.store.x[living], world.store.y[living])
        penned = pen[rr, cc]
        if penned.sum() > 0:
            inner.append(inner_ring[rr, cc][penned].mean() / inner_share)
        if (~penned).sum() > 0:
            outer.append(outer_ring[rr, cc][~penned].mean() / outer_share)

    living = world.store.alive & (world.store.age >= 0)
    rr, cc = world.terrain.cell_indices(world.store.x[living], world.store.y[living])
    penned = pen[rr, cc]
    return {
        "inside_ring_density": float(np.mean(inner)) if inner else float("nan"),
        "outside_ring_density": float(np.mean(outer)) if outer else float("nan"),
        "alive_inside": int(penned.sum()),
        "alive_outside": int((~penned).sum()),
    }


def timed(fn, repeats: int = 7) -> float:
    fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def choose_cost() -> None:
    config = demo_world_config(n_entities=100_000, seed=0)
    config = replace(
        config,
        terrain=replace(config.terrain, width=256, height=256),
        behaviour=replace(config.behaviour, n_candidates=8),
    )
    world = build_world(config, seed=0)
    rng = np.random.default_rng(1)
    population = world.founders

    def fence(up: bool) -> None:
        world.barriers.clear()
        if up:
            # Four pens across the middle band, about a fifth of the map between them.
            for i in range(4):
                world.barriers.enclose(16.0 + 60.0 * i, 80.0, 66.0 + 60.0 * i, 176.0)

    # Interleaved within one process, so drift in machine load lands on both columns alike.
    whole: dict[bool, list[float]] = {False: [], True: []}
    for _ in range(4):
        for up in (False, True):
            fence(up)
            whole[up].append(timed(lambda: world.behaviour.choose(population, rng), repeats=3))
    print(
        f"choose() 100k N=8: unfenced {statistics.median(whole[False]):.1f} ms, "
        f"four pens {statistics.median(whole[True]):.1f} ms"
    )

    # The gate's own work, where the tree has one: marking the candidates, and one masking pass
    # (each drive pays about one or two of these).
    if hasattr(world.behaviour, "candidates"):
        fence(True)
        headings = world.behaviour.candidate_headings(population, rng)
        x, y = world.behaviour.candidate_positions(population, headings)
        marking = timed(lambda: world.behaviour.candidates(population, x, y))
        candidates = world.behaviour.candidates(population, x, y)
        masking = timed(lambda: candidates.only_reachable(x))
        print(f"gate, four pens: marking {marking:.1f} ms, one masking pass {masking:.1f} ms")


if __name__ == "__main__":
    if sys.argv[1] == "wire":
        seeds = [int(s) for s in sys.argv[2:]] or [1, 2, 3]
        for seed in seeds:
            print(f"seed {seed}: {wire_run(seed)}")
    else:
        choose_cost()
