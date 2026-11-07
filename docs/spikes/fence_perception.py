"""What sealed regions do to drives, contact and scent, and what they cost (#169, #276, #139).

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

3. **Contact across the wire** (#276). The same fenced run, counting the touching pairs that mating
   and predation are handed after the fence goes up, and how many of them straddle it. Pen
   membership is computed from the rectangle rather than from `Barriers.regions`, so it runs on
   trees that have no regions.

4. **Scent cost with k pens** (#139). On the 256 x 256 bench grid at 100,000 animals, the cue
   field rebuild (`Scent.rebuild`, once per tick) and the shared candidate read
   (`Scent.perceived_at` over the (n, 9) option block, once per `choose`), with 0, 1, 2, 4 and 8
   pens up. Median of 7.

5. **Genetic distance between pen and outside** (#139). #27's fenced run carried on for 1,000
   ticks after the fence, reporting the distance between the two populations' mean heritable
   phenotypes (`core.genetics.distance.centroid_between`) and the same over the cue signature
   genes alone, which carry no cost and drift neutrally, so they are the clean clock.

Usage:
    python docs/spikes/fence_perception.py wire [seeds...]
    python docs/spikes/fence_perception.py choose
    python docs/spikes/fence_perception.py contact [seeds...]
    python docs/spikes/fence_perception.py scent-cost
    python docs/spikes/fence_perception.py distance [seeds...]
"""

from __future__ import annotations

import statistics
import sys
import time
from dataclasses import replace

import numpy as np

from clients.viewer.demo_world import build_demo_world, demo_world_config
from core.ecology import conception as conception_module
from core.ecology import predation as predation_module
from core.genetics.distance import centroid_between
from core.genetics.registry import Unit
from core.selection import Selection
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


def contact_run(seed: int) -> dict[str, tuple[int, int]]:
    """{verb: (pairs straddling the wire, all pairs)} after the fence goes up."""
    world = build_demo_world(seed=seed, n_entities=300)
    _corners, (r0, r1, c0, c1) = pen_bounds(world)
    counts: dict[str, list[int]] = {}

    def wrap(module, verb):
        original = module.pair_by_contact

        def counted(x, y, rows, contact_range, rng, *rest):
            first, second = original(x, y, rows, contact_range, rng, *rest)
            if world.barriers.any_blocked and first.size:

                def penned(which):
                    r, c = world.terrain.cell_indices(x[which], y[which])
                    return (r >= r0) & (r < r1) & (c >= c0) & (c < c1)

                tally = counts.setdefault(verb, [0, 0])
                tally[0] += int((penned(first) != penned(second)).sum())
                tally[1] += int(first.size)
            return first, second

        module.pair_by_contact = counted
        return original

    originals = (wrap(conception_module, "mating"), wrap(predation_module, "strikes"))
    try:
        world.loop.advance(200)
        corners, _bounds = pen_bounds(world)
        world.loop.interventions.balance = 1.0e6
        world.loop.interventions.request(Fence(world.barriers, *corners))
        world.loop.advance(401)
    finally:
        conception_module.pair_by_contact, predation_module.pair_by_contact = originals
    return {verb: (tally[0], tally[1]) for verb, tally in counts.items()}


def scent_cost() -> None:
    config = demo_world_config(n_entities=100_000, seed=0)
    config = replace(config, terrain=replace(config.terrain, width=256, height=256))
    world = build_world(config, seed=0)
    population = world.founders
    rng = np.random.default_rng(1)
    headings = world.behaviour.candidate_headings(population, rng)
    x, y = world.behaviour.candidate_positions(population, headings)
    for pens in (0, 1, 2, 4, 8):
        world.barriers.clear()
        # Pens of 40 x 40 cells on a lattice over the middle of the map.
        for i in range(pens):
            low_x, low_y = 20.0 + 56.0 * (i % 4), 60.0 + 70.0 * (i // 4)
            world.barriers.enclose(low_x, low_y, low_x + 40.0, low_y + 40.0)
        rebuild = timed(lambda: world.scent.rebuild(population))
        read = timed(
            lambda: world.scent.perceived_at(world.scent.broadcast(population), x, y)
        )
        print(f"{pens} pens: rebuild {rebuild:.1f} ms, candidate read {read:.1f} ms")


def distance_run(seed: int, ticks: int = 1000) -> dict[str, float]:
    world = build_demo_world(seed=seed, n_entities=300)
    world.loop.advance(200)
    corners, (r0, r1, c0, c1) = pen_bounds(world)
    world.loop.interventions.balance = 1.0e6
    world.loop.interventions.request(Fence(world.barriers, *corners))
    world.loop.advance(1)

    def split():
        living = world.store.alive & (world.store.age >= 0)
        rows, cols = world.terrain.cell_indices(world.store.x, world.store.y)
        penned = (rows >= r0) & (rows < r1) & (cols >= c0) & (cols < c1)
        return Selection.from_mask(living & penned), Selection.from_mask(living & ~penned)

    signature_columns = np.array(
        [
            world.genes.index_of(name, unit=Unit.DIMENSIONLESS)
            for name in world.config.scent_genes.signature_genes
        ]
    )

    def distances():
        inside, outside = split()
        if len(inside) == 0 or len(outside) == 0:
            return float("nan"), float("nan"), len(inside), len(outside)
        whole = float(centroid_between(world.genetics, inside, outside))
        a = world.genetics.heritable(inside)[:, signature_columns].mean(axis=0)
        b = world.genetics.heritable(outside)[:, signature_columns].mean(axis=0)
        return whole, float(np.linalg.norm(a - b)), len(inside), len(outside)

    start = distances()
    world.loop.advance(ticks)
    end = distances()
    return {
        "whole_at_fence": round(start[0], 3),
        "whole_after": round(end[0], 3),
        "signature_at_fence": round(start[1], 4),
        "signature_after": round(end[1], 4),
        "alive_in_out_after": (end[2], end[3]),
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
    elif sys.argv[1] == "scent-cost":
        scent_cost()
    elif sys.argv[1] == "distance":
        seeds = [int(s) for s in sys.argv[2:]] or [1, 2, 3]
        for seed in seeds:
            print(f"seed {seed}: {distance_run(seed)}")
    elif sys.argv[1] == "contact":
        seeds = [int(s) for s in sys.argv[2:]] or [1, 2, 3]
        for seed in seeds:
            print(f"seed {seed}: {contact_run(seed)}")
    else:
        choose_cost()
