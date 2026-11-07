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

    # The same measure at the world's rim, for outsiders: the band within one world unit of the
    # edge, which is what the inner ring is to the wire. The rim clips candidates onto itself, so
    # this is the reference a wire that does the same should match.
    width, height = world.terrain.world_width, world.terrain.world_height
    rim_share = (width * height - (width - 2.0) * (height - 2.0)) / (
        width * height - (r1 - r0) * (c1 - c0) * world.terrain.cell_size**2
    )

    inner, outer, rim = [], [], []
    for _ in range(40):
        world.loop.advance(10)
        living = world.store.alive & (world.store.age >= 0)
        rr, cc = world.terrain.cell_indices(world.store.x[living], world.store.y[living])
        penned = pen[rr, cc]
        if penned.sum() > 0:
            inner.append(inner_ring[rr, cc][penned].mean() / inner_share)
        if (~penned).sum() > 0:
            outer.append(outer_ring[rr, cc][~penned].mean() / outer_share)
            px, py = world.store.x[living][~penned], world.store.y[living][~penned]
            at_rim = (px < 1.0) | (px > width - 1.0) | (py < 1.0) | (py > height - 1.0)
            rim.append(at_rim.mean() / rim_share)

    living = world.store.alive & (world.store.age >= 0)
    rr, cc = world.terrain.cell_indices(world.store.x[living], world.store.y[living])
    penned = pen[rr, cc]
    return {
        "inside_ring_density": float(np.mean(inner)) if inner else float("nan"),
        "outside_ring_density": float(np.mean(outer)) if outer else float("nan"),
        "rim_band_density": float(np.mean(rim)) if rim else float("nan"),
        "alive_inside": int(penned.sum()),
        "alive_outside": int((~penned).sum()),
    }


def indifferent_run(seed: int, n: int = 2000, ticks: int = 400) -> dict[str, float]:
    """The wire band for a population with no drives at all, so only the commitment bonus shapes a
    choice and any departure from 1.0 is the boundary rule's doing (`edge_candidates.py`'s method,
    carried to a fence). Founders spread uniformly and the pen over the middle quarter. Only choice
    and movement run, so nothing eats, ages or dies; density is averaged over the second half."""
    rng = np.random.default_rng(seed)
    world = build_demo_world(seed=seed, n_entities=n)
    alive = world.store.alive
    world.store.x[alive] = rng.uniform(0, world.terrain.world_width, alive.sum())
    world.store.y[alive] = rng.uniform(0, world.terrain.world_height, alive.sum())
    corners, (r0, r1, c0, c1) = pen_bounds(world)
    world.barriers.enclose(*corners)
    behaviour, movement = world.behaviour, world.movement
    behaviour._drives = []

    rows_n, cols_n = world.terrain.heights.shape
    r, c = np.mgrid[0:rows_n, 0:cols_n]
    pen = (r >= r0) & (r < r1) & (c >= c0) & (c < c1)
    inner_ring = pen & ~((r >= r0 + 1) & (r < r1 - 1) & (c >= c0 + 1) & (c < c1 - 1))
    inner_share = inner_ring.sum() / pen.sum()
    # The band within one world unit of the world's edge, as the inner ring is to the wire.
    width, height = world.terrain.world_width, world.terrain.world_height
    rim_share = (width * height - (width - 2.0) * (height - 2.0)) / (
        width * height - (r1 - r0) * (c1 - c0) * world.terrain.cell_size**2
    )

    inner, rim_density = [], []
    for t in range(ticks):
        population = Selection.from_mask(world.store.alive)
        behaviour.choose(population, rng)
        target_x, target_y = behaviour.chosen_target(population)
        movement.step(population, target_x, target_y, behaviour.chosen_urge(population))
        if t >= ticks // 2:
            px, py = world.store.x[world.store.alive], world.store.y[world.store.alive]
            rr, cc = world.terrain.cell_indices(px, py)
            penned = pen[rr, cc]
            inner.append(inner_ring[rr, cc][penned].mean() / inner_share)
            px, py = px[~penned], py[~penned]
            at_rim = (px < 1.0) | (px > width - 1.0) | (py < 1.0) | (py > height - 1.0)
            rim_density.append(at_rim.mean() / rim_share)
    return {
        "inside_ring_density": round(float(np.mean(inner)), 3),
        "rim_band_density": round(float(np.mean(rim_density)), 3),
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

    # The wire's own work in perception. Placing the candidates (`candidate_positions`) is where
    # #278 clips them onto the wire, so it is timed fenced against unfenced on every tree. On
    # #169's tree the gate also marks which candidates are reachable and masks every drive's
    # reading with it (each drive pays about one or two masking passes).
    headings = world.behaviour.candidate_headings(population, rng)
    placing: dict[bool, list[float]] = {False: [], True: []}
    for _ in range(4):
        for up in (False, True):
            fence(up)
            placing[up].append(
                timed(lambda: world.behaviour.candidate_positions(population, headings))
            )
    print(
        f"placing candidates: unfenced {statistics.median(placing[False]):.1f} ms, "
        f"four pens {statistics.median(placing[True]):.1f} ms"
    )
    x, y = world.behaviour.candidate_positions(population, headings)
    candidates = world.behaviour.candidates(population, x, y)
    if hasattr(candidates, "only_reachable"):
        fence(True)
        marking = timed(lambda: world.behaviour.candidates(population, x, y))
        candidates = world.behaviour.candidates(population, x, y)
        masking = timed(lambda: candidates.only_reachable(x))
        print(f"gate, four pens: marking {marking:.1f} ms, one masking pass {masking:.1f} ms")

    # `Movement.step`, which #278 moves from checking the wire inside its walk to asking
    # `Barriers.reach` once before it. Timed from the same scored tick each time.
    world.behaviour.choose(population, rng)
    target_x, target_y = world.behaviour.chosen_target(population)
    urge = world.behaviour.chosen_urge(population).astype(np.float64)
    columns = ("x", "y", "z", "velocity_x", "velocity_y", "energy")
    start = {name: getattr(world.store, name).copy() for name in columns}

    def step() -> None:
        for name in columns:
            getattr(world.store, name)[:] = start[name]
        world.movement.step(population, target_x, target_y, urge)

    stepping: dict[bool, list[float]] = {False: [], True: []}
    for _ in range(4):
        for up in (False, True):
            fence(up)
            stepping[up].append(timed(step, repeats=3))
    print(
        f"Movement.step 100k: unfenced {statistics.median(stepping[False]):.1f} ms, "
        f"four pens {statistics.median(stepping[True]):.1f} ms"
    )


if __name__ == "__main__":
    if sys.argv[1] == "wire":
        seeds = [int(s) for s in sys.argv[2:]] or [1, 2, 3]
        for seed in seeds:
            print(f"seed {seed}: {wire_run(seed)}")
    elif sys.argv[1] == "indifferent":
        seeds = [int(s) for s in sys.argv[2:]] or [1, 2, 3, 4]
        for seed in seeds:
            print(f"seed {seed}: {indifferent_run(seed)}")
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
