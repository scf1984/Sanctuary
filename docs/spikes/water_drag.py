"""What pricing water by depth (#143) does to the demo world, and proof that the plumbing is neutral.

Throwaway spike code (CLAUDE.md §8.3). Run from a repository root with `PYTHONPATH=.`:

    python docs/spikes/water_drag.py hash SEED [TICKS]
    python docs/spikes/water_drag.py run MAP DRAG [TICKS] [REPLICATE]
    python docs/spikes/water_drag.py bench

`hash` builds the demo world at drag 0 and fingerprints its state after TICKS (default 300). It is
written to run **unchanged on master and on the branch** — on master `MovementConfig` has no
`drag`, so there is nothing to zero — which is the point: point `PYTHONPATH` at either tree, and
equal fingerprints say the new plumbing changes nothing until the coefficient bites (§2.8 tier 1).

`run` advances the demo world on map MAP at one drag and reports population, dehydration and how
animals use water. REPLICATE seeds the world's generator separately from its terrain, so one map
can be run many times: the demo world's population over 900 ticks varies between runs of one map
by more than any drag moves it, and a single run per map measures nothing (§2.2). `bench` times `Movement.step` at 100,000 entities, again unchanged on either tree.

**What counts as crossing water.** Drinking puts an animal in the margin of every lake it visits,
so time spent wet says little on its own. A *crossing* is a spell in water — depth above zero,
tick after tick — that ends on dry ground more than `_CROSSING_SPAN` world units from where it
began: the animal went through rather than in and back out. A *deep tick* is an animal-tick spent
in water deeper than `_DEEP`, the demo world's own `max_rooting_depth`, past which nothing grows
and nothing but crossing explains being there.
"""

from __future__ import annotations

import dataclasses
import hashlib
import sys
import time

import numpy as np

from clients.viewer.demo_world import build_demo_world, demo_world_config
from core.selection import Selection
from core.world.assembly import build_world

_CROSSING_SPAN = 4.0
_DEEP = 0.5


def _config(map_seed, n, drag):
    config = demo_world_config(n, map_seed)
    if "drag" in {f.name for f in dataclasses.fields(config.movement)}:
        config = dataclasses.replace(
            config, movement=dataclasses.replace(config.movement, drag=drag)
        )
    elif drag != 0.0:
        raise SystemExit("this tree has no drag to set")
    return config


def fingerprint(seed, ticks):
    world = build_world(_config(seed, 200, 0.0), seed=seed)
    world.loop.advance(ticks)
    digest = hashlib.sha256()
    for name in sorted(vars(world.store)):
        value = getattr(world.store, name)
        if isinstance(value, np.ndarray):
            digest.update(name.encode())
            digest.update(np.ascontiguousarray(value).tobytes())
    for grid in (world.plants.biomass, world.plants.soil_nutrients, world.carrion.mass):
        digest.update(np.ascontiguousarray(grid).tobytes())
    living = int(world.store.alive.sum())
    print(f"seed={seed} ticks={ticks} living={living} sha256={digest.hexdigest()[:16]}")


def run(map_seed, drag, ticks, replicate):
    seed = map_seed if replicate == 0 else 1000 * replicate + map_seed
    world = build_world(_config(map_seed, 200, drag), seed=seed)
    store, water = world.store, world.water
    # Per row: the id the spell belongs to (rows are reused, ids never are), whether it is in
    # water, and where it went in.
    spell_id = np.full(store.capacity, -1, dtype=np.int64)
    in_water = np.zeros(store.capacity, dtype=bool)
    entry_x = np.zeros(store.capacity)
    entry_y = np.zeros(store.capacity)
    deepest = np.zeros(store.capacity)
    crossings = deep_crossings = dips = animal_ticks = deep_ticks = wet_ticks = 0
    minimum = int(store.alive.sum())
    start = time.perf_counter()
    for _ in range(ticks):
        world.loop.advance(1)
        if store.capacity != spell_id.shape[0]:
            grow = store.capacity - spell_id.shape[0]
            spell_id = np.concatenate([spell_id, np.full(grow, -1, dtype=np.int64)])
            in_water = np.concatenate([in_water, np.zeros(grow, dtype=bool)])
            entry_x = np.concatenate([entry_x, np.zeros(grow)])
            entry_y = np.concatenate([entry_y, np.zeros(grow)])
            deepest = np.concatenate([deepest, np.zeros(grow)])
        living = store.alive & (store.age >= 0)
        n = int(living.sum())
        minimum = min(minimum, n)
        if not n:
            break
        rows = np.flatnonzero(living)
        ids = store._row_to_id[rows]
        x = store.x[rows].astype(np.float64)
        y = store.y[rows].astype(np.float64)
        depth = water.depth_at(x, y)
        wet = depth > 0.0
        animal_ticks += n
        wet_ticks += int(wet.sum())
        deep_ticks += int((depth > _DEEP).sum())

        same = spell_id[rows] == ids
        was_wet = in_water[rows] & same
        entering = wet & ~was_wet
        leaving = ~wet & was_wet
        span = np.hypot(x - entry_x[rows], y - entry_y[rows])
        crossed = leaving & (span > _CROSSING_SPAN)
        crossings += int(crossed.sum())
        deep_crossings += int((crossed & (deepest[rows] > _DEEP)).sum())
        deepest[rows] = np.where(wet, np.maximum(np.where(was_wet, deepest[rows], 0.0), depth), 0.0)
        dips += int((leaving & (span <= _CROSSING_SPAN)).sum())
        entry_x[rows] = np.where(entering, x, entry_x[rows])
        entry_y[rows] = np.where(entering, y, entry_y[rows])
        in_water[rows] = wet
        spell_id[rows] = ids

    living = Selection.from_mask(store.alive & (store.age >= 0))
    final = len(living)
    dehydration = float(np.mean(store.dehydration[living.to_mask()])) if final else float("nan")
    per_k = 1000.0 / max(animal_ticks, 1)
    print(
        f"map={map_seed} rep={replicate} drag={drag:g} final={final} min={minimum} "
        f"dehydration={dehydration:.3f} wet%={100 * wet_ticks / max(animal_ticks, 1):.2f} "
        f"deep%={100 * deep_ticks / max(animal_ticks, 1):.3f} "
        f"crossings/k={crossings * per_k:.3f} deep_crossings/k={deep_crossings * per_k:.3f} "
        f"dips/k={dips * per_k:.3f} "
        f"wall={time.perf_counter() - start:.0f}s",
        flush=True,
    )


def bench(n=100_000, repeats=9):
    world = build_demo_world(seed=1, n_entities=n)
    population = Selection.from_mask(world.store.alive)
    world.behaviour.choose(population, np.random.default_rng(0))
    target_x, target_y = world.behaviour.chosen_target(population)
    urge = world.behaviour.chosen_urge(population).astype(np.float64)
    x, y = world.store.x.copy(), world.store.y.copy()
    vx, vy = world.store.velocity_x.copy(), world.store.velocity_y.copy()
    energy = world.store.energy.copy()
    times = []
    for _ in range(repeats):
        # The same step every time: restored state, so each repeat walks the same paths.
        world.store.x[:], world.store.y[:] = x, y
        world.store.velocity_x[:], world.store.velocity_y[:] = vx, vy
        world.store.energy[:] = energy
        started = time.perf_counter()
        world.movement.step(population, target_x, target_y, urge)
        times.append(time.perf_counter() - started)
    print(f"n={n} median={1e3 * float(np.median(times)):.1f} ms best={1e3 * min(times):.1f} ms")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "hash":
        fingerprint(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 300)
    elif mode == "run":
        run(
            int(sys.argv[2]),
            float(sys.argv[3]),
            int(sys.argv[4]) if len(sys.argv) > 4 else 900,
            int(sys.argv[5]) if len(sys.argv) > 5 else 0,
        )
    elif mode == "bench":
        bench()
    else:
        raise SystemExit(__doc__)
