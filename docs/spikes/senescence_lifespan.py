"""How long animals live once performance traits degrade with age (#239), and whether the world holds.

Throwaway spike code (CLAUDE.md §8.3), not part of `core/`. It produces the numbers in
docs/spikes/senescence.md: the age-at-death distribution and the population on the demo world,
for a given `base_decay`, or with no senescence at all on a checkout from before #239.

Run from a repository root (or worktree) with `PYTHONPATH=.`:

    python docs/spikes/senescence_lifespan.py --seeds 1 2 3 4 5 --ticks 4000 --base-decay 0.002

`--base-decay` is ignored (and must be omitted) on a pre-#239 checkout, where nothing ages.
Deaths are read from the store alone: a row whose id changes or empties across a tick held an
animal that died in it, at the age it had reached. The unborn (negative age) are excluded, since
a gestating row that is released has miscarried rather than lived.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from clients.viewer.demo_world import demo_world_config
from core.world.assembly import build_world


def run(seed: int, ticks: int, n_founders: int, base_decay: float | None) -> dict:
    config = demo_world_config(n_founders, seed)
    if base_decay is not None:
        senescence = dataclasses.replace(config.genetics.senescence, base_decay=base_decay)
        config = dataclasses.replace(
            config, genetics=dataclasses.replace(config.genetics, senescence=senescence)
        )
    world = build_world(config, seed=seed)
    store = world.store
    death_ages: list[int] = []
    oldest_alive = 0
    started = time.perf_counter()
    for _ in range(ticks):
        before_ids = store.row_ids()
        before_age = store.age[: len(before_ids)].copy()
        world.loop.advance(1)
        after_ids = store.row_ids()
        width = len(before_ids)
        died = (before_ids >= 0) & (after_ids[:width] != before_ids) & (before_age >= 0)
        death_ages.extend(before_age[died].tolist())
        if world.loop.collapsed:
            break
    living = store.alive & (store.age >= 0)
    if living.any():
        oldest_alive = int(store.age[living].max())
    ages = np.asarray(death_ages, dtype=np.float64)

    def q(p: float) -> float | None:
        return float(np.percentile(ages, p)) if ages.size else None

    return {
        "seed": seed,
        "base_decay": base_decay,
        "ticks_run": int(world.loop.tick_count),
        "population": int(living.sum()),
        "collapsed": bool(world.loop.collapsed),
        "deaths": int(ages.size),
        "age_p10": q(10),
        "age_p50": q(50),
        "age_p90": q(90),
        "age_max": float(ages.max()) if ages.size else None,
        "deaths_over_1000": int((ages > 1000).sum()),
        "oldest_alive": oldest_alive,
        "seconds": round(time.perf_counter() - started, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--ticks", type=int, default=4000)
    parser.add_argument("--founders", type=int, default=200)
    parser.add_argument("--base-decay", type=float, nargs="*", default=[None])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    jobs = [(seed, args.ticks, args.founders, b) for b in args.base_decay for seed in args.seeds]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(run, *zip(*jobs)):
            print(json.dumps(result), flush=True)
            sys.stdout.flush()


if __name__ == "__main__":
    main()
