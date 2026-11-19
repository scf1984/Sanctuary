"""Spike for #16: choosing the speciation threshold, in mutation steps, on the demo world.

Throwaway measurement code (§8.3). The threshold trades two things against each other. Fertility
falls linearly with a pair's distance, so a small threshold makes ordinary couples less fertile.
And a split fires only once two groups' centroids are a threshold apart, so a large one makes a
split take longer. This measures both, plus the margin against false splits: how far apart the two
halves a 2-means finds in an *unisolated* population sit.

Run from the repository root: PYTHONPATH=. python docs/spikes/speciation_threshold.py
"""

from __future__ import annotations

import dataclasses
import sys
import time

import numpy as np

sys.path.insert(0, ".")

from core.genetics.speciation import mutation_step, propose  # noqa: E402
from core.selection import Selection  # noqa: E402
from core.world.assembly import build_world  # noqa: E402
from worlds.demo import demo_world_config  # noqa: E402

CHECKPOINTS = (0, 1000, 2000, 3000)
PEN = (10.0, 10.0, 70.0, 40.0)


def run(seed: int, threshold: float, fenced: bool) -> None:
    config = demo_world_config(200, seed)
    config = dataclasses.replace(
        config, speciation=dataclasses.replace(config.speciation, threshold=threshold)
    )
    world = build_world(config, seed=seed)
    if fenced:
        world.barriers.enclose(*PEN)
    speciation = world.speciation
    rng = np.random.default_rng(seed + 7)
    print(f"\n## seed {seed}, threshold {threshold:g}, {'fenced' if fenced else 'open'}")
    print("tick  living  species  fertility  pair/step p50  p90  2means/step  cross-fence/step")
    done, started = 0, time.perf_counter()
    for checkpoint in CHECKPOINTS:
        world.loop.advance(checkpoint - done)
        done = checkpoint
        store = world.store
        rows = np.flatnonzero(store.alive & (store.age >= 0))
        if len(rows) < 4:
            print(f"{checkpoint:>5}  collapsed")
            return
        a, b = rows[rng.integers(0, len(rows), 3000)], rows[rng.integers(0, len(rows), 3000)]
        keep = a != b
        a, b = a[keep], b[keep]
        fertility = float(speciation.compatibility(a, b).mean())
        sig = world.genetics.heritable_columns_at(rows, speciation.columns).astype(np.float64)
        step = mutation_step(world.genetics, rows).astype(np.float64)
        pair_steps = np.linalg.norm(
            world.genetics.heritable_columns_at(a, speciation.columns)
            - world.genetics.heritable_columns_at(b, speciation.columns),
            axis=1,
        ) / (0.5 * (mutation_step(world.genetics, a) + mutation_step(world.genetics, b)))
        living = Selection.from_indices(rows, store.capacity)
        first, second = propose(world.genetics, living, speciation.columns)
        sep = float(
            np.linalg.norm(
                world.genetics.heritable_columns_at(first.to_indices(), speciation.columns).mean(0)
                - world.genetics.heritable_columns_at(second.to_indices(), speciation.columns).mean(0)
            )
            / step.mean()
        )
        line = (
            f"{checkpoint:>5}  {len(rows):>6}  {world.species.n_species:>7}  {fertility:9.3f}"
            f"  {np.median(pair_steps):13.1f}  {np.percentile(pair_steps, 90):5.1f}  {sep:11.1f}"
        )
        if fenced:
            x, y = store.x[rows], store.y[rows]
            inside = (x > PEN[0]) & (x < PEN[2]) & (y > PEN[1]) & (y < PEN[3])
            if inside.sum() > 1 and (~inside).sum() > 1:
                gap = np.linalg.norm(sig[inside].mean(0) - sig[~inside].mean(0)) / step.mean()
                line += f"  {gap:16.1f} ({int(inside.sum())} in)"
        print(line, flush=True)
    print(f"({time.perf_counter() - started:.0f} s)")


if __name__ == "__main__":
    for threshold in (40.0, 80.0, 160.0):
        for seed in (1, 2):
            run(seed, threshold, fenced=False)
    for seed in (1, 2):
        run(seed, 80.0, fenced=True)
