"""Spike for #16: what the scent-signature block does in a live demo world, fenced and not.

Throwaway measurement code (§8.3). It answers the questions the design rests on, before a single
constant is chosen:

1. Within one interbreeding population, how far apart are two animals' signatures (pairwise), and
   how far apart are the two halves a 2-way clustering finds (centroid)? Both set floors that a
   threshold has to clear.
2. Across a fence, how fast do the two sides' signatures drift apart?
3. How much of that depends on how widely founders' signatures are spread?

Run: python -I docs/spikes/speciation_signature.py   (from the repository root, with PYTHONPATH=.)
"""

from __future__ import annotations

import dataclasses
import sys
import time

import numpy as np

sys.path.insert(0, ".")

from core.world.assembly import build_world  # noqa: E402
from worlds.demo import demo_world_config  # noqa: E402

SIGNATURE = tuple(f"signature_{k}" for k in range(8))
CHECKPOINTS = (0, 500, 1000, 2000, 3000)
PEN = (10.0, 10.0, 70.0, 40.0)  # world units, about 28% of the 80x80 map


def kmeans_two(points: np.ndarray, iterations: int = 25) -> np.ndarray:
    """Labels 0/1 from a farthest-point-seeded 2-means. Deterministic, so the spike is too."""
    centre = points.mean(axis=0)
    a = points[np.argmax(((points - centre) ** 2).sum(axis=1))]
    b = points[np.argmax(((points - a) ** 2).sum(axis=1))]
    labels = np.zeros(len(points), dtype=np.int64)
    for _ in range(iterations):
        labels = (((points - b) ** 2).sum(axis=1) < ((points - a) ** 2).sum(axis=1)).astype(np.int64)
        if labels.all() or not labels.any():
            break
        a, b = points[labels == 0].mean(axis=0), points[labels == 1].mean(axis=0)
    return labels


def pairwise(points: np.ndarray, rng: np.random.Generator, n: int = 2000) -> np.ndarray:
    i = rng.integers(0, len(points), n)
    j = rng.integers(0, len(points), n)
    keep = i != j
    return np.linalg.norm(points[i[keep]] - points[j[keep]], axis=1)


def cross(p: np.ndarray, q: np.ndarray, rng: np.random.Generator, n: int = 2000) -> np.ndarray:
    return np.linalg.norm(p[rng.integers(0, len(p), n)] - q[rng.integers(0, len(q), n)], axis=1)


def run(seed: int, signature_range: tuple[float, float], fenced: bool) -> None:
    config = demo_world_config(200, seed)
    ranges = dict(config.founder_gene_ranges)
    ranges.update({name: signature_range for name in SIGNATURE})
    config = dataclasses.replace(config, founder_gene_ranges=ranges)
    world = build_world(config, seed=seed)
    names = world.genes.vocabulary.names
    columns = np.array([names.index(name) for name in SIGNATURE])
    if fenced:
        world.barriers.enclose(*PEN)
    rng = np.random.default_rng(1_000 + seed)
    label = f"seed {seed}, signatures {signature_range}, {'fenced' if fenced else 'open'}"
    print(f"\n## {label}")
    print("tick  living  pair p50  pair p90  2means sep  inside  cross-fence centroid  cross pair p50")
    done = 0
    started = time.perf_counter()
    for checkpoint in CHECKPOINTS:
        world.loop.advance(checkpoint - done)
        done = checkpoint
        store = world.store
        rows = np.flatnonzero(store.alive & (store.age >= 0))
        if len(rows) < 4:
            print(f"{checkpoint:>5}  collapsed ({len(rows)} living)")
            return
        sig = world.genetics.heritable_at(rows)[:, columns].astype(np.float64)
        pairs = pairwise(sig, rng)
        labels = kmeans_two(sig)
        sep = (
            np.linalg.norm(sig[labels == 0].mean(axis=0) - sig[labels == 1].mean(axis=0))
            if labels.any() and not labels.all()
            else 0.0
        )
        line = (
            f"{checkpoint:>5}  {len(rows):>6}  {np.median(pairs):8.3f}  {np.percentile(pairs, 90):8.3f}"
            f"  {sep:10.3f}"
        )
        if fenced:
            x, y = store.x[rows], store.y[rows]
            inside = (x > PEN[0]) & (x < PEN[2]) & (y > PEN[1]) & (y < PEN[3])
            if inside.sum() >= 2 and (~inside).sum() >= 2:
                centroid = np.linalg.norm(sig[inside].mean(axis=0) - sig[~inside].mean(axis=0))
                line += (
                    f"  {int(inside.sum()):>6}  {centroid:20.3f}"
                    f"  {np.median(cross(sig[inside], sig[~inside], rng)):14.3f}"
                )
            else:
                line += f"  {int(inside.sum()):>6}  (one side empty)"
        print(line, flush=True)
    print(f"({time.perf_counter() - started:.0f} s)")


if __name__ == "__main__":
    for seed in (1, 2):
        for signature_range in ((0.0, 1.0), (0.45, 0.55)):
            run(seed, signature_range, fenced=True)
    run(1, (0.0, 1.0), fenced=False)
