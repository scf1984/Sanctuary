"""Spike: how far apart do isolated sub-populations drift, and how far apart do mixed ones?

Throwaway spike code (CLAUDE.md §8.3) — not part of `core/`, not imported by anything. It exists
to pick the distance threshold and generation count that
`tests/core/genetics/test_speciation.py::TestIsolationCausesSpeciation` asserts against, so those
constants cite a measurement instead of a guess (CLAUDE.md §8.5).

Unlike `soa_throughput_bench.py` this one imports `core.genetics.inheritance` rather than
reimplementing it — the whole point is to measure the drift the real inheritance rule produces —
so it needs the repo root on `sys.path`. Run from the repo root:

    PYTHONPATH=. python docs/spikes/speciation_drift_spike.py
"""

from __future__ import annotations

import numpy as np

from core.genetics.inheritance import inherit_genes

N_GENES = 6
POPULATION = 40
INHERIT_GAIN = 1.05
SEEDS = 200
GENERATION_CHECKPOINTS = (20, 50, 100)
CANDIDATE_THRESHOLDS = (0.08, 0.10, 0.12, 0.15)


def breed(genes: np.ndarray, rng: np.random.Generator, n_offspring: int) -> np.ndarray:
    """One generation: `n_offspring` young from parents drawn at random, with replacement."""
    parent_a = genes[rng.integers(0, genes.shape[0], size=n_offspring)]
    parent_b = genes[rng.integers(0, genes.shape[0], size=n_offspring)]
    return inherit_genes(parent_a, parent_b, INHERIT_GAIN, rng)


def run(seed: int) -> tuple[dict[int, float], dict[int, float]]:
    """Centroid distance between two sub-populations, isolated vs. interbreeding.

    Both arms share founders, population size, gene count and generation count. The only
    difference is whether genes cross the boundary between the two halves.
    """
    rng = np.random.default_rng(seed)
    founders = rng.uniform(0.5, 1.5, size=(2 * POPULATION, N_GENES)).astype(np.float32)

    left, right = founders[:POPULATION].copy(), founders[POPULATION:].copy()
    mixed = founders.copy()

    isolated_distance: dict[int, float] = {}
    mixed_distance: dict[int, float] = {}
    for generation in range(1, max(GENERATION_CHECKPOINTS) + 1):
        left = breed(left, rng, POPULATION)
        right = breed(right, rng, POPULATION)
        mixed = breed(mixed, rng, 2 * POPULATION)
        if generation in GENERATION_CHECKPOINTS:
            isolated_distance[generation] = float(
                np.linalg.norm(left.mean(axis=0) - right.mean(axis=0))
            )
            # The mixed arm's two "halves" are an arbitrary partition of one gene pool — the
            # control for "would any two subsets of a population look diverged anyway?".
            mixed_distance[generation] = float(
                np.linalg.norm(mixed[:POPULATION].mean(axis=0) - mixed[POPULATION:].mean(axis=0))
            )
    return isolated_distance, mixed_distance


def main() -> None:
    results = [run(seed) for seed in range(SEEDS)]
    print(f"{SEEDS} seeds, population {POPULATION} per sub-population, {N_GENES} genes, "
          f"inherit_gain {INHERIT_GAIN}\n")
    for generation in GENERATION_CHECKPOINTS:
        isolated = np.array([r[0][generation] for r in results])
        mixed = np.array([r[1][generation] for r in results])
        print(f"generation {generation}")
        print(
            f"  isolated centroid distance: min {isolated.min():.3f}  "
            f"p5 {np.percentile(isolated, 5):.3f}  median {np.median(isolated):.3f}  "
            f"p95 {np.percentile(isolated, 95):.3f}"
        )
        print(
            f"  mixed centroid distance:    median {np.median(mixed):.3f}  "
            f"p95 {np.percentile(mixed, 95):.3f}  max {mixed.max():.3f}"
        )
        for threshold in CANDIDATE_THRESHOLDS:
            print(
                f"    threshold {threshold:.2f}: isolated speciate "
                f"{100 * (isolated >= threshold).mean():5.1f}%   "
                f"mixed speciate {100 * (mixed >= threshold).mean():5.1f}%"
            )
        print()


if __name__ == "__main__":
    main()
