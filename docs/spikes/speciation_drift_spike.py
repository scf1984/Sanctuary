"""Spike: how far apart do isolated sub-populations drift, and how far apart do mixed ones?

Throwaway spike code (CLAUDE.md §8.3) — not part of `core/`, not imported by anything. It exists
to pick the distance threshold and generation count that
`tests/core/genetics/test_speciation.py::TestIsolationCausesSpeciation` asserts against, so those
constants cite a measurement instead of a guess (CLAUDE.md §8.5).

Unlike `soa_throughput_bench.py` this one imports `core.genetics.inheritance` rather than
reimplementing it — the whole point is to measure the drift the real inheritance rule produces —
so it needs the repo root on `sys.path`. Run from the repo root:

    PYTHONPATH=. python docs/spikes/speciation_drift_spike.py

**Re-measured for #104**, which changed all three of the things this spike measures: the draw is
logistic rather than Gaussian, its spread is floored by a `mutability` gene rather than being parental
disagreement alone, and the drift range is additive in spreads rather than multiplicative. The first
version of this spike found and correctly read the defect #104 fixes — "each pool converges
internally, so the gap between the two pools is largely set early and then holds" — so the quantity it
now has to report is not only the between-pool distance but the **within-pool spread**, which is what
was collapsing. `mutability = 0` reproduces the old collapse and is kept as the control.
"""

from __future__ import annotations

import numpy as np

from core.genetics.inheritance import inherit_genes

N_GENES = 6
POPULATION = 40
DRIFT_MARGIN = 2.0
SEEDS = 200
GENERATION_CHECKPOINTS = (20, 50, 100)
# 0.0 is the control: no floor at all, which is the rule #104 replaced (bar the draw's shape), and
# it is what shows the collapse. The rest span 0.5%–5% of the founders' gene scale, which is ~1.
MUTABILITY_VALUES = (0.0, 0.005, 0.01, 0.02, 0.05)
CANDIDATE_THRESHOLDS = (0.08, 0.12, 0.20, 0.40)


def breed(
    genes: np.ndarray, rng: np.random.Generator, n_offspring: int, mutability: float
) -> np.ndarray:
    """One generation: `n_offspring` young from parents drawn at random, with replacement."""
    parent_a = genes[rng.integers(0, genes.shape[0], size=n_offspring)]
    parent_b = genes[rng.integers(0, genes.shape[0], size=n_offspring)]
    floor = np.full(n_offspring, mutability, dtype=np.float32)
    return inherit_genes(parent_a, parent_b, floor, DRIFT_MARGIN, rng)


def run(seed: int, mutability: float) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
    """Centroid distance between two sub-populations, isolated vs. interbreeding, plus the
    within-pool spread that decides whether either arm can still move at all.

    Both arms share founders, population size, gene count and generation count. The only
    difference is whether genes cross the boundary between the two halves.
    """
    rng = np.random.default_rng(seed)
    founders = rng.uniform(0.5, 1.5, size=(2 * POPULATION, N_GENES)).astype(np.float32)

    left, right = founders[:POPULATION].copy(), founders[POPULATION:].copy()
    mixed = founders.copy()

    isolated_distance: dict[int, float] = {}
    mixed_distance: dict[int, float] = {}
    within_pool_spread: dict[int, float] = {}
    for generation in range(1, max(GENERATION_CHECKPOINTS) + 1):
        left = breed(left, rng, POPULATION, mutability)
        right = breed(right, rng, POPULATION, mutability)
        mixed = breed(mixed, rng, 2 * POPULATION, mutability)
        if generation in GENERATION_CHECKPOINTS:
            isolated_distance[generation] = float(
                np.linalg.norm(left.mean(axis=0) - right.mean(axis=0))
            )
            # The mixed arm's two "halves" are an arbitrary partition of one gene pool — the
            # control for "would any two subsets of a population look diverged anyway?".
            mixed_distance[generation] = float(
                np.linalg.norm(mixed[:POPULATION].mean(axis=0) - mixed[POPULATION:].mean(axis=0))
            )
            # Standard deviation within one isolated pool, averaged over genes: the variance
            # selection has left to act on. Under the old rule this fell toward zero and took
            # evolvability with it (#104).
            within_pool_spread[generation] = float(left.std(axis=0).mean())
    return isolated_distance, mixed_distance, within_pool_spread


def main() -> None:
    print(
        f"{SEEDS} seeds, population {POPULATION} per sub-population, {N_GENES} genes, "
        f"drift_margin {DRIFT_MARGIN}, logistic draw (#104)"
    )
    for mutability in MUTABILITY_VALUES:
        results = [run(seed, mutability) for seed in range(SEEDS)]
        print(f"\n=== mutability {mutability}" + ("  (control: no floor)" if not mutability else ""))
        for generation in GENERATION_CHECKPOINTS:
            isolated = np.array([r[0][generation] for r in results])
            mixed = np.array([r[1][generation] for r in results])
            spread = np.array([r[2][generation] for r in results])
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
            print(f"  within-pool gene stddev:    median {np.median(spread):.4f}")
            for threshold in CANDIDATE_THRESHOLDS:
                print(
                    f"    threshold {threshold:.2f}: isolated speciate "
                    f"{100 * (isolated >= threshold).mean():5.1f}%   "
                    f"mixed speciate {100 * (mixed >= threshold).mean():5.1f}%"
                )


if __name__ == "__main__":
    main()
