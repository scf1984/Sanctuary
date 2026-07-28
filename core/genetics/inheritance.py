"""Trait inheritance with mutation as bounded drift (CLAUDE.md §2.5, issue #14).

Ports `legacy/traits.py`'s `Trait.__mul__` on its merits: an offspring gene is a Gaussian draw
centered on the parental mean, spread by how much the parents differ, bounded to a drift range
derived from `inherit_gain` around the parental min/max. Widening that range by `inherit_gain`
(rather than clamping to the exact parental spread) is what lets a population's trait mean keep
moving under sustained selection instead of being trapped inside its founders' range forever.

The legacy version drew from this Gaussian in a `while True` rejection loop with no escape
(CLAUDE.md §1) -- a pathological (mean, stddev, bounds) triple could spin forever. This version
draws a full batch, resamples only the rows still out of range, for a fixed number of rounds, then
clamps whatever remains. The parental mean is always inside the clamp range (proof in `inherit_genes`
below), so clamping only ever touches the exponentially rare tail that missed every resampling
round -- the loop always terminates in bounded time regardless of input.
"""

from __future__ import annotations

import numpy as np

# Each round independently redraws only the still-out-of-range rows, so the chance of surviving
# every round shrinks fast (this is a Gaussian draw against a range that always contains the
# mean). 8 rounds makes that chance negligible without a probability computation to justify a
# tighter or looser figure standing in the way of §8.5 -- the fallback clamp is a correctness
# backstop, not a distribution shortcut, so 8 vs. some other small constant does not change
# correctness, only how often the backstop fires.
_MAX_RESAMPLE_ROUNDS = 8


def inherit_genes(
    parent_a_genes: np.ndarray,
    parent_b_genes: np.ndarray,
    inherit_gain: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """(n_pairs, n_genes) float32: one offspring gene row per (parent_a, parent_b) row pair.

    parent_a_genes, parent_b_genes: (n_pairs, n_genes) float32, non-negative -- every gene is a
        magnitude (size, speed, sight, ...), matching the store's zero-initialized `genes` column
        (CLAUDE.md §2.3). Both arrays are full genotypes, expressed or not: this function has no
        notion of species, so an unexpressed gene drifts exactly like an expressed one.
    inherit_gain: float, > 1.0. Offspring land in
        [min(a, b) / inherit_gain, max(a, b) * inherit_gain] -- wider than the parents' own
        [min, max] so a population can drift past its founders' range under selection. 1.0 would
        collapse that margin to zero and forbid any drift; <= 1.0 also risks inverting the range
        for values below 1, which is rejected rather than silently producing low > high
        (CLAUDE.md §8.7).
    rng: caller-owned, so callers control seeding and reproducibility logging (CLAUDE.md §2.2)
        rather than this function reading global random state.
    """
    if inherit_gain <= 1.0:
        raise ValueError(f"inherit_gain must be > 1.0, got {inherit_gain}")
    if parent_a_genes.shape != parent_b_genes.shape:
        raise ValueError(
            f"parent gene shapes must match: {parent_a_genes.shape} vs {parent_b_genes.shape}"
        )

    mean = (parent_a_genes + parent_b_genes) / 2.0
    # Spread scales with parental disagreement -- identical parents draw a zero-variance
    # offspring exactly at the mean; distant parents draw from a wide Gaussian. This is the
    # legacy formula's shape, ported as-is (CLAUDE.md "Trait genetics ... conceptually sound").
    stddev = np.sqrt((parent_a_genes - mean) ** 2 + (parent_b_genes - mean) ** 2)
    low = np.minimum(parent_a_genes, parent_b_genes) / inherit_gain
    high = np.maximum(parent_a_genes, parent_b_genes) * inherit_gain

    # `mean` always satisfies low <= mean <= high: mean is between parent_a and parent_b by
    # construction, low <= min(a, b) <= mean since inherit_gain > 1, and mean <= max(a, b) <= high
    # by the same reasoning. So the fallback clamp below always lands on a value the resampling
    # loop was already trying to reach, never an arbitrary boundary substitute.
    offspring = rng.normal(mean, stddev)
    out_of_range = (offspring < low) | (offspring > high)
    for _ in range(_MAX_RESAMPLE_ROUNDS):
        if not out_of_range.any():
            break
        resampled = rng.normal(mean, stddev)
        offspring = np.where(out_of_range, resampled, offspring)
        out_of_range = (offspring < low) | (offspring > high)

    offspring = np.clip(offspring, low, high)
    return offspring.astype(np.float32)
