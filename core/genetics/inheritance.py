"""Trait inheritance with mutation as bounded drift (CLAUDE.md §2.5, issues #14 and #104).

An offspring gene is a **logistic** draw centered on the parental mean, spread by how much the
parents differ but never by less than the offspring's own `mutability` gene, bounded to a drift
range `drift_margin` spreads outside the parental min/max.

Three things about that sentence were decided in #104, and each replaces something that was wrong:

**The spread has a floor, and the floor is a gene.** The original rule spread the draw by parental
disagreement alone, so identical parents produced an identical offspring — zero variance. A closed
population therefore converged, and the more alike its members became the smaller the spread and the
faster they became alike: a feedback loop ending in a permanently frozen lineage where nothing new
ever appears. `docs/spikes/speciation-drift.md` measured exactly that ("each pool converges
internally, so the gap between the two pools is largely set early and then holds"). Carrying the
floor as a gene rather than as a world constant means a lineage evolves its own **evolvability** —
and that needs no energy cost to bound it, because high mutability produces more unfit offspring, so
a stable environment pushes it down on its own.

**The draw is logistic rather than Gaussian.** The logistic is the difference of two independent
Gumbel variables, and Gumbel is the extreme-value limit: if a parent produces a hundred eggs and two
survive, the survivors' traits are that brood's *extremes*, and §2.1 already compresses reproduction
deliberately rather than simulating each egg. The *two-way* form is what matters — a one-sided Gumbel
has its mean above its mode, so every trait would ratchet upward each generation regardless of
selection, and the metabolic budget would be fighting a built-in bias rather than a neutral one. What
it buys over a Gaussian is **fatter tails**, so a lineage can leave a local optimum in one jump
instead of only crawling out of it.

**The drift range is additive, so it is valid on the whole real line.** Genes are signed (#104): a
cue signature or an aversion direction carries information in its sign. The original range,
`[min / gain, max * gain]`, is multiplicative and therefore *inverts* for negative values — two
parents at -3 and -2 produced `low = -2, high = -3` — which made signed genes impossible rather than
merely awkward. An additive margin in units of the draw's own spread is translation-invariant, so a
gene at -5 drifts exactly as one at +5.

**What the clamp is for.** Not bounding traits — energy and selection do that (CLAUDE.md §2.5): a
trait that drifts upward costs more upkeep, so its bearer starves sooner and leaves fewer offspring.
The clamp is a numerical backstop on the draw's tail, and expressing its width in spreads rather than
in absolute units is what keeps it one: it scales with whatever the draw is actually doing, so it can
never quietly become the thing that decides where a lineage may go.

The legacy version drew in a `while True` rejection loop with no escape (CLAUDE.md §1) -- a
pathological (mean, spread, bounds) triple could spin forever. This version draws a full batch,
resamples only the rows still out of range, for a fixed number of rounds, then clamps whatever
remains. The parental mean is always inside the clamp range (proof in `inherit_genes` below), so
clamping only ever touches the rare tail that missed every resampling round -- the loop always
terminates in bounded time regardless of input.
"""

from __future__ import annotations

import numpy as np

# Each round independently redraws only the still-out-of-range rows, so the chance of surviving
# every round shrinks fast (this is a draw against a range that always contains the mean). 8 rounds
# makes that chance negligible without a probability computation to justify a tighter or looser
# figure standing in the way of §8.5 -- the fallback clamp is a correctness backstop, not a
# distribution shortcut, so 8 vs. some other small constant does not change correctness, only how
# often the backstop fires.
_MAX_RESAMPLE_ROUNDS = 8

# A logistic distribution's standard deviation is `scale * pi / sqrt(3)`, so dividing by that factor
# gives a draw whose width matches the Gaussian this replaced. Deliberate: #104 changes the
# distribution's *shape* — fatter tails — and matching the width isolates that change, so the drift
# re-measurement in docs/spikes/speciation-drift.md compares like with like rather than reporting a
# spread increase as a tail effect.
_LOGISTIC_SCALE_PER_STDDEV = float(np.sqrt(3.0) / np.pi)

# Parental disagreement as a standard deviation: half the gap between the parents, and the half is
# the one value that leaves a closed pool's variance where it was. For parents drawn from a pool of
# variance s^2, the midpoint carries s^2 / 2 and a draw of stddev k|a - b| adds 2k^2 s^2, since
# E[(a - b)^2] = 2s^2. So one generation multiplies variance by (1/2 + 2k^2), which is 1 exactly at
# k = 1/2 — measured at 0.995 over 200,000 pairs, against 1.4996 for the legacy k = 1/sqrt(2).
#
# That legacy value inflated variance by half again every generation, and the reason nobody saw it is
# that the multiplicative clamp it shipped with was tight enough to crush the excess: the rule
# converged because its two halves were wrong in opposite directions. Loosening the clamp into the
# backstop it was documented as being is what exposed this (#104), and it is why the coefficient is
# derived here rather than ported.
#
# Neutral is the right target rather than merely a safe one: it makes an unselected gene a random walk
# whose step size does not grow, which is what §2.5 means by cue signature drift being a molecular
# clock. Population-level convergence still happens — finite-pool sampling loses about s^2/N per
# generation — and `mutability` is what balances it, so the equilibrium is a real mutation-drift
# balance instead of either collapse or blow-up.
_DISAGREEMENT_PER_GAP = 0.5


def inherit_genes(
    parent_a_genes: np.ndarray,
    parent_b_genes: np.ndarray,
    mutability: np.ndarray,
    drift_margin: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """(n_pairs, n_genes) float32: one offspring gene row per (parent_a, parent_b) row pair.

    parent_a_genes, parent_b_genes: (n_pairs, n_genes) float32, **signed** -- a gene is any real
        number, and how it is read as a phenotype is its expression mode's business
        (`core.genetics.expression`), not this function's. Both arrays are full genotypes, expressed
        or not: this function has no notion of species, so an unexpressed gene drifts exactly like an
        expressed one (CLAUDE.md §2.3).
    mutability: (n_pairs,) float32, >= 0, in each gene's own units. The floor on the draw's spread,
        one value per offspring rather than per gene: it is a single gene, resolved by the caller
        through its magnitude expression mode, so a lineage carries one evolvability rather than one
        per trait. Zero is legal and means a lineage that has evolved its own drift away -- an
        outcome it reached, not a rule imposed on it. Negative is a caller bug and raises (§8.7).
    drift_margin: float, > 0, in units of the draw's spread. Offspring land in
        `[min(a, b) - drift_margin * spread, max(a, b) + drift_margin * spread]` -- wider than the
        parents' own [min, max] so a population can drift past its founders' range under selection.
        Zero would collapse that margin and forbid drift outright.
    rng: caller-owned, so callers control seeding and reproducibility logging (CLAUDE.md §2.2)
        rather than this function reading global random state.
    """
    if drift_margin <= 0.0:
        raise ValueError(f"drift_margin must be > 0, got {drift_margin}")
    if parent_a_genes.shape != parent_b_genes.shape:
        raise ValueError(
            f"parent gene shapes must match: {parent_a_genes.shape} vs {parent_b_genes.shape}"
        )
    if mutability.shape != (parent_a_genes.shape[0],):
        raise ValueError(
            f"mutability must hold one value per pair, expected {(parent_a_genes.shape[0],)}, "
            f"got {mutability.shape}"
        )
    if (mutability < 0.0).any():
        raise ValueError("mutability must be non-negative; it is the spread of a draw")

    mean = (parent_a_genes + parent_b_genes) / 2.0
    # How far this offspring's genes may wander: the parents' own disagreement where they disagree,
    # and the lineage's evolvability where they do not. Without the floor, agreement between parents
    # is inherited as an inability to vary at all (see the module docstring).
    spread = np.maximum(
        np.abs(parent_a_genes - parent_b_genes) * _DISAGREEMENT_PER_GAP,
        mutability[:, np.newaxis],
    )
    margin = drift_margin * spread
    low = np.minimum(parent_a_genes, parent_b_genes) - margin
    high = np.maximum(parent_a_genes, parent_b_genes) + margin

    # `mean` always satisfies low <= mean <= high: mean is between parent_a and parent_b by
    # construction, low <= min(a, b) <= mean since margin >= 0, and mean <= max(a, b) <= high by the
    # same reasoning. So the fallback clamp below always lands on a value the resampling loop was
    # already trying to reach, never an arbitrary boundary substitute.
    scale = spread * _LOGISTIC_SCALE_PER_STDDEV
    offspring = rng.logistic(mean, scale)
    out_of_range = (offspring < low) | (offspring > high)
    for _ in range(_MAX_RESAMPLE_ROUNDS):
        if not out_of_range.any():
            break
        resampled = rng.logistic(mean, scale)
        offspring = np.where(out_of_range, resampled, offspring)
        out_of_range = (offspring < low) | (offspring > high)

    offspring = np.clip(offspring, low, high)
    return offspring.astype(np.float32)
