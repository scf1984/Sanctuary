import numpy as np
import pytest

from core.genetics.inheritance import inherit_genes


def _mutability(n_pairs, value=0.0):
    return np.full(n_pairs, value, dtype=np.float32)


class TestValidation:
    def test_rejects_drift_margin_of_zero(self):
        a = np.array([[1.0]], dtype=np.float32)
        b = np.array([[2.0]], dtype=np.float32)
        with pytest.raises(ValueError, match="drift_margin"):
            inherit_genes(a, b, _mutability(1), drift_margin=0.0, rng=np.random.default_rng(0))

    def test_rejects_negative_drift_margin(self):
        a = np.array([[1.0]], dtype=np.float32)
        b = np.array([[2.0]], dtype=np.float32)
        with pytest.raises(ValueError, match="drift_margin"):
            inherit_genes(a, b, _mutability(1), drift_margin=-0.5, rng=np.random.default_rng(0))

    def test_rejects_mismatched_parent_shapes(self):
        a = np.zeros((2, 3), dtype=np.float32)
        b = np.zeros((3, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="shape"):
            inherit_genes(a, b, _mutability(2), drift_margin=0.5, rng=np.random.default_rng(0))

    def test_rejects_mutability_of_the_wrong_length(self):
        a = np.zeros((3, 2), dtype=np.float32)
        b = np.zeros((3, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="mutability"):
            inherit_genes(a, b, _mutability(2), drift_margin=0.5, rng=np.random.default_rng(0))

    def test_rejects_negative_mutability(self):
        """A negative floor would make the draw's scale negative. The caller resolves this from a
        gene read through its magnitude expression mode, so a negative here is a caller bug and is
        refused rather than absorbed (§8.7)."""
        a = np.zeros((2, 2), dtype=np.float32)
        b = np.zeros((2, 2), dtype=np.float32)
        mutability = np.array([0.1, -0.1], dtype=np.float32)
        with pytest.raises(ValueError, match="mutability"):
            inherit_genes(a, b, mutability, drift_margin=0.5, rng=np.random.default_rng(0))


class TestVarianceNeverReachesZero:
    """The defect this issue exists to fix (#104). The previous rule spread the draw by the
    parents' own disagreement alone, so identical parents produced an identical offspring: a
    closed population converged, and the more alike its members became the faster they became
    alike. `docs/spikes/speciation-drift.md` measured that happening.
    """

    def test_identical_parents_still_produce_variation(self):
        rng = np.random.default_rng(0)
        value = np.full((400, 1), 3.0, dtype=np.float32)

        offspring = inherit_genes(value, value, _mutability(400, 0.2), 4.0, rng)

        assert offspring.std() > 0.0, "identical parents produced a clone: evolution has stopped"
        # Centred on the parents, not displaced from them — a one-sided draw would ratchet the
        # trait upward every generation regardless of selection (#104's reason for the two-way form).
        assert abs(float(offspring.mean()) - 3.0) < 0.1

    def test_a_lineage_with_zero_mutability_freezes(self):
        """The floor is a gene, so a lineage may evolve its own evolvability to nothing. That is
        its outcome to reach, not a rule forbidding it — and with no disagreement and no floor
        there is nothing left to draw from."""
        rng = np.random.default_rng(0)
        value = np.full((20, 2), 2.0, dtype=np.float32)

        offspring = inherit_genes(value, value, _mutability(20, 0.0), 4.0, rng)

        np.testing.assert_allclose(offspring, value)

    def test_zero_valued_genes_do_not_stay_at_zero(self):
        """Zero is an ordinary point on the real line now, not a floor (#104's signed genes), so a
        gene sitting at zero drifts in both directions like any other value."""
        rng = np.random.default_rng(0)
        zeros = np.zeros((400, 1), dtype=np.float32)

        offspring = inherit_genes(zeros, zeros, _mutability(400, 0.5), 4.0, rng)

        assert (offspring < 0).any() and (offspring > 0).any()


class TestSignedGenesDriftSymmetrically:
    """Genes live on ℝ (#104). Cue signature and aversion are directions whose sign carries
    information, so the rule may not treat the negative half-line differently from the positive
    one — the old multiplicative range did exactly that, and inverted outright below zero.
    """

    def test_translating_both_parents_translates_the_offspring_distribution(self):
        n = 2000
        a = np.full((n, 1), 1.0, dtype=np.float32)
        b = np.full((n, 1), 2.0, dtype=np.float32)
        shift = 100.0

        here = inherit_genes(a, b, _mutability(n, 0.3), 2.0, np.random.default_rng(7))
        there = inherit_genes(a + shift, b + shift, _mutability(n, 0.3), 2.0, np.random.default_rng(7))

        np.testing.assert_allclose(there, here + shift, rtol=1e-4)

    def test_mirroring_both_parents_mirrors_the_offspring_distribution(self):
        """A *distributional* claim, not a pointwise one: `rng.logistic(loc, scale)` adds the same
        stream of deviations to whatever centre it is given, so mirroring the parents does not
        mirror each individual draw. What must hold is that the negative half-line is not treated
        differently from the positive one, which is a statement about the distribution — so this
        compares central quantiles and leaves the extreme order statistics alone, where a finite
        sample of a fat-tailed distribution is noisiest.
        """
        n = 40_000
        a = np.full((n, 1), 0.5, dtype=np.float32)
        b = np.full((n, 1), 2.5, dtype=np.float32)
        quantiles = np.linspace(0.05, 0.95, 19)

        positive = inherit_genes(a, b, _mutability(n, 0.3), 2.0, np.random.default_rng(11))
        negative = inherit_genes(-a, -b, _mutability(n, 0.3), 2.0, np.random.default_rng(11))

        np.testing.assert_allclose(
            np.quantile(negative, quantiles),
            -np.quantile(positive, 1.0 - quantiles),
            atol=0.05,
        )

    def test_negative_parents_do_not_invert_the_clamp_range(self):
        """The old range was [min / gain, max * gain], which for two negative parents put `low`
        above `high` — the bug that made signed genes impossible."""
        n = 200
        a = np.full((n, 1), -3.0, dtype=np.float32)
        b = np.full((n, 1), -2.0, dtype=np.float32)

        offspring = inherit_genes(a, b, _mutability(n, 0.4), 2.0, np.random.default_rng(3))

        margin = 2.0 * np.maximum(np.abs(a - b) * 0.5, 0.4)
        assert (offspring >= -3.0 - margin - 1e-4).all()
        assert (offspring <= -2.0 + margin + 1e-4).all()


class TestOffspringLandWithinClampRange:
    """Property test (CLAUDE.md §6, §8.1): whatever the parents and the floor, an offspring lands
    within `drift_margin` spreads of the parental min/max. The range is additive and therefore
    valid on the whole real line, where the multiplicative one was not.
    """

    @pytest.mark.parametrize("seed", range(20))
    def test_offspring_within_clamp_range_across_random_inputs(self, seed):
        rng = np.random.default_rng(seed)
        n_pairs, n_genes = 50, 6
        parent_a = rng.uniform(-10.0, 10.0, size=(n_pairs, n_genes)).astype(np.float32)
        parent_b = rng.uniform(-10.0, 10.0, size=(n_pairs, n_genes)).astype(np.float32)
        mutability = rng.uniform(0.0, 2.0, size=n_pairs).astype(np.float32)
        drift_margin = float(rng.uniform(0.1, 4.0))

        offspring = inherit_genes(parent_a, parent_b, mutability, drift_margin, rng)

        spread = np.maximum(
            np.abs(parent_a - parent_b) * 0.5, mutability[:, np.newaxis]
        )
        margin = drift_margin * spread
        low = np.minimum(parent_a, parent_b) - margin
        high = np.maximum(parent_a, parent_b) + margin
        assert (offspring >= low - 1e-3).all()
        assert (offspring <= high + 1e-3).all()

    def test_output_dtype_is_float32(self):
        rng = np.random.default_rng(0)
        a = np.array([[1.0, 2.0]], dtype=np.float32)
        b = np.array([[3.0, 4.0]], dtype=np.float32)
        offspring = inherit_genes(a, b, _mutability(1, 0.1), 1.5, rng)
        assert offspring.dtype == np.float32


class TestTailsAreFatterThanGaussian:
    """#104's reason for the logistic: a Gaussian only crawls, so a lineage sitting in a local
    optimum can only leave it by an accumulation of small steps. The logistic is the difference of
    two Gumbels — the extreme-value form, which is what a brood's surviving extremes are — and its
    tails are exponential rather than squared-exponential, so rare large jumps happen.

    The width is matched to the old Gaussian's deliberately (see `inherit_genes`), so this asserts
    a shape difference rather than a spread difference: the same standard deviation must yield more
    mass beyond three of them.
    """

    def test_more_mass_beyond_three_standard_deviations_than_a_gaussian(self):
        n = 200_000
        rng = np.random.default_rng(19)
        a = np.zeros((n, 1), dtype=np.float32)
        b = np.zeros((n, 1), dtype=np.float32)
        # A large margin so the clamp cannot be what removes the tail being measured.
        offspring = inherit_genes(a, b, _mutability(n, 1.0), 20.0, rng)

        beyond = float((np.abs(offspring) > 3.0).mean())
        gaussian_beyond = 0.0027  # two-tailed, 3 sigma
        assert beyond > 2 * gaussian_beyond, (
            f"tail mass {beyond:.4f} is no fatter than a Gaussian's {gaussian_beyond}"
        )


class TestBoundedLoopAlwaysTerminates:
    """CLAUDE.md's explicit ask: fix the prototype's unbounded `while True` rejection loop —
    no code path may loop unboundedly, however adversarial the draw.
    """

    def test_terminates_and_clamps_even_when_every_draw_misses_the_range(self):
        class AlwaysOutOfRangeRNG:
            """Stands in for a Generator whose every draw lands outside [low, high], forcing every
            resample round to fire so the fixed round cap -- not luck -- is what stops the loop."""

            def logistic(self, loc, scale):
                return np.asarray(loc) + 1e6

        parent_a = np.array([[1.0, 2.0]], dtype=np.float32)
        parent_b = np.array([[3.0, 4.0]], dtype=np.float32)
        mutability = _mutability(1, 0.5)
        drift_margin = 2.0

        offspring = inherit_genes(
            parent_a, parent_b, mutability, drift_margin, AlwaysOutOfRangeRNG()
        )

        spread = np.maximum(np.abs(parent_a - parent_b) * 0.5, 0.5)
        high = np.maximum(parent_a, parent_b) + drift_margin * spread
        # The fallback clamp landed exactly on the upper bound, since every draw overshot high.
        np.testing.assert_allclose(offspring, high, rtol=1e-6)


class TestDirectionalSelectionShiftsPopulationMean:
    """Statistical test (CLAUDE.md §6, §8.1): selecting the highest-valued individuals to breed,
    generation after generation, should trend the population mean upward. Never asserts an exact
    value — only the direction, and only that it holds in most of several seeded runs, since a
    single run's variance (CLAUDE.md §2.2) can obscure the trend by chance.
    """

    @staticmethod
    def _run_selection_experiment(seed, generations=25, population_size=40):
        rng = np.random.default_rng(seed)
        # Founders may now start identical: the mutability floor supplies the variance selection
        # acts on, which under the old rule had to come from a spread in the founders themselves.
        population = np.ones((population_size, 1), dtype=np.float32)
        starting_mean = float(population.mean())

        for _ in range(generations):
            order = np.argsort(population[:, 0])[::-1]
            breeders = population[order[: population_size // 2]]
            parent_a = breeders
            parent_b = breeders[rng.permutation(len(breeders))]
            offspring = inherit_genes(
                parent_a, parent_b, _mutability(len(breeders), 0.05), 2.0, rng
            )
            population = np.concatenate([offspring, offspring], axis=0)

        return starting_mean, float(population.mean())

    def test_mean_trends_upward_under_sustained_directional_selection(self):
        results = [self._run_selection_experiment(seed) for seed in range(5)]
        increased = sum(final > starting for starting, final in results)
        assert increased >= 4, (
            f"expected upward drift in most seeded runs, got (start, end): {results}"
        )

    def test_selection_still_moves_a_population_of_identical_founders(self):
        """Directly the defect's consequence: under the old rule a population of identical
        founders had zero variance forever, so selection had nothing to select on and the mean
        could not move at all."""
        starting, final = self._run_selection_experiment(seed=0)
        assert final > starting
