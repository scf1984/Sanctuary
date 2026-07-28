import numpy as np
import pytest

from core.genetics.inheritance import inherit_genes


class TestValidation:
    def test_rejects_inherit_gain_of_exactly_one(self):
        a = np.array([[1.0]], dtype=np.float32)
        b = np.array([[2.0]], dtype=np.float32)
        with pytest.raises(ValueError):
            inherit_genes(a, b, inherit_gain=1.0, rng=np.random.default_rng(0))

    def test_rejects_inherit_gain_below_one(self):
        a = np.array([[1.0]], dtype=np.float32)
        b = np.array([[2.0]], dtype=np.float32)
        with pytest.raises(ValueError):
            inherit_genes(a, b, inherit_gain=0.5, rng=np.random.default_rng(0))

    def test_rejects_mismatched_parent_shapes(self):
        a = np.zeros((2, 3), dtype=np.float32)
        b = np.zeros((3, 3), dtype=np.float32)
        with pytest.raises(ValueError):
            inherit_genes(a, b, inherit_gain=1.5, rng=np.random.default_rng(0))


class TestOffspringLandWithinClampRange:
    """Property test (CLAUDE.md §6, §8.1): for any parent pair and any valid inherit_gain,
    offspring genes always land in [min(a, b) / inherit_gain, max(a, b) * inherit_gain].
    """

    @pytest.mark.parametrize("seed", range(20))
    def test_offspring_within_clamp_range_across_random_inputs(self, seed):
        rng = np.random.default_rng(seed)
        n_pairs, n_genes = 50, 6
        parent_a = rng.uniform(0.0, 10.0, size=(n_pairs, n_genes)).astype(np.float32)
        parent_b = rng.uniform(0.0, 10.0, size=(n_pairs, n_genes)).astype(np.float32)
        inherit_gain = float(rng.uniform(1.01, 4.0))

        offspring = inherit_genes(parent_a, parent_b, inherit_gain, rng)

        low = np.minimum(parent_a, parent_b) / inherit_gain
        high = np.maximum(parent_a, parent_b) * inherit_gain
        assert (offspring >= low - 1e-4).all()
        assert (offspring <= high + 1e-4).all()

    def test_identical_parents_produce_the_shared_value_exactly(self):
        rng = np.random.default_rng(0)
        value = np.array([[3.0, 5.0]], dtype=np.float32)
        offspring = inherit_genes(value, value, inherit_gain=2.0, rng=rng)
        np.testing.assert_allclose(offspring, value)

    def test_zero_valued_genes_stay_at_zero(self):
        rng = np.random.default_rng(0)
        zeros = np.zeros((4, 3), dtype=np.float32)
        offspring = inherit_genes(zeros, zeros, inherit_gain=1.5, rng=rng)
        assert (offspring == 0.0).all()

    def test_output_dtype_is_float32(self):
        rng = np.random.default_rng(0)
        a = np.array([[1.0, 2.0]], dtype=np.float32)
        b = np.array([[3.0, 4.0]], dtype=np.float32)
        offspring = inherit_genes(a, b, inherit_gain=1.5, rng=rng)
        assert offspring.dtype == np.float32


class TestBoundedLoopAlwaysTerminates:
    """CLAUDE.md's explicit ask: fix the prototype's unbounded `while True` rejection loop —
    no code path may loop unboundedly, however adversarial the draw.
    """

    def test_terminates_and_clamps_even_when_every_draw_misses_the_range(self):
        class AlwaysOutOfRangeRNG:
            """Stands in for a Generator whose every draw lands outside [low, high], forcing
            every resample round to fire so the fixed round cap -- not luck -- is what stops the
            loop."""

            def normal(self, loc, scale):
                # Always far above `high`, regardless of loc/scale, so out_of_range never clears.
                return np.asarray(loc) + 1e6

        parent_a = np.array([[1.0, 2.0]], dtype=np.float32)
        parent_b = np.array([[3.0, 4.0]], dtype=np.float32)
        inherit_gain = 2.0

        offspring = inherit_genes(parent_a, parent_b, inherit_gain, AlwaysOutOfRangeRNG())

        low = np.minimum(parent_a, parent_b) / inherit_gain
        high = np.maximum(parent_a, parent_b) * inherit_gain
        # The fallback clamp landed exactly on the upper bound, since every draw overshot high.
        np.testing.assert_allclose(offspring, high)
        assert (offspring >= low).all()
        assert (offspring <= high).all()


class TestDirectionalSelectionShiftsPopulationMean:
    """Statistical test (CLAUDE.md §6, §8.1): selecting the highest-valued individuals to breed,
    generation after generation, should trend the population mean upward. Never asserts an exact
    value — only the direction, and only that it holds in most of several seeded runs, since a
    single run's variance (CLAUDE.md §2.2) can obscure the trend by chance.
    """

    @staticmethod
    def _run_selection_experiment(seed, generations=25, population_size=40):
        rng = np.random.default_rng(seed)
        # Founders start with some spread, not identical values -- selection needs variance to
        # act on, and identical founders paired with themselves would draw a zero-stddev
        # offspring forever (see inherit_genes: stddev is derived from parental disagreement).
        population = (
            rng.normal(1.0, 0.15, size=(population_size, 1)).clip(0.05, None).astype(np.float32)
        )
        starting_mean = float(population.mean())
        inherit_gain = 1.2

        for _ in range(generations):
            order = np.argsort(population[:, 0])[::-1]
            breeders = population[order[: population_size // 2]]
            parent_a = breeders
            parent_b = breeders[rng.permutation(len(breeders))]
            offspring = inherit_genes(parent_a, parent_b, inherit_gain, rng)
            population = np.concatenate([offspring, offspring], axis=0)

        return starting_mean, float(population.mean())

    def test_mean_trends_upward_under_sustained_directional_selection(self):
        results = [self._run_selection_experiment(seed) for seed in range(5)]
        increased = sum(final > starting for starting, final in results)
        assert increased >= 4, f"expected upward drift in most seeded runs, got (start, end): {results}"
