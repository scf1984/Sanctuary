"""What a world looks like from outside it, and what a client is allowed to be handed (#30).

The contract is checkable in advance, so these were written against it rather than after it
(§8.1): a sample's shape, the cadence, the flow arithmetic, and — the one that matters most for
§3.1 — that nothing crossing the boundary is a NumPy view a client could hold.
"""

import json

import numpy as np
import pytest

from clients.viewer.demo_world import build_demo_world
from core.selection import Selection
from metrics import MetricHistory, MetricsConfig

EVERY_TICK = MetricsConfig(every_n_ticks=1, history_limit=1_000)


def world(seed=1, n_entities=60, config=EVERY_TICK):
    """A real assembled world with a recorder attached, as `build_demo_world` wires one."""
    built = build_demo_world(seed=seed, n_entities=n_entities)
    built.loop.metrics = MetricHistory(
        built.store, built.genetics, built.plants, built.genes.vocabulary, config
    )
    return built


class TestMetricsConfig:
    @pytest.mark.parametrize("cadence", [0, -1])
    def test_a_non_positive_cadence_is_rejected(self, cadence):
        """At zero the modulus is a division by zero, and the history never fills — a recorder that
        silently records nothing is worse than one that refuses to be built (§8.7)."""
        with pytest.raises(ValueError, match="every_n_ticks"):
            MetricsConfig(every_n_ticks=cadence, history_limit=10)

    @pytest.mark.parametrize("limit", [0, -1])
    def test_a_non_positive_history_limit_is_rejected(self, limit):
        with pytest.raises(ValueError, match="history_limit"):
            MetricsConfig(every_n_ticks=1, history_limit=limit)


class TestNothingCrossesTheBoundaryButPlainValues:
    """§3.1 puts this on a shared machine with clients asking for view information. A metric that
    hands back a NumPy view is one a client will hold, and the fix at that point is a rewrite."""

    def test_a_sample_survives_a_json_round_trip(self):
        built = world()
        built.loop.advance(3)

        restored = json.loads(json.dumps(built.loop.metrics.samples[-1].as_dict()))

        assert restored["tick"] == 3
        assert isinstance(restored["living"], int)
        assert isinstance(restored["expressed_mean"], list)

    def test_no_field_is_a_numpy_array_or_scalar(self):
        """Explicit rather than implied by the JSON test, because `json` accepts a Python float that
        came out of NumPy while a socket's schema will not, and a `np.float32` reads as a float
        everywhere except where it matters."""
        built = world()
        built.loop.advance(1)

        for name, value in built.loop.metrics.samples[-1].as_dict().items():
            flat = value if isinstance(value, (list, tuple)) else [value]
            for item in flat:
                assert not isinstance(item, np.generic), f"{name} carries a NumPy scalar"
                assert not isinstance(item, np.ndarray), f"{name} carries a NumPy array"

    def test_a_sample_is_a_snapshot_and_not_a_window_onto_the_store(self):
        """Held samples must not change as the world runs — a client plotting a history would
        otherwise watch its own past rewrite itself."""
        built = world()
        built.loop.advance(1)
        first = built.loop.metrics.samples[0]
        living_then = first.living
        means_then = first.expressed_mean

        built.loop.advance(40)

        assert first.living == living_then
        assert first.expressed_mean == means_then


class TestCadence:
    def test_it_records_on_the_cadence_and_not_between(self):
        built = world(config=MetricsConfig(every_n_ticks=10, history_limit=1_000))

        built.loop.advance(35)

        assert built.loop.metrics.series("tick") == [10, 20, 30]

    def test_batching_does_not_change_what_is_recorded(self):
        """§2.4 forbids how a client calls `advance` from changing outcomes, and a history is an
        outcome a client reads. One batch of thirty and thirty batches of one must agree."""
        config = MetricsConfig(every_n_ticks=5, history_limit=1_000)
        one_batch = world(config=config)
        one_batch.loop.advance(30)

        many = world(config=config)
        for _ in range(30):
            many.loop.advance(1)

        assert one_batch.loop.metrics.series("tick") == many.loop.metrics.series("tick")

    def test_the_history_is_bounded_and_keeps_the_recent_past(self):
        """A world runs indefinitely (§2.1), so an unbounded list is a leak that only shows up in
        the deployment that matters."""
        built = world(config=MetricsConfig(every_n_ticks=1, history_limit=4))

        built.loop.advance(20)

        assert built.loop.metrics.series("tick") == [17, 18, 19, 20]


class TestWhatASampleSays:
    def test_a_gestating_row_is_counted_apart_from_the_living(self):
        """A gestating row is allocated and carries a negative age (#20), so counting `alive` alone
        would report an unborn young as a member of the population."""
        built = world()
        built.loop.advance(120)
        sample = built.loop.metrics.samples[-1]

        assert sample.living == int((built.store.alive & (built.store.age >= 0)).sum())
        assert sample.gestating == int((built.store.alive & (built.store.age < 0)).sum())
        assert sample.gestating > 0, "the demo world should be breeding by tick 120"

    def test_the_first_sample_reports_no_flows(self):
        """There is no previous reading to difference against, and a first sample reporting the
        whole founding population as conceptions would put a spike at the origin of every plot."""
        built = world()
        built.loop.advance(1)

        assert built.loop.metrics.samples[0].conceptions == 0
        assert built.loop.metrics.samples[0].deaths == 0

    def test_conceptions_and_deaths_are_derived_from_ids_and_occupancy(self):
        """Neither `Conception` nor `Death` reports anything. Ids are never reused, so the number
        issued between two samples is the number of rows allocated, and the shortfall against the
        change in occupancy is the number released — one derivation, nothing to disagree with."""
        built = world()
        built.loop.advance(120)
        history = built.loop.metrics

        issued = built.store.ids_issued - len(built.founders)
        assert sum(history.series("conceptions")) == issued

        occupancy = int(built.store.alive.sum()) - len(built.founders)
        assert sum(history.series("deaths")) == issued - occupancy

    def test_the_closed_loop_is_visible_in_the_three_nutrient_columns(self):
        """Their total is what `nutrients_are_conserved` asserts never moves (§6), so a client
        plotting them sees the loop rather than being told about it."""
        built = world()
        built.loop.advance(30)
        totals = [
            sample.standing_biomass + sample.soil_nutrients + sample.exported_nutrients
            for sample in built.loop.metrics.samples
        ]

        assert totals == pytest.approx([totals[0]] * len(totals), rel=1e-9)

    def test_an_empty_world_still_produces_a_sample_of_the_same_shape(self):
        """A series a client plots against must not change width when the last animal dies."""
        built = world()
        built.store.release(built.store.row_ids()[built.store.alive])

        sample = built.loop.metrics.sample(tick=0)

        assert sample.living == 0
        assert sample.median_energy == 0.0
        assert len(sample.expressed_mean) == len(built.genes.vocabulary.names)


class TestWhatShowsEvolutionHappening:
    """A count cannot: "240 herbivores" reads the same before and after selection moves a
    population's speed by a third. A per-gene mean and spread can, and that is why they are here."""

    def test_the_mean_tracks_the_expressed_phenotype_of_the_living(self):
        built = world()
        living = Selection.from_mask(built.store.alive & (built.store.age >= 0))
        expected = built.genetics.expressed(living).mean(axis=0).tolist()

        sample = built.loop.metrics.sample(tick=0)

        # An absolute tolerance as well as a relative one: a cue gene's mean sits near zero, where
        # relative precision is meaningless, and the reduction runs in float64 over a float32 block.
        assert list(sample.expressed_mean) == pytest.approx(expected, rel=1e-5, abs=1e-6)

    def test_a_population_shifted_upward_reports_a_higher_mean(self):
        """The assertion that the series would show selection: move the population and the number
        moves with it, in the expressed phenotype rather than in stored genes."""
        built = world()
        speed = built.genes.index_of("speed")
        before = built.loop.metrics.sample(tick=0).expressed_mean[speed]

        living = Selection.from_mask(built.store.alive & (built.store.age >= 0))
        genes = built.genetics.genes(living)
        genes[:, speed] += 1.0
        built.genetics.set_genes(living, genes)

        after = built.loop.metrics.sample(tick=1).expressed_mean[speed]

        # Founders draw `speed` from (1, 3) and it is read as a magnitude, so every stored value is
        # already positive and a uniform shift moves the expressed mean by exactly as much.
        assert after == pytest.approx(before + 1.0, rel=1e-4)

    def test_a_population_of_clones_has_no_spread(self):
        """The other end of the same reading: a spread that collapses is a population that has run
        out of the variation it would need to adapt again."""
        built = world()
        living = Selection.from_mask(built.store.alive & (built.store.age >= 0))
        genes = built.genetics.genes(living)
        built.genetics.set_genes(living, np.broadcast_to(genes[0], genes.shape).copy())

        sample = built.loop.metrics.sample(tick=0)

        assert list(sample.expressed_spread) == pytest.approx([0.0] * len(sample.expressed_spread))

    def test_a_gene_series_is_named_rather_than_indexed(self):
        """A client naming a column by position would break the first time the vocabulary is
        widened, which §2.3 makes an additive and expected event."""
        built = world()
        built.loop.advance(3)

        assert len(built.loop.metrics.gene_series("speed")) == 3
        assert built.loop.metrics.samples[0].gene_names == built.genes.vocabulary.names

    def test_an_unknown_gene_or_metric_is_refused_rather_than_answered_emptily(self):
        """An empty list reads as "nothing happened", which is the wrong answer to a typo (§8.7)."""
        built = world()
        built.loop.advance(1)

        with pytest.raises(KeyError, match="wingspan"):
            built.loop.metrics.gene_series("wingspan")
        with pytest.raises(KeyError, match="populaton"):
            built.loop.metrics.series("populaton")


class TestRecordingChangesNothing:
    def test_a_world_with_a_recorder_runs_the_same_as_one_without(self):
        """Recording is an observation, which is why it is not in `TICK_ORDER` (§2.1). If attaching
        one moved an outcome it would be a rule change under §2.8, and a metric that perturbs what
        it measures is not a metric."""
        watched = build_demo_world(seed=7, n_entities=60)
        watched.loop.metrics = MetricHistory(
            watched.store, watched.genetics, watched.plants, watched.genes.vocabulary, EVERY_TICK
        )
        unwatched = build_demo_world(seed=7, n_entities=60)
        unwatched.loop.metrics = None

        watched.loop.advance(40)
        unwatched.loop.advance(40)

        assert np.array_equal(watched.store.x, unwatched.store.x)
        assert np.array_equal(watched.store.energy, unwatched.store.energy)
        assert np.array_equal(watched.store.genes, unwatched.store.genes)
