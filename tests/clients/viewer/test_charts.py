"""Where a point goes, and what the label says (#39).

Geometry lives in `charts.py` rather than `app.py` precisely so it can be asserted: this module is
collected in CI and the pygame wiring is not (#110). Every test here runs without a display.
"""

import numpy as np
import pytest

from clients.viewer.charts import Chart, draw_chart, plot_points, stack_charts, world_charts
from clients.viewer.demo_world import build_demo_world


class TestChart:
    def test_values_and_ticks_must_be_the_same_length(self):
        """A series plotted against the wrong time axis still looks like a chart, which is why the
        mismatch is refused rather than broadcast away (§8.7)."""
        with pytest.raises(ValueError, match="living"):
            Chart("living", ticks=[0, 10, 20], values=[1.0, 2.0])


class TestPlotPoints:
    def test_the_largest_value_sits_at_the_top_row(self):
        """Rows count downward in an image, and applying that convention once here is what stops
        every caller getting it right or wrong independently."""
        points = plot_points([1.0, 5.0, 3.0], width=3, height=10)

        assert points[1, 1] == 0
        assert points[0, 1] == 9

    def test_points_span_the_full_width(self):
        points = plot_points([1.0, 2.0, 3.0, 4.0], width=8, height=4)

        assert points[0, 0] == 0
        assert points[-1, 0] == 7

    def test_a_flat_series_is_centred_rather_than_pinned_to_an_edge(self):
        """The case worth stating: a population holding steady and one that has flatlined at zero
        are both flat, and putting them at different heights would read meaning into a constant
        that it does not carry. The label says the value; the line says the shape."""
        steady = plot_points([240.0] * 5, width=5, height=11)
        dead = plot_points([0.0] * 5, width=5, height=11)

        assert list(steady[:, 1]) == [5] * 5
        assert list(dead[:, 1]) == list(steady[:, 1])

    def test_an_empty_series_plots_nothing(self):
        """A world whose history has not reached its first sample is the ordinary opening state,
        not an error."""
        assert plot_points([], width=10, height=4).shape == (0, 2)

    def test_a_single_sample_plots_at_the_left_edge(self):
        # An odd height so the middle row is unambiguous: with four rows the centre falls between
        # two and the rounding, not the rule, would be what the assertion measured.
        points = plot_points([7.0], width=10, height=5)

        assert points.tolist() == [[0, 2]]

    @pytest.mark.parametrize("size", [(1, 4), (10, 0)])
    def test_a_panel_too_small_to_draw_in_is_refused(self, size):
        with pytest.raises(ValueError, match="at least 2x1"):
            plot_points([1.0, 2.0], *size)


class TestDrawChart:
    def test_the_panel_has_the_size_it_was_asked_for(self):
        panel = draw_chart([1.0, 2.0, 3.0], width=20, height=6, color=(1, 2, 3))

        assert panel.shape == (6, 20, 3)
        assert panel.dtype == np.uint8

    def test_every_column_is_inked_so_the_line_has_no_gaps(self):
        """A plain scatter leaves holes the moment the series is longer than the panel is wide, and
        a chart with holes reads as missing data rather than as a rendering choice."""
        panel = draw_chart(list(np.linspace(0.0, 1.0, 200)), width=40, height=8, color=(9, 9, 9))

        inked = (panel == np.array([9, 9, 9], dtype=np.uint8)).all(axis=2)
        assert inked.any(axis=0).all(), "some column carries no line"

    def test_a_rising_series_is_drawn_rising(self):
        """The one assertion that would catch an inverted axis, which is otherwise invisible: a
        chart drawn upside down looks entirely plausible."""
        panel = draw_chart([0.0, 1.0], width=2, height=9, color=(9, 9, 9))
        inked = (panel == np.array([9, 9, 9], dtype=np.uint8)).all(axis=2)

        assert np.flatnonzero(inked[:, 1]).min() < np.flatnonzero(inked[:, 0]).max()


class TestStackCharts:
    def test_charts_stack_with_a_rule_between_them(self):
        charts = [Chart("a", [0, 1], [1.0, 2.0]), Chart("b", [0, 1], [3.0, 4.0])]

        image, labels = stack_charts(charts, width=30, chart_height=5, gap=4)

        # two charts, each with a one-pixel axis under it, and one gap between them
        assert image.shape == (5 + 1 + 4 + 5 + 1, 30, 3)
        assert [label for label, _, _ in labels] == ["a", "b"]

    def test_each_chart_gets_its_own_colour(self):
        charts = [Chart(str(i), [0], [1.0]) for i in range(3)]

        _, labels = stack_charts(charts, width=10, chart_height=3)

        assert len({color for _, _, color in labels}) == 3

    def test_the_label_carries_the_current_value_and_the_range(self):
        """The line cannot: it is normalised to its own extremes, so two charts of wildly different
        magnitude look identical without the numbers beside them."""
        image, labels = stack_charts(
            [Chart("living", [0, 1, 2], [10.0, 90.0, 50.0])], width=10, chart_height=3
        )

        _, detail, _ = labels[0]
        assert "now 50" in detail
        assert "10" in detail and "90" in detail
        assert image.shape[1] == 10

    def test_a_history_with_no_samples_yet_says_so_rather_than_reading_zero(self):
        _, labels = stack_charts([Chart("living", [], [])], width=10, chart_height=3)

        assert labels[0][1] == "no samples yet"

    def test_nothing_to_draw_produces_an_empty_image_rather_than_raising(self):
        image, labels = stack_charts([], width=10, chart_height=3)

        assert image.shape == (0, 10, 3)
        assert labels == []


class TestWorldCharts:
    """The panel's contents, against a real recorded history — the one place this module meets
    `metrics` and the only thing it needs from it is three read methods."""

    def world(self, ticks=60):
        built = build_demo_world(seed=1, n_entities=60)
        built.loop.advance(ticks)
        return built

    def test_population_and_condition_are_always_shown(self):
        """A trait series is unreadable without them: a mean speed climbing while the population
        halves is a bottleneck, and the same climb while it holds is selection."""
        charts = world_charts(self.world().loop.metrics, gene=None)

        assert [chart.label for chart in charts] == ["living", "median energy"]

    def test_naming_a_gene_adds_its_mean_and_its_spread(self):
        charts = world_charts(self.world().loop.metrics, gene="speed")

        assert [chart.label for chart in charts][2:] == ["speed mean", "speed spread"]

    def test_every_chart_is_plotted_against_the_recorded_ticks(self):
        history = self.world().loop.metrics

        charts = world_charts(history, gene="speed")

        for chart in charts:
            assert list(chart.ticks) == history.series("tick")
            assert len(chart.values) == len(chart.ticks)

    def test_an_unknown_gene_is_refused_by_the_history_rather_than_drawn_empty(self):
        with pytest.raises(KeyError, match="wingspan"):
            world_charts(self.world().loop.metrics, gene="wingspan")
