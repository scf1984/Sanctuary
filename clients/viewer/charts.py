"""Recorded metric series, drawn as small charts over the world view (#39).

The viewer answers "why did this population crash" (§3.3), and a position snapshot cannot: the
crash is a shape in *time*. `metrics.MetricHistory` records that shape (#30) and this module turns
it into pixels.

**Nothing here reads `core/` or reduces anything.** A chart is handed a list of numbers that
already crossed the metric boundary, which is what keeps this drawable against a series that
arrived over a socket rather than out of the process (§3.1, §4). If a client ever has to compute
what it draws, the reduction is on the wrong side.

**Geometry lives here rather than in `app.py`**, for the reason #110 gives: this module is
collectable in CI and `app.py` is not, so anything that decides *where a point goes* has to be
testable. `app.py` blits the surface and binds the keys.

Drawn with NumPy into an RGB array rather than with a plotting library, which is the same decision
§3.3 records for the terrain: a library that redraws a whole figure per frame fights an
immediate-mode loop instead of supporting it, and a chart of at most a few thousand points is a
handful of array writes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

_BACKGROUND = np.array([14, 16, 20], dtype=np.uint8)
_AXIS = np.array([70, 76, 88], dtype=np.uint8)
# One hue per chart, assigned by the caller's order rather than by name: the panel stacks whatever
# it is given, and a colour keyed to a metric name would need a table to be kept in step with #30.
SERIES_COLORS: tuple[tuple[int, int, int], ...] = (
    (126, 200, 227),
    (232, 176, 96),
    (150, 214, 140),
    (222, 138, 168),
)


@dataclass(frozen=True)
class Chart:
    """One series to draw: its label, its values oldest-first, and the ticks they were taken at.

    values and ticks are plain lists — what `MetricHistory.series` and `gene_series` return, and
    what would arrive over a socket. Equal length is the caller's contract and is checked, because
    a mismatch would silently plot a series against the wrong time axis and look plausible (§8.7).
    """

    label: str
    ticks: Sequence[int]
    values: Sequence[float]

    def __post_init__(self) -> None:
        if len(self.ticks) != len(self.values):
            raise ValueError(
                f"chart '{self.label}' has {len(self.values)} values against {len(self.ticks)} "
                "ticks; a series plotted against the wrong time axis still looks like a chart"
            )


def plot_points(values: Sequence[float], width: int, height: int) -> np.ndarray:
    """(n, 2) int: where each value lands in a `width` x `height` panel, as (column, row).

    Rows count downward, so the **largest** value is at row 0 — the pixel convention, applied here
    once rather than at every caller.

    A flat series is centred rather than pinned to an edge. That is the case worth stating: a
    population holding steady and a population that has flatlined at zero are both flat, and a
    chart that put them in different places would be reading meaning into a constant it does not
    have. The axis labels carry the value; the line carries the shape.
    """
    if width < 2 or height < 1:
        raise ValueError(f"a chart panel must be at least 2x1, got {width}x{height}")
    series = np.asarray(values, dtype=np.float64)
    if series.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    columns = (
        np.zeros(1, dtype=np.int64)
        if series.size == 1
        else np.round(np.linspace(0, width - 1, series.size)).astype(np.int64)
    )
    low, high = float(series.min()), float(series.max())
    span = high - low
    # A constant series has no span to scale against, so it sits on the middle row.
    fraction = np.full(series.size, 0.5) if span == 0.0 else (series - low) / span
    rows = np.round((1.0 - fraction) * (height - 1)).astype(np.int64)
    return np.stack([columns, rows], axis=1)


def draw_chart(values: Sequence[float], width: int, height: int, color) -> np.ndarray:
    """(height, width, 3) uint8: one series as a line on a dark panel.

    Consecutive points are joined by filling the column between their rows, which is a vertical
    span rather than a true line. At these sizes the two are indistinguishable and the span is one
    slice per point instead of a Bresenham walk — and it never leaves a gap, which a plain scatter
    does the moment the series is longer than the panel is wide.
    """
    panel = np.broadcast_to(_BACKGROUND, (height, width, 3)).copy()
    points = plot_points(values, width, height)
    if not points.size:
        return panel
    ink = np.array(color, dtype=np.uint8)
    previous_row = points[0, 1]
    for column, row in points:
        low, high = sorted((int(previous_row), int(row)))
        panel[low : high + 1, int(column)] = ink
        previous_row = row
    return panel


def stack_charts(charts: Sequence[Chart], width: int, chart_height: int, gap: int = 6):
    """(image, labels): the charts stacked vertically, and what to write beside each.

    Returns the labels rather than rendering them, because text is a font and a font is `pygame` —
    keeping it out is what lets this module be tested without a display (#110). Each label carries
    the series' current value and its range, since the line alone says only the shape.
    """
    if not charts:
        return np.zeros((0, width, 3), dtype=np.uint8), []

    rows = []
    labels = []
    for index, chart in enumerate(charts):
        color = SERIES_COLORS[index % len(SERIES_COLORS)]
        rows.append(draw_chart(chart.values, width, chart_height, color))
        rows.append(np.broadcast_to(_AXIS, (1, width, 3)).copy())
        if index + 1 < len(charts):
            rows.append(np.broadcast_to(_BACKGROUND, (gap, width, 3)).copy())
        labels.append((chart.label, _describe(chart), color))
    return np.concatenate(rows, axis=0), labels


def _describe(chart: Chart) -> str:
    """"now 2431  (1204-2588)" — the numbers the line cannot carry.

    Empty when there is nothing recorded yet, rather than "0": a world whose history has not
    reached its first sample has no value, and printing a zero would be an answer to a question
    nobody asked (§8.7).
    """
    if not len(chart.values):
        return "no samples yet"
    series = np.asarray(chart.values, dtype=np.float64)
    return f"now {series[-1]:.3g}   ({series.min():.3g} - {series.max():.3g})"


def world_charts(history, gene: Optional[str]) -> list[Chart]:
    """The panel's contents: population, condition, and one gene's mean and spread.

    history: a `metrics.MetricHistory`. Typed loosely on purpose — this module reads only
        `series`, `gene_series` and `samples`, which is the whole of what a socket would carry
        (§3.1), and naming the class would tie a client to a Python object it will not always have.
    gene: which gene to plot, or `None` to show population and condition alone. Chosen by the
        viewer rather than fixed here, because *which* trait is under selection is the question
        being asked and it changes run to run.

    Population and energy are always shown because a trait series is unreadable without them: a
    mean speed that climbs while the population halves is a bottleneck, and the same climb while
    the population holds is selection.
    """
    ticks = history.series("tick")
    charts = [
        Chart("living", ticks, history.series("living")),
        Chart("median energy", ticks, history.series("median_energy")),
    ]
    if gene is not None:
        charts.append(Chart(f"{gene} mean", ticks, history.gene_series(gene)))
        charts.append(
            Chart(f"{gene} spread", ticks, history.gene_series(gene, "expressed_spread"))
        )
    return charts
