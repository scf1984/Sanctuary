"""Metrics: what a world looks like from outside it (CLAUDE.md §2.7, §4, issue #30).

This package is the *secondary surface* — what a dashboard, a phone widget, an offline warning
(#34) and a competition (#42) all read. Defining each quantity once, here, is what stops three
consumers disagreeing about what "diversity" means.

**Everything crossing this boundary is a plain Python value.** That is not a stylistic preference:
§3.1 puts the simulation on a shared machine with clients asking for view information, so a metric
that hands back a NumPy view is one a client will hold, and the fix at that point is a rewrite
rather than a wrapper. `Sample` and `MetricHistory.series` are floats, ints and lists.

**Statistics are computed here, never in a client.** A histogram reduced in `clients/` cannot cross
a socket, and it puts a pass over the whole gene matrix on the wrong side of the boundary #35
exists to draw. This module returns the summary; the client draws it.

**A series is recorded, not recomputed.** A client asking how mean speed moved over ten thousand
ticks must not trigger a replay — the simulation is non-deterministic (§2.2), so a replay would
answer a different question. The history is filled as the world runs, which is also the only way it
survives §2.4's offline catch-up: a metric that exists only while somebody is watching has a hole in
it for every absence.

What is **deliberately absent** (§8.2), and reserved on #30 rather than left to whoever reaches for
it first: species richness, and Shannon entropy over species abundance. Both need more than one
species to say anything, so they land with #16 — with one species they read 1 and 0 forever, and a
metric that cannot move is worse than a missing one because it looks like an answer.
"""

from metrics.history import MetricHistory, MetricsConfig, Sample

__all__ = ["MetricHistory", "MetricsConfig", "Sample"]
