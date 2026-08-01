"""Fixed-timestep tick loop: advances systems in a declared order, no wall clock (CLAUDE.md §2.1).

The tick counter is the only clock in the simulation. ``advance(n)`` runs the registered systems
``n`` times, in the order they were registered, and has no notion of real time, frame rate, or
rendering — the caller decides how many ticks are owed, whether that is one tick from a live game
loop or a week's worth from offline catch-up (§2.4), and this loop treats both the same way.
"""

from __future__ import annotations

from typing import Callable, Optional, Protocol, Sequence

import numpy as np

from core.entities.growth import GrowthConfig, grow_if_crowded
from core.entities.store import EntityStore
from core.invariants import InvariantRegistry

System = Callable[[], None]


class MetricRecorder(Protocol):
    """Anything that can be offered a tick and decide for itself whether to record it (#30).

    A structural type rather than an import, because `core/` must stay importable standalone (§4)
    and `metrics/` reads `core/`. Inverting that for one call would make the simulation depend on
    its own instrumentation, which is the wrong way round: a world that cannot be observed is a
    smaller loss than a world that cannot run without an observer.
    """

    def record_if_due(self, tick: int) -> object:
        """Called once per tick, with the tick just completed. Sampling cadence is the recorder's."""
        ...


class TickLoop:
    """Advances a fixed, ordered list of systems by whole ticks.

    systems: called in this exact order, once per tick advanced. The order is fixed at
        construction time from the sequence the caller passes — never derived from import order
        or registration side effects — and exposed via this attribute so it is inspectable.
    tick_count: ticks advanced since construction. This is the only clock; nothing here reads
        wall-clock time, so ``advance(1000)`` and a thousand calls to ``advance(1)`` leave it at
        the same value.
    previous_positions, current_positions: ``(x, y, z)`` tuples of ``(n_entities,)`` float32
        arrays, world units, snapshotted from ``store`` at the tick boundary before and after the
        most recent ``advance()`` call. The renderer interpolates between these two snapshots
        (§2.1) so tick size is never constrained by visual smoothness. A snapshot is taken once
        per ``advance()`` call, not once per tick within it, so a large catch-up batch copies
        positions twice regardless of how many ticks it covers.
    previous_row_ids, current_row_ids: ``(n_entities,)`` int64, the stable id in each row at those
        same two boundaries, -1 for a free row (`EntityStore.row_ids`). Taken together with the
        positions rather than left to the reader to fetch, because a position snapshot is not
        interpretable without the occupancy that qualifies it: a row's coordinates survive
        `release` untouched, so a position alone cannot say whether it belongs to a live entity, to
        one that died during the interval, or to a newborn that inherited the row. Reading
        ``store.alive`` at draw time answers only the first of those, and only for the *current*
        end of the interval (#119).
    metrics: an optional recorder (`metrics.MetricHistory`, #30), offered every tick and sampling
        on its own cadence. Optional because the entity invariants and the loop itself are testable
        against a bare store, and a recorder needs a genetics stack and a plant field to read.
        **Public and settable after construction**, because those do not exist until `build_world`
        has finished — attaching it afterwards is what keeps the dependency pointing one way (§4).

        Called here rather than placed in `TICK_ORDER`, and that is a rule rather than a
        convenience: the order is what a tick *does*, and it is frozen into the MAJOR version
        (§2.1, §2.8). Recording an observation changes no outcome, so putting it in the tuple would
        make the sampling cadence part of the rule set and freeze it for the life of a world.
        Capacity growth sits outside the tuple for the same reason (#127).

        It is offered **every** tick with the tick number, never every `advance()` call, so a world
        advanced in one batch of a hundred records exactly what one advanced in a hundred batches
        of one records — §2.4 forbids batching from changing outcomes, and a history that depended
        on how a client chose to call `advance` would be precisely that.
    growth: an optional capacity policy (`core.entities.growth`, #127), evaluated after every
        tick and **only between ticks**. Growth belongs here rather than in `TICK_ORDER` because a
        system runs mid-tick with others still to follow it, and `EntityStore.grow` replaces every
        column array with a new object — a system holding a NumPy view into the old one would keep
        reading pre-growth values and produce silently wrong results (§2.3). At a boundary no view
        and no `Selection` is live, which is the whole reason the boundary is where this happens.
        None disables it, which is what a world that never allocates during a tick wants.
    invariants, debug_checks: an optional InvariantRegistry (CLAUDE.md §6) evaluated after every
        tick, only when ``debug_checks`` is True. It is handed ``store``; invariants over anything
        else the world holds — the plant field, say — close over it when they are built, so the
        loop needs no knowledge of which domains are being checked. Disabled is the default
        and costs nothing beyond the one boolean check per tick — no registry lookup, no
        predicate call — so production runs pay nothing for a check meant for debug builds.
    """

    def __init__(
        self,
        store: EntityStore,
        systems: Sequence[System],
        invariants: Optional[InvariantRegistry] = None,
        debug_checks: bool = False,
        growth: Optional[GrowthConfig] = None,
        metrics: Optional[MetricRecorder] = None,
    ) -> None:
        if debug_checks and invariants is None:
            raise ValueError("debug_checks requires an invariants registry")

        self._growth = growth
        self.metrics = metrics

        self.systems = tuple(systems)
        self._store = store
        self._invariants = invariants if invariants is not None else InvariantRegistry()
        self.debug_checks = debug_checks
        self.tick_count = 0
        self.current_positions, self.current_row_ids = self._snapshot()
        self.previous_positions, self.previous_row_ids = (
            self.current_positions,
            self.current_row_ids,
        )

    def advance(self, n_ticks: int) -> None:
        """Run every system in order, ``n_ticks`` times, advancing the tick counter by n_ticks.

        Raises ValueError for negative ``n_ticks``. When ``debug_checks`` is enabled, raises
        InvariantViolation (leaving ``tick_count`` at the failing tick) as soon as a registered
        invariant reports a violation, before running any further ticks.
        """
        if n_ticks < 0:
            raise ValueError("n_ticks must be non-negative")

        self.previous_positions, self.previous_row_ids = (
            self.current_positions,
            self.current_row_ids,
        )
        for _ in range(n_ticks):
            for system in self.systems:
                system()
            self.tick_count += 1
            if self.debug_checks:
                self._invariants.check_all(self._store, self.tick_count)
            # After the invariants rather than before, so a check never sees a store whose columns
            # were replaced halfway through the tick it is reporting on.
            if self.metrics is not None:
                # Before growth rather than after: a sample taken across a reallocation would read
                # some columns from the old arrays and some from the new (§2.3), and this is the
                # one caller that holds no view of its own to be invalidated.
                self.metrics.record_if_due(self.tick_count)
            if self._growth is not None and grow_if_crowded(self._store, self._growth):
                self._extend_previous_snapshot()
        self.current_positions, self.current_row_ids = self._snapshot()

    def _extend_previous_snapshot(self) -> None:
        """Widen the opening snapshot to the store's new capacity after a growth.

        The renderer compares the two snapshots elementwise (#119), so they have to be the same
        length or the comparison indexes off the end of the shorter one. New rows are padded with
        id **-1**, which is not a placeholder but the truthful answer: nobody occupied that row at
        the opening boundary. `live_positions` already reads -1 as "not the same entity as before"
        and draws such a row where it is now rather than blending it in from a stale coordinate,
        which is exactly right for a row that did not exist.
        """
        capacity = self._store.capacity
        old_ids = self.previous_row_ids
        row_ids = np.full(capacity, -1, dtype=np.int64)
        row_ids[: old_ids.shape[0]] = old_ids
        def widened(axis: np.ndarray) -> np.ndarray:
            return np.concatenate(
                [axis, np.zeros(capacity - axis.shape[0], dtype=axis.dtype)]
            )

        x, y, z = self.previous_positions
        self.previous_positions = (widened(x), widened(y), widened(z))
        self.previous_row_ids = row_ids

    def _snapshot(self) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        """Positions and the row occupancy that qualifies them, as of right now.

        Returned together, and assigned together by both callers, so the two halves are always
        from the same instant. Split into two calls they could drift by a system, which would be a
        silent mismatch rather than an error.
        """
        return (
            (self._store.x.copy(), self._store.y.copy(), self._store.z.copy()),
            self._store.row_ids(),
        )
