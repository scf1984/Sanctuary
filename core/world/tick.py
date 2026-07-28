"""Fixed-timestep tick loop: advances systems in a declared order, no wall clock (CLAUDE.md §2.1).

The tick counter is the only clock in the simulation. ``advance(n)`` runs the registered systems
``n`` times, in the order they were registered, and has no notion of real time, frame rate, or
rendering — the caller decides how many ticks are owed, whether that is one tick from a live game
loop or a week's worth from offline catch-up (§2.4), and this loop treats both the same way.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from core.entities.store import EntityStore
from core.invariants import InvariantRegistry

System = Callable[[], None]


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
    ) -> None:
        if debug_checks and invariants is None:
            raise ValueError("debug_checks requires an invariants registry")

        self.systems = tuple(systems)
        self._store = store
        self._invariants = invariants if invariants is not None else InvariantRegistry()
        self.debug_checks = debug_checks
        self.tick_count = 0
        self.current_positions = self._snapshot_positions()
        self.previous_positions = self.current_positions

    def advance(self, n_ticks: int) -> None:
        """Run every system in order, ``n_ticks`` times, advancing the tick counter by n_ticks.

        Raises ValueError for negative ``n_ticks``. When ``debug_checks`` is enabled, raises
        InvariantViolation (leaving ``tick_count`` at the failing tick) as soon as a registered
        invariant reports a violation, before running any further ticks.
        """
        if n_ticks < 0:
            raise ValueError("n_ticks must be non-negative")

        self.previous_positions = self.current_positions
        for _ in range(n_ticks):
            for system in self.systems:
                system()
            self.tick_count += 1
            if self.debug_checks:
                self._invariants.check_all(self._store, self.tick_count)
        self.current_positions = self._snapshot_positions()

    def _snapshot_positions(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (self._store.x.copy(), self._store.y.copy(), self._store.z.copy())
