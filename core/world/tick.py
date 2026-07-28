"""Fixed-timestep tick loop: advances systems in a declared order, no wall clock (CLAUDE.md §2.1).

The tick counter is the only clock in the simulation. ``advance(n)`` runs the registered systems
``n`` times, in the order they were registered, and has no notion of real time, frame rate, or
rendering — the caller decides how many ticks are owed, whether that is one tick from a live game
loop or a week's worth from offline catch-up (§2.4), and this loop treats both the same way.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from core.entities.store import EntityStore

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
    """

    def __init__(self, store: EntityStore, systems: Sequence[System]) -> None:
        self.systems = tuple(systems)
        self._store = store
        self.tick_count = 0
        self.current_positions = self._snapshot_positions()
        self.previous_positions = self.current_positions

    def advance(self, n_ticks: int) -> None:
        """Run every system in order, ``n_ticks`` times, advancing the tick counter by n_ticks.

        Raises ValueError for negative ``n_ticks``.
        """
        if n_ticks < 0:
            raise ValueError("n_ticks must be non-negative")

        self.previous_positions = self.current_positions
        for _ in range(n_ticks):
            for system in self.systems:
                system()
            self.tick_count += 1
        self.current_positions = self._snapshot_positions()

    def _snapshot_positions(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (self._store.x.copy(), self._store.y.copy(), self._store.z.copy())
