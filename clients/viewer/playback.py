"""Playback state machine: pause, single-step, and adjustable speed drive the tick loop.

Owns no simulation state and no rendering; it only decides, given how much wall-clock time has
passed, how many ticks the caller should ask the core `TickLoop` to advance this frame, and what
interpolation `alpha` (CLAUDE.md §2.1) the renderer should blend at. Kept free of pygame so it is
testable without a display.
"""

from __future__ import annotations


class Playback:
    """Converts elapsed wall-clock time into ticks owed, independent of any tick-rate the
    simulation itself would run at offline (CLAUDE.md §2.4: the schedule controls when compute
    happens, not how fast the world moves) — here it controls how fast a *live* view plays back.

    ticks_per_second: float, > 0. Baseline sim-ticks advanced per real second at speed 1.0.
    speed: float, > 0. Multiplier on ticks_per_second; adjustable at runtime.
    paused: bool. While True, advance() owes zero ticks regardless of elapsed time — time spent
        paused is discarded, not queued, so resuming never produces a burst of catch-up ticks.
    """

    def __init__(self, ticks_per_second: float, speed: float = 1.0) -> None:
        if ticks_per_second <= 0:
            raise ValueError("ticks_per_second must be positive")
        if speed <= 0:
            raise ValueError("speed must be positive")

        self.ticks_per_second = ticks_per_second
        self.speed = speed
        self.paused = False
        self._tick_debt = 0.0
        self._pending_steps = 0

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def set_speed(self, speed: float) -> None:
        if speed <= 0:
            raise ValueError("speed must be positive")
        self.speed = speed

    def request_step(self) -> None:
        """Queue exactly one tick and pause, for frame-by-frame inspection.

        Pausing here (rather than requiring the caller to already be paused) means a single
        control does the obvious thing whether playback was running or already stopped.
        """
        self._pending_steps += 1
        self.paused = True

    def advance(self, elapsed_seconds: float) -> tuple[int, float]:
        """How many ticks are owed this frame, and the interpolation alpha to render at.

        A queued step always wins: it returns exactly the queued tick count and alpha=1.0 (render
        the freshly-stepped state exactly, no blending). Otherwise, while paused, nothing is owed
        and alpha=1.0 holds the view at the last simulated state. While running, elapsed time
        accumulates into a fractional tick debt; whole ticks owed are returned and subtracted from
        the debt, and the leftover fraction becomes alpha for the renderer to blend between the
        tick loop's previous and current position snapshots.
        """
        if elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be non-negative")

        if self._pending_steps > 0:
            n_ticks = self._pending_steps
            self._pending_steps = 0
            self._tick_debt = 0.0
            return n_ticks, 1.0

        if self.paused:
            return 0, 1.0

        self._tick_debt += elapsed_seconds * self.ticks_per_second * self.speed
        n_ticks = int(self._tick_debt)
        self._tick_debt -= n_ticks
        return n_ticks, self._tick_debt
