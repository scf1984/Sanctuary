"""When a world wakes to catch up: a per-world schedule that decays over an absence (#32, §2.4).

**The schedule decides when compute happens, never how fast the world moves.** It returns how long
to wait before the next wake, and nothing else. How many ticks a wake owes comes from the real
time that has passed (`scheduler.catchup.Anchor`), so waking hourly and then daily batches the same
ticks differently and cannot make the world age more slowly.

**Watching is the same dial turned the other way** (gate 2 on #35, decided by the owner). There is
no live loop. A watched world is caught up on a short interval, and an unwatched one on a long,
decaying one. Both drive the same executor, so there is one execution path. The
minimum interval is the one number the decision left to measure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Stage:
    """Wake every `interval` seconds until the player has been away for `until` seconds.

    until: seconds since a client last looked, exclusive. The last stage of a schedule must be
        `math.inf`, so that every absence, however long, has an interval.
    interval: seconds between wakes during this stage. Positive.
    """

    until: float
    interval: float


@dataclass(frozen=True)
class WakeSchedule:
    """How long a world waits between catch-ups.

    watched_interval: seconds between wakes while a client is attached. Short, because watching
        is the one time somebody is looking at what a batch produced.
    absent: the stages an absence decays through, in order. For example, hourly for a day and
        then daily: `(Stage(86_400, 3_600), Stage(math.inf, 86_400))`.

    Per world, never a constant (§2.4), and never part of a world's rules: two worlds that differ
    only in schedule are the same world caught up at different moments, which the batching tests
    in `tests/scheduler` hold to bit for bit.
    """

    watched_interval: float
    absent: tuple[Stage, ...]

    def __post_init__(self) -> None:
        if not self.watched_interval > 0:
            raise ValueError(f"watched_interval must be positive, got {self.watched_interval}")
        if not self.absent:
            raise ValueError("a schedule needs at least one absent stage")
        for stage in self.absent:
            if not stage.interval > 0:
                raise ValueError(f"a stage's interval must be positive, got {stage.interval}")
        untils = [stage.until for stage in self.absent]
        if any(later <= earlier for earlier, later in zip(untils, untils[1:])):
            raise ValueError(f"stages must end in increasing order, got {untils}")
        if untils[-1] != math.inf:
            raise ValueError(
                "the last stage must last forever (until=math.inf); otherwise an absence longer "
                f"than {untils[-1]} seconds has no wake at all"
            )

    def interval(self, *, watched: bool, away: float) -> float:
        """Seconds until the next wake, for a world watched or not, `away` seconds after a client
        last looked."""
        if watched:
            return self.watched_interval
        return next(stage.interval for stage in self.absent if away < stage.until)
