"""A served world's clock as a series of rate segments: pause and fast-forward (#307, §2.4).

**The anchor stays fixed, and every rate change after it is another fixed point.** A change is the
moment it was made, the tick due at that moment, and the rate from then on (0 for a pause). The
tick due at any real time is read off the segment in force then, from that segment's own start:
never summed from the gaps between wakes. So the property #32 pinned survives time controls
unchanged: however a schedule batches its wakes, a world reaches the same tick at the same moment.

**A change lands on a tick boundary** (§2.7). Its tick is the last whole tick due under the old
rate when it was made, and the fraction of a tick in progress is dropped, at most one tick per
change and never per wake. Starting the new segment at that tick's own moment instead would keep
the fraction, but it would apply the new rate retroactively: a fast-forward would owe ticks for the
instant before it was asked for.

**A change is made against the tick due, not the tick the world has reached.** A world still paying
a backlog when the player pauses goes on paying it, because those ticks were owed before the pause.
Measuring from the world's own tick would forgive the backlog, and then how much a pause forgave
would depend on when the last wake happened, which is the batching dependence §2.4 forbids.

**The series is the world's history of time.** With it, the anchor and nothing else, `due` answers
for any moment the world has lived through, so its age can always be reconstructed.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

from scheduler.catchup import Anchor


@dataclass(frozen=True)
class RateChange:
    """The player changed how fast a served world runs.

    wall_seconds: real time of the change, on the anchor's clock.
    tick: the tick due at that moment under the rate before it, so the boundary it lands on.
    ticks_per_second: the rate from then on. Zero is a pause. Never negative, since a world never
        runs backwards, and finite, since a rate is a number of ticks to compute.
    """

    wall_seconds: float
    tick: int
    ticks_per_second: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.ticks_per_second) and self.ticks_per_second >= 0):
            raise ValueError(
                f"a rate must be zero (paused) or a positive number of ticks per second, got "
                f"{self.ticks_per_second}"
            )


@dataclass(frozen=True)
class Timeline:
    """A world's anchor and every rate change since, in the order they were made.

    anchor: the founding moment, whose rate is the world's own (§2.1's live rate). Leaving always
        returns to it (#292, decision 8's follow-up).
    changes: each one at or after the one before, in both wall time and tick.
    """

    anchor: Anchor
    changes: tuple[RateChange, ...] = ()

    def __post_init__(self) -> None:
        previous: Anchor | RateChange = self.anchor
        for change in self.changes:
            if change.wall_seconds < previous.wall_seconds or change.tick < previous.tick:
                raise ValueError(
                    f"rate changes must follow one another in time and in ticks: {change} comes "
                    f"after {previous}"
                )
            previous = change

    @property
    def own_ticks_per_second(self) -> float:
        return self.anchor.ticks_per_second

    @property
    def ticks_per_second(self) -> float:
        """The rate now: the last change's, or the world's own if it has never been changed."""
        return self.changes[-1].ticks_per_second if self.changes else self.own_ticks_per_second

    def due(self, now: float) -> int:
        """The tick the world should have reached at real time `now`.

        From the segment in force at `now`, so a past moment reads the tick it was then. Before the
        anchor, the anchor's own tick, as `Anchor.due` already answers for a clock stepped back.
        """
        walls = [change.wall_seconds for change in self.changes]
        index = bisect.bisect_right(walls, now) - 1
        if index < 0:
            return self.anchor.due(now)
        change = self.changes[index]
        return change.tick + math.floor((now - change.wall_seconds) * change.ticks_per_second)

    def at_rate(self, now: float, ticks_per_second: float) -> Timeline:
        """This timeline with the rate changed to `ticks_per_second` at real time `now`.

        Unchanged if the rate already is that, so a repeated request records nothing. A `now`
        earlier than the last change (a clock stepped back) lands at the last change's moment, so
        the series stays in order and the change still owes nothing for the lost time.
        """
        if ticks_per_second == self.ticks_per_second:
            return self
        last = self.changes[-1].wall_seconds if self.changes else self.anchor.wall_seconds
        moment = max(now, last)
        change = RateChange(wall_seconds=moment, tick=self.due(moment), ticks_per_second=ticks_per_second)
        return Timeline(self.anchor, self.changes + (change,))

    def at_own_rate(self, now: float) -> Timeline:
        """Back to the world's own rate at `now`: what leaving does (#307)."""
        return self.at_rate(now, self.own_ticks_per_second)
