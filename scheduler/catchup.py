"""Catch-up: advancing a world by the ticks real time says it owes (#32, §2.4).

**Catch-up is not a second simulation.** It calls the same `advance` a watched world calls, with
nothing drawn, so it cannot diverge from live behaviour. That holds bit for bit and not merely
statistically: `TickLoop.advance(n)` runs every tick through the identical system sequence however
the ticks are batched, and `tests/scheduler` pins one call, single ticks and uneven chunks to the
same final state.

**Ticks owed come from a fixed anchor, never from the gap since the last wake.** The anchor is one
moment at which both the real clock and the world's tick were known. Every wake computes the tick
the world *should* be at from that anchor and runs the difference. Summing per-wake gaps instead
would round at every wake, and a schedule that woke more often would lose more fractions of a tick,
which is exactly the "waking more often makes the world age differently" that §2.4 forbids.

**No clock is read here.** The caller passes `now`, in seconds. That keeps the arithmetic testable
without waiting, and keeps the one place that reads a wall clock in the process that serves the
world (#36). `core/` never does (§2.1).

**Preemptive capacity growth (§2.3's third mitigation) is not built**, and that is a measurement
rather than an omission (§8.5). Doubling a 100,000-row store takes 16 ms, against roughly 3 s for one
tick at that size (#258), and growth happens about log2(n) times over a whole run. A 1,500-tick
catch-up of the demo world grew once and spent 0.0 ms doing it. There is no pause to remove, so it
is held in reserve beside chunked paging.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Protocol

if TYPE_CHECKING:
    # For annotations only. `api` carries `Progress` to its clients, so importing `api` here at
    # runtime would be a cycle: api -> worlds -> scheduler -> api.
    from api.values import Status


class Advanceable(Protocol):
    """The two calls catch-up makes. `api.Session` is one. A test double is another."""

    def advance(self, ticks: int) -> Status: ...

    def status(self) -> Status: ...


@dataclass(frozen=True)
class Anchor:
    """A moment at which both the real clock and the world's tick were known.

    wall_seconds: real time at the anchor, in seconds on whatever clock the caller reads.
    tick: the world's tick count at that moment.
    ticks_per_second: the sim-rate. §2.1's baseline is 1, and it is a ratio-table entry, so it is
        tuned together with the rest of that table and never on its own.

    The anchor outlives any process. A service that stops and restarts keeps it beside the
    snapshot, so on restart the world owes the ticks of its downtime too: the world kept running
    while nobody was serving it, as §2.4 says it does while nobody is watching.
    """

    wall_seconds: float
    tick: int
    ticks_per_second: float

    def __post_init__(self) -> None:
        if not self.ticks_per_second > 0:
            raise ValueError(f"ticks_per_second must be positive, got {self.ticks_per_second}")

    def due(self, now: float) -> int:
        """The tick the world should have reached at real time `now`.

        Never before the anchor's own tick. A clock stepped backwards (an NTP correction, a changed
        timezone on a naive clock) owes nothing rather than a negative count, and a world never runs
        backwards.
        """
        elapsed = max(0.0, now - self.wall_seconds)
        return self.tick + math.floor(elapsed * self.ticks_per_second)


@dataclass(frozen=True)
class Progress:
    """How far one catch-up has got: `done` of `owed` ticks, and where the world now stands.

    `api` carries it to clients, because a served world announces every chunk on its event stream
    (#36). A long absence can take minutes to pay off, and a client showing that wants to say how
    far it has got.
    """

    done: int
    owed: int
    status: Status


class CatchUp:
    """Runs what a world owes, in chunks, reporting after each.

    chunk: ticks per `advance` call between progress reports. Batching never changes outcomes
        (see the module docstring), so this only decides how often a caller hears back, which
        matters because a long absence can take minutes to pay off.
    """

    def __init__(self, world: Advanceable, anchor: Anchor, chunk: int) -> None:
        if chunk < 1:
            raise ValueError(f"chunk must be at least one tick, got {chunk}")
        self.world = world
        self.anchor = anchor
        self.chunk = chunk

    def owed(self, now: float) -> int:
        """Ticks the world is behind at `now`. Zero once a run has collapsed (§2.7): an empty world
        can never refill, so its ticks are owed to nobody, and counting them would leave a
        caller trying to pay a debt that grows forever."""
        status = self.world.status()
        if status.collapsed:
            return 0
        return max(0, self.anchor.due(now) - status.tick)

    def run(self, now: float) -> Iterator[Progress]:
        """Pay off what is owed at `now`, yielding after every chunk.

        The total is fixed when the run starts. Ticks that come due while it is running are the
        next wake's, so a catch-up that takes longer than its own interval still ends rather than
        chasing a target that keeps moving.
        """
        owed = self.owed(now)
        done = 0
        while done < owed:
            before = self.world.status().tick
            status = self.world.advance(min(self.chunk, owed - done))
            done += status.tick - before
            yield Progress(done=done, owed=owed, status=status)
            if status.collapsed:
                return
