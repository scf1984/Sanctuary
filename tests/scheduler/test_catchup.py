"""Catch-up (#32): the world owes what real time says, and pays it exactly as live play would.

The done-when asks for catch-up to be *statistically* indistinguishable from live advancement.
It is tested as **identical**, which is stronger and cheaper: batching runs the same systems over
the same ticks, so the same seed gives the same world bit for bit. A statistical test would only
be needed if that ever stopped holding, and then it would be needed because something broke.
"""

import math

import numpy as np
import pytest

import api
from api import Status
from scheduler import Anchor, CatchUp, Stage, WakeSchedule

HOUR = 3_600.0
DAY = 86_400.0


class CountingWorld:
    """A world that only counts ticks: what the schedule arithmetic needs, at no cost."""

    def __init__(self, collapses_at=None):
        self.tick = 0
        self.calls = []
        self.collapses_at = collapses_at

    def _collapsed(self):
        return self.collapses_at is not None and self.tick >= self.collapses_at

    def advance(self, ticks):
        self.calls.append(ticks)
        for _ in range(ticks):
            if self._collapsed():
                break
            self.tick += 1
        return self.status()

    def status(self):
        return Status(
            tick=self.tick,
            collapsed=self._collapsed(),
            living=0 if self._collapsed() else 1,
            splits=0,
        )


def run_to_end(catchup, now):
    progress = list(catchup.run(now))
    return progress[-1] if progress else None


def wake_times(schedule, start, end, watched):
    """Every wake between `start` and `end` under `schedule`, for a player last seen at `start`
    (watched throughout or away throughout), ending with a wake at `end`."""
    times, now = [], start
    while True:
        now += schedule.interval(watched=watched, away=now - start)
        if now >= end:
            return times + [end]
        times.append(now)


class TestAnchor:
    def test_the_world_owes_one_tick_per_tick_length_of_real_time(self):
        anchor = Anchor(wall_seconds=1_000.0, tick=50, ticks_per_second=1.0)

        assert anchor.due(1_000.0) == 50
        assert anchor.due(1_010.9) == 60

    def test_a_partial_tick_is_not_owed_until_it_is_whole(self):
        anchor = Anchor(wall_seconds=0.0, tick=0, ticks_per_second=2.0)

        assert anchor.due(1.49) == 2
        assert anchor.due(1.5) == 3

    def test_a_clock_stepped_backwards_owes_nothing_rather_than_rewinding(self):
        anchor = Anchor(wall_seconds=1_000.0, tick=50, ticks_per_second=1.0)

        assert anchor.due(400.0) == 50

    @pytest.mark.parametrize("rate", [0.0, -1.0])
    def test_the_sim_rate_must_be_positive(self, rate):
        with pytest.raises(ValueError, match="positive"):
            Anchor(wall_seconds=0.0, tick=0, ticks_per_second=rate)


class TestRun:
    def test_it_pays_exactly_what_is_owed(self):
        world = CountingWorld()
        catchup = CatchUp(world, Anchor(0.0, 0, 1.0), chunk=100)

        last = run_to_end(catchup, now=950.0)

        assert world.tick == 950
        assert last.done == last.owed == 950
        assert catchup.owed(950.0) == 0

    def test_it_reports_after_every_chunk(self):
        """A long absence can take minutes to pay off, and a caller streaming progress (#36) needs
        to hear about it as it goes."""
        world = CountingWorld()

        progress = list(CatchUp(world, Anchor(0.0, 0, 1.0), chunk=100).run(now=950.0))

        assert [p.done for p in progress] == [100, 200, 300, 400, 500, 600, 700, 800, 900, 950]
        assert world.calls == [100] * 9 + [50]

    def test_nothing_owed_does_nothing(self):
        world = CountingWorld()

        assert list(CatchUp(world, Anchor(10.0, 0, 1.0), chunk=10).run(now=10.0)) == []
        assert world.calls == []

    def test_a_world_already_ahead_owes_nothing(self):
        """A world that was advanced by hand past its anchor (a test, a debugging session) is not
        rewound and does not go negative."""
        world = CountingWorld()
        world.tick = 500

        assert CatchUp(world, Anchor(0.0, 0, 1.0), chunk=10).owed(now=100.0) == 0

    def test_a_collapsed_world_stops_and_owes_nothing_more(self):
        """§2.7: an empty world can never refill. Its ticks are owed to nobody, and a debt that
        kept growing would have a caller paying it forever."""
        world = CountingWorld(collapses_at=250)
        catchup = CatchUp(world, Anchor(0.0, 0, 1.0), chunk=100)

        last = run_to_end(catchup, now=1_000.0)

        assert world.tick == 250
        assert last.status.collapsed
        assert last.done == 250
        assert catchup.owed(now=5_000.0) == 0

    def test_the_target_is_fixed_when_a_run_starts(self):
        """Ticks that come due while a run is still going belong to the next wake, so a catch-up
        slower than its own interval still finishes."""
        world = CountingWorld()
        catchup = CatchUp(world, Anchor(0.0, 0, 1.0), chunk=10)

        progress = list(catchup.run(now=100.0))

        assert progress[-1].owed == 100

    def test_a_chunk_must_be_at_least_one_tick(self):
        with pytest.raises(ValueError, match="at least one tick"):
            CatchUp(CountingWorld(), Anchor(0.0, 0, 1.0), chunk=0)


class TestTheScheduleDecidesOnlyWhen:
    """§2.4: waking hourly and then daily batches the same ticks differently, and must never make
    the world age more slowly. This is #32's done-when, as a property over several schedules."""

    @pytest.mark.parametrize(
        "schedule, watched",
        [
            (WakeSchedule(1.0, (Stage(math.inf, HOUR),)), False),
            (WakeSchedule(1.0, (Stage(DAY, HOUR), Stage(math.inf, DAY))), False),
            (WakeSchedule(1.0, (Stage(600.0, 7.3), Stage(DAY, 1_234.5), Stage(math.inf, DAY))), False),
            (WakeSchedule(0.37, (Stage(math.inf, DAY),)), True),
        ],
        ids=["uniform-hourly", "hourly-then-daily", "irregular", "watched"],
    )
    def test_every_schedule_reaches_the_same_tick(self, schedule, watched):
        start, end = 5_000.0, 5_000.0 + 3 * DAY
        if watched:
            end = start + 2_000.0  # watching wakes every 0.37 s; a shorter span keeps this quick
        anchor = Anchor(wall_seconds=start, tick=0, ticks_per_second=1.0)
        world = CountingWorld()
        catchup = CatchUp(world, anchor, chunk=10_000)

        for now in wake_times(schedule, start, end, watched):
            run_to_end(catchup, now)

        assert world.tick == anchor.due(end)

    def test_fractional_rates_do_not_drift_across_many_wakes(self):
        """Summing each wake's own gap would round at every wake. From a fixed anchor, a thousand
        wakes owe exactly what one does."""
        anchor = Anchor(wall_seconds=0.0, tick=0, ticks_per_second=0.7)
        often, once = CountingWorld(), CountingWorld()
        for i in range(1, 1_001):
            run_to_end(CatchUp(often, anchor, chunk=100), now=i * 1.3)
        run_to_end(CatchUp(once, anchor, chunk=100), now=1_000 * 1.3)

        assert often.tick == once.tick == anchor.due(1_300.0)


def world_state(session):
    """Everything a client can read about a world, as comparable values."""
    info = session.describe()
    return (
        session.status(),
        session.frame(),
        session.ledger(),
        tuple(session.field(layer) for layer in info.field_layers),
        tuple(session.condition(mode) for mode in info.conditions),
        session.history(),
    )


class TestCatchUpIsLivePlay:
    """The other half of #32's done-when, on a real world: caught up on a decaying schedule, it
    ends identical to one advanced a tick at a time, the way a watched world is."""

    TICKS = 400

    def test_a_world_caught_up_in_batches_is_the_world_advanced_tick_by_tick(self):
        live = api.create("demo", seed=5, founders=60)
        for _ in range(self.TICKS):
            live.advance(1)

        caught_up = api.create("demo", seed=5, founders=60)
        anchor = Anchor(wall_seconds=0.0, tick=0, ticks_per_second=1.0)
        schedule = WakeSchedule(1.0, (Stage(60.0, 7.0), Stage(math.inf, 97.0)))
        for now in wake_times(schedule, 0.0, float(self.TICKS), watched=False):
            for _progress in CatchUp(caught_up, anchor, chunk=25).run(now):
                pass

        assert caught_up.status().tick == self.TICKS
        assert world_state(caught_up) == world_state(live)

    def test_an_intervention_lands_at_the_same_tick_either_way(self):
        """A request waits for the next tick boundary however the ticks are batched, so a cull
        queued between wakes is charged at the tick it would have been charged live."""

        def run(batched):
            session = api.create("demo", seed=5, founders=60)
            if batched:
                for _ in CatchUp(session, Anchor(0.0, 0, 1.0), chunk=50).run(now=120.0):
                    pass
            else:
                for _ in range(120):
                    session.advance(1)
            session.request(api.CullRequest(species_id=0, count=3))
            if batched:
                for _ in CatchUp(session, Anchor(0.0, 0, 1.0), chunk=50).run(now=300.0):
                    pass
            else:
                for _ in range(180):
                    session.advance(1)
            return world_state(session)

        assert run(batched=True) == run(batched=False)

    def test_the_states_compared_are_not_trivially_equal(self):
        """Otherwise the two tests above could pass by comparing nothing."""
        a = api.create("demo", seed=5, founders=60)
        b = api.create("demo", seed=6, founders=60)
        a.advance(20)
        b.advance(20)

        assert world_state(a) != world_state(b)
        assert np.any(a.frame().x.to_numpy() != 0)
