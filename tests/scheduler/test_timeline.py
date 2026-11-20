"""Rate segments (#307): pause and fast-forward without giving up the fixed anchor (§2.4).

A rate change is another fixed point, so the tick due at any moment is read off the segment in
force then. These tests pin the arithmetic, and then the property #32 pinned for the bare anchor:
how a schedule batches its wakes cannot change how old the world is.
"""

import math

import pytest

from scheduler import Anchor, CatchUp, RateChange, Stage, Timeline, WakeSchedule
from tests.scheduler.test_catchup import CountingWorld, run_to_end, wake_times

ANCHOR = Anchor(wall_seconds=100.0, tick=0, ticks_per_second=1.0)


class TestSegments:
    def test_with_no_changes_it_is_the_anchor(self):
        timeline = Timeline(ANCHOR)

        assert [timeline.due(t) for t in (50.0, 100.0, 160.5)] == [0, 0, 60]
        assert timeline.ticks_per_second == timeline.own_ticks_per_second == 1.0

    def test_a_pause_holds_the_world_at_the_tick_it_was_due(self):
        paused = Timeline(ANCHOR).at_rate(130.0, 0.0)

        assert paused.changes == (RateChange(wall_seconds=130.0, tick=30, ticks_per_second=0.0),)
        assert paused.due(130.0) == paused.due(10_000.0) == 30

    def test_a_fast_forward_runs_from_the_tick_it_was_due(self):
        fast = Timeline(ANCHOR).at_rate(130.0, 8.0)

        assert fast.due(131.0) == 38
        assert fast.ticks_per_second == 8.0

    def test_a_change_lands_on_the_last_whole_tick_and_is_never_retroactive(self):
        """§2.7: at a tick boundary. The tick in progress is dropped rather than replayed at the
        new rate, which would owe a fast-forward for the instant before it was asked for."""
        fast = Timeline(ANCHOR).at_rate(130.9, 100.0)

        assert fast.changes[-1].tick == 30
        assert fast.due(130.9) == 30

    def test_every_moment_reads_the_tick_it_was_then(self):
        """The series is the world's history of time: its age at any past moment is recoverable."""
        timeline = (
            Timeline(ANCHOR).at_rate(130.0, 0.0).at_rate(200.0, 4.0).at_own_rate(210.0)
        )

        assert [timeline.due(t) for t in (90.0, 120.0, 150.0, 205.0, 210.0, 220.0)] == [
            0, 20, 30, 50, 70, 80,
        ]

    def test_asking_for_the_rate_already_set_records_nothing(self):
        timeline = Timeline(ANCHOR).at_rate(130.0, 0.0)

        assert timeline.at_rate(140.0, 0.0) is timeline
        assert Timeline(ANCHOR).at_own_rate(500.0).changes == ()

    def test_a_clock_stepped_back_keeps_the_series_in_order_and_owes_nothing(self):
        paused = Timeline(ANCHOR).at_rate(130.0, 0.0)

        resumed = paused.at_own_rate(120.0)

        assert resumed.changes[-1].wall_seconds == 130.0
        assert resumed.due(130.0) == 30
        assert resumed.due(140.0) == 40

    @pytest.mark.parametrize("rate", [-1.0, math.nan, math.inf])
    def test_a_rate_is_zero_or_a_positive_number_of_ticks(self, rate):
        with pytest.raises(ValueError, match="rate"):
            RateChange(wall_seconds=0.0, tick=0, ticks_per_second=rate)

    def test_a_series_out_of_order_is_refused(self):
        """A timeline read back from disk is checked, since a reordered one would make the world's
        age at a moment depend on which change was found first."""
        with pytest.raises(ValueError, match="follow one another"):
            Timeline(ANCHOR, (RateChange(200.0, 100, 0.0), RateChange(150.0, 50, 1.0)))
        with pytest.raises(ValueError, match="follow one another"):
            Timeline(ANCHOR, (RateChange(50.0, 0, 0.0),))


class TestBatchingStillCannotChangeTheWorldsAge:
    """#32's done-when, with time controls in the timeline: every schedule reaches the same tick."""

    TIMELINE = (
        Timeline(Anchor(0.0, 0, 1.0))
        .at_rate(1_000.3, 0.0)
        .at_rate(1_700.0, 12.5)
        .at_rate(1_900.7, 0.7)
        .at_own_rate(2_500.0)
    )

    @pytest.mark.parametrize(
        "schedule",
        [
            WakeSchedule(1.0, (Stage(math.inf, 1.0),)),
            WakeSchedule(1.0, (Stage(math.inf, 7.3),)),
            WakeSchedule(1.0, (Stage(600.0, 13.0), Stage(math.inf, 501.0))),
        ],
        ids=["every-second", "irregular", "decaying"],
    )
    def test_every_schedule_reaches_the_same_tick(self, schedule):
        world = CountingWorld()
        catchup = CatchUp(world, self.TIMELINE, chunk=1_000)

        for now in wake_times(schedule, 0.0, 3_000.0, watched=False):
            run_to_end(catchup, now)

        assert world.tick == self.TIMELINE.due(3_000.0)

    def test_one_wake_owes_what_many_do(self):
        often, once = CountingWorld(), CountingWorld()
        for i in range(1, 2_001):
            run_to_end(CatchUp(often, self.TIMELINE, chunk=100), now=i * 1.5)
        run_to_end(CatchUp(once, self.TIMELINE, chunk=100), now=3_000.0)

        assert often.tick == once.tick == self.TIMELINE.due(3_000.0)

    def test_a_backlog_owed_before_a_pause_is_still_paid(self):
        """A pause is made against the tick due, not the tick reached. Forgiving the backlog would
        make what a pause forgave depend on when the last wake was."""
        world = CountingWorld()
        paused = Timeline(Anchor(0.0, 0, 1.0)).at_rate(500.0, 0.0)

        run_to_end(CatchUp(world, paused, chunk=50), now=900.0)

        assert world.tick == 500
