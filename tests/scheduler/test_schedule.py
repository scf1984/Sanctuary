"""The wake schedule (#32): when a world wakes, and nothing else."""

import math

import pytest

from scheduler import Stage, WakeSchedule

HOUR = 3_600.0
DAY = 86_400.0


def decaying():
    """Hourly for a day, then daily: §2.4's own example."""
    return WakeSchedule(watched_interval=1.0, absent=(Stage(DAY, HOUR), Stage(math.inf, DAY)))


class TestInterval:
    def test_a_watched_world_wakes_on_the_short_interval_however_long_ago_anyone_left(self):
        """Gate 2 on #35: watching is the same executor on a shorter interval, not a live loop."""
        schedule = decaying()

        assert schedule.interval(watched=True, away=0.0) == 1.0
        assert schedule.interval(watched=True, away=10 * DAY) == 1.0

    def test_an_absence_decays_through_its_stages(self):
        schedule = decaying()

        assert schedule.interval(watched=False, away=0.0) == HOUR
        assert schedule.interval(watched=False, away=DAY - 1.0) == HOUR
        assert schedule.interval(watched=False, away=DAY) == DAY
        assert schedule.interval(watched=False, away=100 * DAY) == DAY


class TestValidation:
    def test_the_last_stage_must_last_forever(self):
        """Otherwise an absence past its end would have no wake at all, and the world would stop
        catching up exactly when the player has been away longest."""
        with pytest.raises(ValueError, match="last forever"):
            WakeSchedule(watched_interval=1.0, absent=(Stage(DAY, HOUR),))

    def test_stages_must_end_in_order(self):
        with pytest.raises(ValueError, match="increasing order"):
            WakeSchedule(
                watched_interval=1.0,
                absent=(Stage(DAY, HOUR), Stage(HOUR, 60.0), Stage(math.inf, DAY)),
            )

    @pytest.mark.parametrize("interval", [0.0, -1.0])
    def test_an_interval_must_be_positive(self, interval):
        with pytest.raises(ValueError, match="positive"):
            WakeSchedule(watched_interval=1.0, absent=(Stage(math.inf, interval),))
        with pytest.raises(ValueError, match="positive"):
            WakeSchedule(watched_interval=interval, absent=(Stage(math.inf, HOUR),))

    def test_a_schedule_needs_somewhere_to_decay_to(self):
        with pytest.raises(ValueError, match="at least one"):
            WakeSchedule(watched_interval=1.0, absent=())
