import pytest

from clients.viewer.playback import Playback


class TestConstruction:
    def test_rejects_non_positive_ticks_per_second(self):
        with pytest.raises(ValueError):
            Playback(ticks_per_second=0)
        with pytest.raises(ValueError):
            Playback(ticks_per_second=-1)

    def test_rejects_non_positive_speed(self):
        with pytest.raises(ValueError):
            Playback(ticks_per_second=10, speed=0)

    def test_starts_unpaused(self):
        assert Playback(ticks_per_second=10).paused is False


class TestRunning:
    def test_whole_tick_owed_after_exact_interval(self):
        playback = Playback(ticks_per_second=10)
        n_ticks, alpha = playback.advance(0.1)
        assert n_ticks == 1
        assert alpha == pytest.approx(0.0)

    def test_partial_tick_owes_zero_ticks_and_fractional_alpha(self):
        playback = Playback(ticks_per_second=10)
        n_ticks, alpha = playback.advance(0.05)
        assert n_ticks == 0
        assert alpha == pytest.approx(0.5)

    def test_debt_accumulates_across_calls(self):
        playback = Playback(ticks_per_second=10)
        playback.advance(0.05)
        n_ticks, alpha = playback.advance(0.05)
        assert n_ticks == 1
        assert alpha == pytest.approx(0.0)

    def test_speed_multiplier_scales_ticks_owed(self):
        playback = Playback(ticks_per_second=10, speed=2.0)
        n_ticks, _ = playback.advance(0.1)
        assert n_ticks == 2

    def test_set_speed_takes_effect_on_next_advance(self):
        playback = Playback(ticks_per_second=10)
        playback.set_speed(3.0)
        n_ticks, _ = playback.advance(0.1)
        assert n_ticks == 3

    def test_set_speed_rejects_non_positive(self):
        playback = Playback(ticks_per_second=10)
        with pytest.raises(ValueError):
            playback.set_speed(0)

    def test_negative_elapsed_seconds_raises(self):
        playback = Playback(ticks_per_second=10)
        with pytest.raises(ValueError):
            playback.advance(-0.1)


class TestPause:
    def test_paused_owes_no_ticks(self):
        playback = Playback(ticks_per_second=10)
        playback.toggle_pause()
        n_ticks, alpha = playback.advance(1.0)
        assert n_ticks == 0
        assert alpha == pytest.approx(1.0)

    def test_time_elapsed_while_paused_is_discarded_not_queued(self):
        playback = Playback(ticks_per_second=10)
        playback.toggle_pause()
        playback.advance(10.0)  # a long pause: no ticks owed
        playback.toggle_pause()  # resume
        n_ticks, _ = playback.advance(0.0)
        assert n_ticks == 0  # no burst of catch-up ticks from the paused interval

    def test_toggle_pause_resumes(self):
        playback = Playback(ticks_per_second=10)
        playback.toggle_pause()
        playback.toggle_pause()
        n_ticks, _ = playback.advance(0.1)
        assert n_ticks == 1


class TestStep:
    def test_step_owes_exactly_one_tick_at_full_alpha(self):
        playback = Playback(ticks_per_second=10)
        playback.request_step()
        n_ticks, alpha = playback.advance(0.0)
        assert n_ticks == 1
        assert alpha == pytest.approx(1.0)

    def test_step_pauses_playback(self):
        playback = Playback(ticks_per_second=10)
        playback.request_step()
        playback.advance(0.0)
        assert playback.paused is True

    def test_step_while_already_paused_still_steps(self):
        playback = Playback(ticks_per_second=10)
        playback.toggle_pause()
        playback.request_step()
        n_ticks, alpha = playback.advance(0.0)
        assert n_ticks == 1
        assert alpha == pytest.approx(1.0)

    def test_multiple_queued_steps_are_consumed_together(self):
        playback = Playback(ticks_per_second=10)
        playback.request_step()
        playback.request_step()
        n_ticks, _ = playback.advance(0.0)
        assert n_ticks == 2

    def test_step_consumes_pending_debt_so_resuming_does_not_burst(self):
        playback = Playback(ticks_per_second=10)
        playback.advance(0.15)  # 1 tick owed, 0.5 tick left in debt
        playback.request_step()
        n_ticks, _ = playback.advance(0.0)
        assert n_ticks == 1  # only the queued step, not the leftover debt too
        playback.toggle_pause()  # resume
        n_ticks, _ = playback.advance(0.0)
        assert n_ticks == 0  # debt was cleared by the step, not carried forward
