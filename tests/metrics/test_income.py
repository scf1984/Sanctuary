"""Intervention income: ecosystem value, paid on the recorded population (#308, #26).

Written before `metrics.income` (§8.1): the owner's four decisions on #308 are each checkable in
advance. Income pays on the **trailing minimum** of the living population over a window, counts
**ticks** rather than seconds, accrues the same whether or not anyone watches, and never falls
below the **floor** #26 settled.
"""

from types import SimpleNamespace

import pytest

from core.world.interventions import Interventions
from metrics import Income, IncomeConfig, MetricHistory, MetricsConfig
from worlds.demo import build_demo_world


class History:
    """The two things income reads from a recorder: its samples and its cadence. A stand-in for
    `MetricHistory` so a series can be written down rather than grown."""

    def __init__(self, every_n_ticks=20):
        self.samples = []
        self.config = SimpleNamespace(every_n_ticks=every_n_ticks, history_limit=1_000)
        self.subscribers = []

    def subscribe(self, callback):
        self.subscribers.append(callback)

    def record(self, tick, living):
        sample = SimpleNamespace(tick=tick, living=living)
        self.samples.append(sample)
        for callback in self.subscribers:
            callback(sample)


CONFIG = IncomeConfig(per_animal=0.01, floor=0.5, window=60)


def paid(series, config=CONFIG, every=20):
    """Record `series` as consecutive samples and return the ledger they paid into."""
    history = History(every)
    ledger = Interventions(balance=0.0)
    Income(config, history, ledger)
    for i, living in enumerate(series, start=1):
        history.record(i * every, living)
    return ledger


class TestTheTrailingMinimumIsWhatPays:
    def test_a_steady_population_earns_its_size_per_tick(self):
        ledger = paid([300])
        # 0.01 per animal per tick, 300 animals, for the 20 ticks the sample covers.
        assert ledger.balance == pytest.approx(0.01 * 300 * 20)
        assert ledger.income_per_tick == pytest.approx(3.0)

    def test_a_boom_pays_only_once_it_has_held(self):
        """Window 60 ticks is three samples: the dip to 100 is still inside it at the boom."""
        ledger = paid([400, 100, 2_000])
        assert ledger.income_per_tick == pytest.approx(0.01 * 100)

    def test_a_dip_stops_counting_once_it_leaves_the_window(self):
        ledger = paid([400, 100, 2_000, 2_000, 2_000])
        assert ledger.income_per_tick == pytest.approx(0.01 * 2_000)

    def test_a_crash_cuts_income_at_once(self):
        ledger = paid([2_000, 2_000, 50])
        assert ledger.income_per_tick == pytest.approx(0.5)  # 0.01 × 50 = the floor exactly


class TestTheFloor:
    """#26: a minimum income, so a world one bad decision from collapse can still act, slowly."""

    def test_income_never_falls_below_the_floor(self):
        ledger = paid([3])
        assert ledger.income_per_tick == pytest.approx(CONFIG.floor)

    def test_an_empty_world_still_earns_the_floor(self):
        assert paid([0]).income_per_tick == pytest.approx(CONFIG.floor)


class TestIncomeCountsTicks:
    """#308 decision 2: per tick, so the recording cadence decides how often income lands, never
    how much a stretch of world time is worth."""

    def test_the_cadence_does_not_change_what_a_stretch_of_world_time_earns(self):
        coarse = paid([500] * 3, every=40)  # 120 ticks in three samples
        fine = paid([500] * 6, every=20)  # 120 ticks in six
        assert coarse.balance == pytest.approx(fine.balance)

    def test_earnings_are_tallied_apart_from_the_balance(self):
        """So a ledger reconciles: balance = start + earned − what was spent."""
        history = History()
        ledger = Interventions(balance=100.0)
        Income(CONFIG, history, ledger)
        history.record(20, 300)
        assert ledger.earned == pytest.approx(60.0)
        assert ledger.balance == pytest.approx(160.0)


class TestTheConfig:
    def test_rates_must_not_be_negative(self):
        with pytest.raises(ValueError, match="per_animal"):
            IncomeConfig(per_animal=-0.1, floor=0.5, window=60)
        with pytest.raises(ValueError, match="floor"):
            IncomeConfig(per_animal=0.1, floor=-0.5, window=60)

    def test_the_window_must_cover_at_least_one_sample(self):
        with pytest.raises(ValueError, match="window"):
            Income(IncomeConfig(per_animal=0.1, floor=0.5, window=10), History(20), Interventions(0.0))

    def test_the_window_must_fit_in_what_the_recorder_keeps(self):
        """A window longer than the kept history would silently read a shorter one (§8.7)."""
        history = History(20)
        history.config.history_limit = 2
        with pytest.raises(ValueError, match="window"):
            Income(IncomeConfig(per_animal=0.1, floor=0.5, window=60), history, Interventions(0.0))


class TestCreditingTheLedger:
    def test_a_credit_cannot_be_negative(self):
        with pytest.raises(ValueError, match="negative"):
            Interventions(balance=0.0).credit(per_tick=-1.0, ticks=20)


class TestInAWorld:
    """The demo world wires income in its composition root (`worlds.demo`)."""

    def test_the_demo_world_earns_as_it_runs(self):
        world = build_demo_world(seed=1, n_entities=60)
        start = world.loop.interventions.balance
        world.loop.advance(120)
        ledger = world.loop.interventions
        assert ledger.earned > 0
        assert ledger.balance == pytest.approx(start + ledger.earned)

    def test_batching_does_not_change_what_a_world_earns(self):
        """§2.4: catch-up is identical to live play, and #308 decision 3 makes income part of it."""
        one = build_demo_world(seed=4, n_entities=60)
        many = build_demo_world(seed=4, n_entities=60)
        one.loop.advance(200)
        for _ in range(200):
            many.loop.advance(1)
        assert one.loop.interventions.balance == many.loop.interventions.balance
        assert one.loop.interventions.earned == many.loop.interventions.earned

    def test_a_real_recorder_pays_on_its_own_samples(self):
        world = build_demo_world(seed=2, n_entities=40)
        history = MetricHistory(
            world.store, world.genetics, world.plants, world.speciation, world.genes.vocabulary,
            MetricsConfig(every_n_ticks=10, history_limit=100),
        )
        ledger = Interventions(balance=0.0)
        world.loop.metrics = history
        world.loop.interventions = ledger
        Income(IncomeConfig(per_animal=0.01, floor=0.0, window=30), history, ledger)
        world.loop.advance(30)
        lowest = min(sample.living for sample in history.samples)
        assert ledger.income_per_tick == pytest.approx(0.01 * lowest)
