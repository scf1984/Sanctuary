"""Intervention income: ecosystem value, paid on the recorded population (#308, #26).

The owner chose ecosystem value on #26 (income scales with how rich and stable the world is) and a
minimum income as the floor under a declining world. #308 settled the slice's formula:

    per tick = max(floor, per_animal × fewest animals alive at any sample in the last `window`)

paid at every metrics sample for the ticks that sample covers.

- **The trailing minimum is both "rich" and "stable"** (#308, decision 1). A boom pays only once
  it has held for a window, and a crash cuts income at once and keeps it cut for a window. It does
  not penalise swings above the trough, and it rewards size linearly, both accepted on #308.
- **Per tick, never per second** (decision 2). Income is a function of the recorded series and
  nothing else, so a restored world has the balance it was saved with (§3.2) and the wall clock
  stays out of everything but the count of ticks owed (§2.1).
- **The same whether or not anyone watches** (decision 3). It is paid inside the tick loop, which
  catch-up runs exactly as live play does (§2.4). Nothing here knows who is looking.
- **No species term yet** (decision 4): it waits on #30's species metric.

It lives in `metrics/` because it reads the recorded series (§4: `metrics/` reads `core/`), and it
writes only the ledger, through `Interventions.credit`.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.world.interventions import Interventions
from metrics.history import MetricHistory, Sample


@dataclass(frozen=True)
class IncomeConfig:
    """Per-world income rates. Both are tuning for playtests (#292 phase 5), not decisions.

    per_animal: currency units per animal per tick, on the trailing minimum.
    floor: currency units per tick, paid however small the population (#26's minimum income).
    window: ticks the trailing minimum looks back over. The demo world takes one generation (#308).
    """

    per_animal: float
    floor: float
    window: int

    def __post_init__(self) -> None:
        if self.per_animal < 0:
            raise ValueError(f"per_animal cannot be negative, got {self.per_animal}")
        if self.floor < 0:
            raise ValueError(f"floor cannot be negative, got {self.floor}")


class Income:
    """Pays `ledger` at every sample `history` records. Subscribes itself when built, so a world's
    income is one line in its composition root."""

    def __init__(self, config: IncomeConfig, history: MetricHistory, ledger: Interventions) -> None:
        cadence = history.config.every_n_ticks
        # A window shorter than the cadence holds no sample but the newest, so it would not be a
        # trailing minimum at all; one longer than the kept history would silently read a shorter
        # one. Both are refused rather than quietly meaning something else (§8.7).
        if config.window < cadence:
            raise ValueError(
                f"window ({config.window} ticks) is shorter than the {cadence}-tick sample cadence"
            )
        if config.window > cadence * history.config.history_limit:
            raise ValueError(
                f"window ({config.window} ticks) is longer than the history keeps "
                f"({history.config.history_limit} samples of {cadence} ticks)"
            )
        self.config = config
        self.history = history
        self.ledger = ledger
        history.subscribe(self.pay)

    def per_tick(self, tick: int) -> float:
        """What the world earns a tick at `tick`, from the samples recorded in the window."""
        lowest = min(
            sample.living for sample in self.history.samples if sample.tick > tick - self.config.window
        )
        return max(self.config.floor, self.config.per_animal * lowest)

    def pay(self, sample: Sample) -> None:
        """Credit the ticks this sample covers, at the rate the window gives."""
        self.ledger.credit(self.per_tick(sample.tick), self.history.config.every_n_ticks)
