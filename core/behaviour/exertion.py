"""Exertion: the record of how hard an animal has been working (issue #107).

`Fatigue` scored `weight × (1 - health)` and nothing else, because when #22 shipped the drives
nothing spent effort. #25 then landed movement, so effort *is* spent — but it goes straight out of
the energy pool through `Ecology.spend`, and a pool level records what an animal has *left*, not
what it just did. A creature that sprinted across a ridge and one that stood still all tick were
indistinguishable to fatigue as long as their health matched, so resting was only ever selected for
as recovery from injury and the drive's whole ecological purpose — an animal that has run itself
ragged stops running — was absent.

This is the column that was missing. It exists rather than being derived from what is already
stored, because neither candidate works:

- **The energy pool** is hunger's signal. Two drives reading one number is how a drive contest
  becomes a coin flip, and hunger already scores on exactly that quantity.
- **Distance covered between position snapshots** is available from `TickLoop`, but those snapshots
  are taken once per `advance()` call rather than once per tick, so a catch-up batch of a thousand
  ticks would report one straight line — and §2.4 requires that batching ticks differently must not
  change what the world does.

**Work per unit of body size, not joules.** The bill movement pays is ``size × (haul + climb)``;
what accumulates here is the parenthesised half. Exertion is then a statement about how hard *this*
animal worked rather than how much fuel it burned, so one saturation constant means the same thing
to a mouse and to an elephant. Raw joules would make a large animal permanently exhausted by an
ordinary walk and would leave `FatigueConfig.exertion_saturation` quietly meaning a different
tiredness per body size — the same class of trap as a coefficient whose unit depends on something
undeclared (#112).

**It sheds geometrically and is never charged for.** Recovery costs nothing beyond the upkeep an
animal pays to exist: resting is the *absence* of expenditure, and charging for it would make
resting a third way to starve rather than the escape from exertion it exists to be.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.entities.store import EntityStore
from core.selection import Selection
from core.services import ColumnRegistry, DomainService


@dataclass(frozen=True)
class ExertionConfig:
    """Per-world tuning for how fast effort is shed.

    recovery_rate: fraction of accumulated exertion shed per tick, in (0, 1]. Must be positive:
        at zero nothing ever recovers, every animal that has moved saturates permanently, and
        fatigue stops being a signal at all. At 1.0 exertion clears completely each tick, so
        fatigue reflects only the tick just gone — a legitimate extreme rather than a degenerate
        one, which is why the range is closed at that end.

        Geometric rather than a fixed subtraction, because a fixed one would let a hard enough
        tick outrun recovery without bound while an easy one floored at zero — so the same rate
        would mean "recovers quickly" for a walker and "never recovers" for a sprinter. A
        fraction gives every animal the same half-life.
    """

    recovery_rate: float

    def __post_init__(self) -> None:
        if not 0.0 < self.recovery_rate <= 1.0:
            raise ValueError(
                f"recovery_rate must be in (0, 1], got {self.recovery_rate}; at zero exertion "
                "never sheds and every animal that has ever moved is permanently exhausted"
            )


class Exertion(DomainService):
    """Owns the `exertion` column: recent work per unit of body size, decaying every tick.

    Movement hands over what a step took exactly as it hands `Ecology` the bill for it — this
    service owns the column, so a mover cannot add to it directly (CLAUDE.md §2.3). That is the
    same shape as `Ecology.spend`, and for the same reason: as #19's chase and #24's flight come
    to spend effort, each is one more call here rather than one more writer of a shared array.
    """

    owns = ("exertion",)

    # Narrows DomainService.store (typed `object`, the base being store-shape-agnostic) to the
    # concrete EntityStore whose `exertion` column this service writes.
    store: EntityStore

    def __init__(
        self, store: EntityStore, registry: ColumnRegistry, config: ExertionConfig
    ) -> None:
        super().__init__(store, registry)
        self.config = config

    def exerted(self, selection: Selection) -> np.ndarray:
        """(len(selection),) float64: accumulated work per unit size, in ascending row order.

        Read by `Fatigue`, which owns turning it into an urgency. Handing back the raw quantity
        rather than a 0-to-1 shape keeps the saturation point in the drive's config, where the
        rest of the drive's tuning lives.
        """
        return self.store.exertion[selection.to_mask()].astype(np.float64)

    def accumulate(self, selection: Selection, work: np.ndarray) -> None:
        """Add `work` to `selection`'s exertion.

        work: (len(selection),) work per unit of expressed size — the size-independent half of the
            movement bill, in ascending row order.

        Raises ValueError on a negative entry: work is effort spent, and a negative one would make
        moving a way to *become* less tired, which would have an animal sprint to rest (§8.7).
        """
        work = np.asarray(work, dtype=np.float64)
        if work.shape != (len(selection),):
            # Checked rather than left to NumPy: a scalar or length-1 array broadcasts cleanly and
            # would credit one animal's exertion to the whole selection.
            raise ValueError(
                f"work must have shape ({len(selection)},) for {len(selection)} entities; "
                f"got {work.shape}"
            )
        if np.any(work < 0.0):
            raise ValueError("work must be non-negative; a negative entry would rest an animal")
        self.write(
            "exertion", selection, (self.exerted(selection) + work).astype(np.float32)
        )

    def recover(self, selection: Selection) -> None:
        """Shed one tick's worth of recovery from `selection`.

        Runs over the whole living population once per tick, **before drives are scored** (§2.1's
        system order): fatigue is scored from what an animal has recovered *to*, and movement then
        adds this tick's work after the decision it informed. Ordered that way, a tick spent
        standing still strictly lowers the next fatigue score, which is what makes resting a
        strategy rather than a state.

        Every entity recovers, not only the ones that moved — an animal that did not move is
        precisely the one recovery is *for*, and a system that only touched movers would leave the
        resting population permanently at whatever exertion it last reached.
        """
        self.write(
            "exertion",
            selection,
            (self.exerted(selection) * (1.0 - self.config.recovery_rate)).astype(np.float32),
        )
