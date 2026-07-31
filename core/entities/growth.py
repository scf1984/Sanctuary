"""Capacity growth at the tick boundary (CLAUDE.md §2.3, issue #127).

`EntityStore.grow` has existed since #4 and **had no caller anywhere** — not in `core/`, not in
`clients/`, not in #115's assembly. `allocate` raises `EntityStoreFull` and tells the caller to
"call grow() at the next tick boundary before retrying", and no tick boundary ever did.

It stayed that way deliberately: until #20 there was nothing to grow *for*. The assembly allocated
its founders once, at build time, into a store sized to hold exactly them, and no registered system
allocated a row during a tick. A growth policy written then would have been a hook with no caller
and, worse, a threshold chosen against a guess about birth rates rather than a world that breeds.

**Why the boundary, and why this is not a system.** A system runs mid-tick, with other systems still
to follow it, and §2.3 forbids growing there — not for the stall but because `grow` replaces every
column array with a new object. A vectorized operation elsewhere may be holding a NumPy view into
`store.energy` for the duration of a tick's computation; that view stays valid and keeps its
pre-growth values, so the failure is *silently wrong results* rather than a crash. At a boundary no
system holds a view and no `Selection` is live, which is the whole reason the boundary exists.

**Why a reserve rather than waiting for `EntityStoreFull`.** That exception fires mid-tick, which is
the one place growth may not happen, so a policy that waited for it could never act. The check has
to run *before* the store is full and therefore has to anticipate.

**What it is not.** Not a population cap, and not a brake. §2.5 makes population emergent —
carrying capacity is area × primary productivity ÷ per-animal upkeep — so the array follows the
ecology rather than bounding it, and §2.3's engine ceiling is meant to be invisible because the
ecology plateaus first. Measured in `docs/spikes/conception-and-capacity.md`: a demo world given
headroom grows from 200 founders past 3,000 within a thousand ticks and is still climbing, which is
the ecology asking for room rather than a runaway to be clamped.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.entities.store import EntityStore


@dataclass(frozen=True)
class GrowthConfig:
    """Per-world capacity policy — never constants in `core/` (§2.1).

    reserve_fraction: grow when free rows fall below this fraction of the rows in use. A
        *fraction* rather than a count because what consumes rows scales with the population:
        conceptions per tick and the gestating young already holding rows both grow with it, so a
        fixed reserve that was ample at 200 animals is nothing at 20,000.

        It must clear one tick's worth of allocation, since growth happens between ticks and a tick
        that runs short conceives fewer young (`core.ecology.conception`) rather than raising —
        which would be an array quietly suppressing births, exactly the engine reaching into the
        ecology that §2.3 forbids. Measured: at the steepest growth observed, allocation ran at
        about 1% of occupancy per tick, so the default sits an order of magnitude above it.
    """

    reserve_fraction: float

    def __post_init__(self) -> None:
        if self.reserve_fraction <= 0.0:
            raise ValueError(
                f"reserve_fraction must be positive, got {self.reserve_fraction}; at zero the "
                "store only grows once it is already full, and a tick that runs out of rows "
                "conceives fewer young rather than raising — so the array would silently cap the "
                "population"
            )


def grow_if_crowded(store: EntityStore, config: GrowthConfig) -> bool:
    """Double the store if its free rows have fallen below the configured reserve.

    Returns whether it grew, so a caller that must react to a capacity change — the tick loop's
    position snapshots — can do so without comparing sizes itself.

    **Call only at a tick boundary.** `EntityStore.grow` cannot check that for itself: it has no
    notion of mid-tick, and the hazard is a live NumPy view rather than anything visible from
    inside the store.

    Occupancy rather than capacity is what the reserve is measured against, because it is the
    population that consumes rows. Measured against capacity instead, a mostly-empty store would
    keep growing after a die-off — free rows would be plentiful in absolute terms and the ratio
    would still read low.
    """
    occupied = store.capacity - store.available
    if store.available >= occupied * config.reserve_fraction:
        return False
    store.grow()
    return True
