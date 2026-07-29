"""Aging: the one writer of `age`, advancing it by whole ticks lived (CLAUDE.md §2.1, issue #109).

`age` was allocated, zeroed on reuse, and read — by `Lust`'s maturity gate — but never written by
anything, so every entity in an assembled world stayed newborn forever and one of the five authored
drives could not fire. Everything life-history rests on this column advancing: §2.5's senescence
decays performance traits as ``exp(-rate x age)`` at expression time, and death then falls out of
starvation rather than from an age check.

**A service rather than the store advancing its own column.** `EntityStore` is storage and
deliberately not a simulation participant — it has no notion of a tick to count, and CLAUDE.md §2.3
keeps per-tick work in services that own their columns. Being a registered system instead means
``advance(1000)`` and a thousand ``advance(1)`` calls age the world identically, which §2.4 requires
of every system: the wake schedule decides when compute happens, never how fast the world moves.

**It runs late in the tick, and before reproduction** (§2.1's system order). Incrementing after
births would hand a newborn an age of one having lived no whole tick, which is exactly the
off-by-one that a maturity gate and a senescence curve both read.

It lives in `core.ecology` because aging is life-history: maturity (#20), senescence and death
(#21) are the things that consult it, and §4 puts reproduction and decomposition here.
"""

from __future__ import annotations

from core.entities.store import EntityStore
from core.selection import Selection
from core.services import DomainService


class Aging(DomainService):
    """Owns the `age` column: how many whole ticks each entity has lived.

    The tick counter is the only clock (§2.1), so this service holds no rate, no interval and no
    configuration — one tick of the world is one tick of every life in it, and a world that ages
    faster is a world ticked more often.
    """

    owns = ("age",)

    # Narrows DomainService.store (typed `object`, the base being store-shape-agnostic) to the
    # concrete EntityStore whose `age` column this service advances.
    store: EntityStore

    def advance(self, selection: Selection) -> None:
        """Record one whole tick of living against every entity in `selection`.

        `selection` is the caller's choice of who ages, as `Ecology.drain`'s is: the tick loop
        passes the living, and nothing here filters to `alive`, because a loop aging anything else
        is a bug in the loop rather than a condition to absorb silently (§8.7). A row on the free
        list carries no cost either way — `allocate()` resets `age` to 0 when it hands the row out
        again, so an entity can never inherit its predecessor's years.
        """
        self.write("age", selection, self.store.age[selection.to_mask()] + 1)
