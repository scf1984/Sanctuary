"""Death: an animal that has run out of energy stops being one (CLAUDE.md §2.5, issue #21).

`Ecology.starving` has existed since #17 with nothing consuming it — the two decisions, "has run
out of energy" and "is therefore dead", were deliberately split so that a metabolism module would
not quietly decide mortality. This is the other half, and #19 is what finally made the first one
reachable: before feeding existed every animal drained at the same rate and starvation said nothing
about an individual.

**There is no separate cause of death, and §2.5 says there should not be.** No age check, no
lifespan gene: living longer would be pure benefit and every lineage would evolve toward
immortality. Senescence instead degrades performance with age, so an old animal catches less and
escapes less until it cannot cover its own upkeep — and then it arrives *here*, through the path
that already exists. Predation (#179) and crowding are the same: they empty a pool, and emptying a
pool is what this module answers.

**There is no carcass, and that is arithmetic rather than an omission.** An animal's nutrient debt
is exactly its energy: founding records `E₀` on the export ledger, feeding adds what it assimilated,
and every `Ecology.spend` returns what it burned. So a starved animal owes nothing and leaves
nothing — it has metabolised its own body, which is what starving to death is. Carrion needs a
death that is *not* starvation and a body distinct from its fuel, and the second of those is #20's
gestation. Filed rather than built, because a carrion field nothing can put mass into is a mechanic
in name only (§8.2).

**Rows are freed, not marked.** `EntityStore.release` returns them to the free list, and §2.1 runs
death before reproduction precisely so that a world at capacity can still breed — the dead have
already made room by the time anything is born.
"""

from __future__ import annotations

import numpy as np

from core.ecology.service import Ecology
from core.entities.store import EntityStore
from core.selection import Selection


class Death:
    """Turns emptied energy pools into freed rows, once per tick.

    Owns no store column. `energy` is `Ecology`'s and the row bookkeeping is the store's, so this
    service decides only *who* dies — which is the one judgement neither of them should be making.
    """

    def __init__(self, store: EntityStore, ecology: Ecology) -> None:
        self.store = store
        self.ecology = ecology

    def reap(self, selection: Selection) -> None:
        """Release every entity in `selection` whose energy pool has run out.

        `selection` is the caller's choice of who is subject to death; pass the living. Nothing
        here filters beyond that, for the same reason `Ecology.drain` does not — a tick loop
        reaping anything other than the living is a bug in the loop rather than a condition to
        absorb quietly (§8.7). `Ecology.starving` already restricts itself to live rows, so a row
        freed earlier in the same tick cannot be released twice.
        """
        dying = self.ecology.starving(selection)
        if not len(dying):
            return

        # Ids, not rows: `release` addresses entities by their stable id, and reading them before
        # the release is the only order that works — it clears the mapping this depends on.
        ids = self.store.row_ids()[dying.to_mask()]
        self.store.release(np.asarray(ids, dtype=np.int64))
