"""Ecology domain service: the metabolic pool and the upkeep drawn against it (CLAUDE.md §2.5,
issue #17).

Owns the `energy` column on the shared entity store — the single pool every trait charges against.
This module only ever *removes* energy. Income is somebody else's: sunlight into plants (#18) and
transfers through feeding (#19). Keeping every withdrawal here and the income there is what makes
the closed loop of §2.5 auditable at all, and it is why `spend()` is the only method that writes:
metabolic upkeep goes through it, and so does the locomotion bill `core.behaviour.movement` (#25)
hands over, since a service that does not own `energy` cannot subtract from it itself.

Death is not here either. An entity whose pool reaches zero is *starving*, exposed as a Selection
for #21 to turn into carcasses and decomposition. Splitting it this way keeps the two decisions —
"has run out of energy" and "is therefore dead" — owned by the issues that can actually justify
them, rather than having a metabolism module quietly decide mortality.
"""

from __future__ import annotations

import numpy as np

from core.ecology.metabolism import Metabolism
from core.entities.store import EntityStore
from core.genetics.service import Genetics
from core.selection import Selection
from core.services import ColumnRegistry, DomainService
from core.world.climate import Climate


class Ecology(DomainService):
    """Charges every entity's expressed phenotype against its energy pool, once per tick.

    genetics: consulted for expressed phenotype only — this service never writes a gene. Read
        across the boundary through `Genetics.expressed`, so the species expression mask is
        applied by the service that owns it (CLAUDE.md §2.3).
    climate: the temperature field thermoregulation cost is sampled from, at each entity's own
        position. Queried as a continuous field, never as a zone label (`core.world.climate`).
    metabolism: the per-world cost table (`core.ecology.metabolism`).
    """

    owns = ("energy",)

    # Narrows DomainService.store (typed `object`, since the base is store-shape-agnostic) to the
    # concrete EntityStore whose `energy`, `x`, `y` and `alive` columns this service reads.
    store: EntityStore

    def __init__(
        self,
        store: EntityStore,
        registry: ColumnRegistry,
        genetics: Genetics,
        climate: Climate,
        metabolism: Metabolism,
    ) -> None:
        super().__init__(store, registry)
        self.genetics = genetics
        self.climate = climate
        self.metabolism = metabolism

    def energy(self, selection: Selection) -> np.ndarray:
        """(len(selection),) float32, joules: the current metabolic pool, in ascending row order."""
        return self.store.energy[selection.to_mask()]

    def upkeep(self, selection: Selection) -> np.ndarray:
        """(len(selection),) float32, joules per tick: what one tick will cost each entity.

        Exposed separately from `drain` because it is a pure read: the diagnostic viewer's energy
        overlay and any future intervention that asks "what does this animal cost to keep" need
        the number without spending it.
        """
        mask = selection.to_mask()
        temperature = self.climate.temperature_at(self.store.x[mask], self.store.y[mask])
        return self.metabolism.upkeep(self.genetics.expressed(selection), temperature)

    def drain(self, selection: Selection) -> None:
        """Charge one tick of upkeep to `selection`, flooring the pool at zero.

        `selection` is the caller's choice of who metabolises; pass the live entities. Nothing
        here filters, because a tick loop draining anything other than the living is a bug in the
        loop, not a condition to absorb silently (§8.7).
        """
        self.spend(selection, self.upkeep(selection))

    def spend(self, selection: Selection, joules: np.ndarray) -> None:
        """Charge `joules` to `selection`'s pools, flooring each at zero.

        The floor is what makes the pool a hard budget rather than a debt (CLAUDE.md §2.5): an
        entity can be emptied, never overdrawn, which is the invariant #7 asserts every tick. The
        shortfall is not carried forward — an animal that could not pay is already starving, and
        how badly it failed to pay changes nothing downstream.

        Upkeep is not the only draw on the pool. §2.5's "effort is charged, not just distance"
        makes locomotion a second one (`core.behaviour.movement`, #25), and this service owns
        `energy`, so a mover cannot write the column itself and hands the bill here instead.
        Exposed as a general charge rather than as a `charge_movement` method because nothing
        about a floored subtraction is specific to what the energy went on, and #19's chase and
        #20's gestation are the same operation again.

        Raises ValueError for a negative charge: energy entering the world is #18's sunlight and
        #19's feeding, never a cost with its sign flipped, and §2.5's closed loop has no other
        income (§8.7).
        """
        joules = np.asarray(joules, dtype=np.float32)
        if np.any(joules < 0.0):
            raise ValueError(
                "spend() charges energy and cannot be negative; income belongs to #18 and #19"
            )
        self.write("energy", selection, np.maximum(self.energy(selection) - joules, 0.0))

    def starving(self, selection: Selection) -> Selection:
        """The entities in `selection` whose pool has run out — energy at or below zero.

        Restricted to live rows: a released row's `energy` still reads whatever it held, and #21
        will turn this selection into deaths, so a free row appearing in it would "kill" an entity
        that does not exist.
        """
        return Selection.from_mask(
            selection.to_mask() & self.store.alive & (self.store.energy <= 0.0)
        )
