"""Culling: the one intervention #26 builds to prove its framework, and the rule it must obey.

§2.7 permits culling and forbids one thing outright — **never total eradication of a species by a
single action**. That is what makes this a good first intervention rather than an arbitrary one: it
exercises all three parts of the contract, and its precondition is a real design rule rather than a
placeholder.

The catalogue is not here. #27's fence and #28's siblings own their own effects; what this file
owns is the demonstration that an effect can be recorded, costed, refused and applied.
"""

from __future__ import annotations

import numpy as np

from core.ecology.service import Ecology
from core.entities.store import EntityStore
from core.selection import Selection


class Cull:
    """Remove `count` animals of one species, returning their bodies to the soil.

    ecology: consulted for the energy each culled animal still holds, and for the plant field that
        energy goes back into. Bound at construction rather than passed, per `Intervention`.
    species_id: which species to thin. The precondition is per-species because §2.7's rule is:
        removing the last of a *species* is the forbidden act, and a world with three species has
        three separate limits rather than one population floor.
    count: how many to remove.
    survivors: how many of that species must remain. **A number, not zero**, because "not total
        eradication" is not the same as "leave one alive": a population of one is extinct on a
        delay, and a rule that permitted it would be honouring §2.7's letter while breaking it.
        Per-world rather than a constant here, for the same reason every other threshold is (§2.1).

    **The bodies are returned to the field, and no invariant would have made us.** That is worth
    stating precisely, because the obvious justification is wrong and was checked: releasing ten
    animals holding 1,950 energy units without returning it leaves `total_nutrients()` at exactly
    2,570,800 either way. The ledger counts their energy as *exported* — grazed out of the field
    and not yet returned — so conservation holds over a world in which the nutrient is stranded
    where nothing can ever reach it. `nutrients_are_conserved` cannot see the difference between a
    closed loop and a slow leak into that ledger, which is filed separately (§7.4).

    So the argument is ecological rather than arithmetic: an animal's body decomposes where it
    falls, and a cull is decomposition compressed into one tick. Returning it at each animal's own
    position is one call to the method `Ecology.spend` already excretes through, so it needed no
    new mechanism — and a cull that skipped it would quietly sterilise the ground it was used on,
    which is the opposite of what a steward asked for.
    """

    name = "cull"

    def __init__(
        self,
        ecology: Ecology,
        store: EntityStore,
        species_id: int,
        count: int,
        survivors: int,
    ) -> None:
        if count < 1:
            raise ValueError(f"a cull must remove at least one animal, got {count}")
        if survivors < 1:
            raise ValueError(
                f"survivors must be at least 1, got {survivors}; a cull that may empty a species "
                "is the eradication §2.7 forbids"
            )
        self.ecology = ecology
        self.store = store
        self.species_id = species_id
        self.count = count
        self.survivors = survivors

    def cost(self) -> float:
        """One unit per animal. A placeholder rate and openly so: what an intervention *should*
        cost is §5's open question, and this issue was told to build the framework and leave the
        catalogue to #27 and #28. What matters here is that a cost is charged and reconciled, not
        what it is."""
        return float(self.count)

    def refusal(self):
        """§2.7's rule, checked against the world as it is at the boundary rather than as it was
        when the player clicked — the species may have starved in between."""
        alive = len(self._living())
        if alive == 0:
            return f"species {self.species_id} has no living members"
        if alive - self.count < self.survivors:
            return (
                f"culling {self.count} would leave {alive - self.count} of species "
                f"{self.species_id}, below the {self.survivors} this world requires; §2.7 forbids "
                "eradicating a species by a single action"
            )
        return None

    def apply(self, store: EntityStore) -> None:
        """Return each animal's remaining energy to the ground it stands on, then release it."""
        victims = Selection.from_indices(
            self._living().to_indices()[: self.count], store.capacity
        )
        mask = victims.to_mask()
        # The pool goes back into the field before the rows are freed: `release` clears the id
        # mapping these positions are read through, and reading after it is the ordering bug
        # `Death.reap` documents for exactly the same reason.
        self.ecology.plants.return_nutrients(
            store.x[mask].astype(np.float64),
            store.y[mask].astype(np.float64),
            self.ecology.energy(victims).astype(np.float64),
        )
        store.release(store.row_ids()[mask])

    def _living(self) -> Selection:
        """This species' born, living members — the population §2.7's rule is about.

        Gestating rows are excluded: one carries a negative age and has not been born (#20), so
        counting it as a survivor would let a cull leave a species whose only remaining members
        are unborn.
        """
        return Selection.from_mask(
            self.store.alive
            & (self.store.age >= 0)
            & (self.store.species_id == self.species_id)
        )
