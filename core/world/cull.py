"""Culling: removing animals of one species, and the rule it must obey (#26, #28).

§2.7 permits culling and forbids one thing outright — **never total eradication of a species by a
single action**. #26 built this to prove its framework; #28 settled the four things a player-facing
cull needs (the owner's decisions on #28): what the guard promises, who sets it, which animals are
taken, and where their bodies go.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.ecology.carrion import Carrion
from core.ecology.service import Ecology
from core.entities.store import EntityStore
from core.selection import Selection


@dataclass(frozen=True)
class CullConfig:
    """The guard every cull in a world is held to.

    survivors: how many born animals of a species a cull must leave. **A world rule, never a field
        of the request** (#28, decision 2): a client choosing its own floor could send 1, and a
        standing policy (#33) firing during an absence must be held to the number a click is. It is
        fingerprinted with every other rule, so a world keeps its guard for life (§2.8). No default:
        it is an absolute count, and what it promises is only true of the world it was measured on
        (§2.6's rule for absolute quantities).

        **At least two**, because reproduction needs two parents: a guard leaving one honours
        §2.7's letter and eradicates the species on a delay. What the number should be above that is
        a measurement of the world, not a choice; the demo world's is on #28.
    """

    survivors: int

    def __post_init__(self) -> None:
        if self.survivors < 2:
            raise ValueError(
                f"survivors must be at least 2, got {self.survivors}: reproduction needs two "
                "parents, so a cull leaving fewer has eradicated the species on a delay (§2.7)"
            )


class Cull:
    """Remove `count` born animals of one species, at random, leaving their bodies as carrion.

    ecology: whose pools the bodies are taken out of. carrion: the field they are laid on. Both are
        bound at construction rather than passed, per `Intervention`.
    config: the world's guard (`CullConfig`).
    species_id: which species to thin. Per-species because §2.7's rule is: removing the last of a
        *species* is the forbidden act, and a world with three species has three separate limits
        rather than one population floor.
    count: how many to remove.
    rng: the world's own generator, so a reloaded world draws onward rather than replaying (#31).

    **Victims are drawn at random from the species** (#28, decision 3). The first build took the
    first `count` rows, which nobody chose: row order follows allocation and free-list reuse, so a
    cull fell hardest on whichever animals got their rows first — a selection rule hidden in an
    implementation detail. A random draw removes numbers and selects for nothing, so a cull never
    quietly evolves the species it was used on. Thinning one place rather than one species is a
    different intervention, to be added beside this one rather than in it.

    **The bodies become carrion where each animal stood** (#28, decision 4), exactly as a strike's
    wound does (#179). A cull is a death that is not starvation and leaves a body distinct from its
    fuel, which is what `Death` says carrion is for. The first build returned each body to the soil
    at once, because carrion did not exist yet. It now reaches the soil as every other carcass does,
    through `Carrion.decompose`, after whatever scavenges it has eaten. So a cull still does not
    sterilise the ground it was used on (#26's point), and it briefly feeds what eats meat.
    """

    name = "cull"

    def __init__(
        self,
        ecology: Ecology,
        carrion: Carrion,
        store: EntityStore,
        config: CullConfig,
        species_id: int,
        count: int,
        rng: np.random.Generator,
    ) -> None:
        if count < 1:
            raise ValueError(f"a cull must remove at least one animal, got {count}")
        self.ecology = ecology
        self.carrion = carrion
        self.store = store
        self.config = config
        self.species_id = species_id
        self.count = count
        self.rng = rng

    def cost(self) -> float:
        """One unit per animal. A placeholder rate and openly so: it is tuned as a table with the
        fence's price and the world's income in the playtest (#292 phase 5), not settled ahead of
        it (#28)."""
        return float(self.count)

    def refusal(self):
        """§2.7's rule, checked against the world as it is at the boundary rather than as it was
        when the player clicked — the species may have starved in between."""
        alive = len(self._living())
        if alive == 0:
            return f"species {self.species_id} has no living members"
        left = alive - self.count
        if left < self.config.survivors:
            return (
                f"culling {self.count} would leave {max(left, 0)} of species {self.species_id}; "
                f"this world keeps at least {self.config.survivors} so a species can always regrow "
                "(§2.7 forbids eradicating a species by a single action)"
            )
        return None

    def apply(self, store: EntityStore) -> None:
        """Lay each victim's body where it stands, then release it."""
        rows = self._victims()
        # The whole pool, at the pool's own float64 (#240), taken out by `kill` and laid by
        # `deposit` in the same amount: the flesh stays owed on the export ledger while it lies on
        # the ground, and `Carrion.decompose` pays it back (#214's invariant). Read before
        # `release`, which clears the id mapping these rows are addressed through.
        bodies = self.ecology.kill(rows, store.energy[rows].astype(np.float64))
        self.carrion.deposit(store.x[rows], store.y[rows], bodies)
        store.release(store.row_ids()[rows])

    def _victims(self) -> np.ndarray:
        """`count` born members of the species, each equally likely, as `(count,) int64` rows."""
        return self.rng.choice(self._living().to_indices(), size=self.count, replace=False)

    def _living(self) -> Selection:
        """This species' born, living members — the population §2.7's rule is about.

        Gestating rows are excluded: one carries a negative age and has not been born (#20), so
        counting it as a survivor would let a cull leave a species whose only remaining members are
        unborn. Nor is one culled: a pregnancy is born whether or not its mother survives (§2.7).
        """
        return Selection.from_mask(
            self.store.alive
            & (self.store.age >= 0)
            & (self.store.species_id == self.species_id)
        )
