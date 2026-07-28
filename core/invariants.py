"""Invariant harness: vectorized predicates over global arrays, checked after every tick in
debug builds (CLAUDE.md §6).

The simulation is deliberately non-deterministic (§2.2), which rules out golden-output tests.
Correctness is enforced instead by invariants that must hold after every tick regardless of what
any given run's random draws did. Each `Invariant` is a pure, vectorized function of an
`EntityStore` snapshot returning the rows that violate it; `InvariantRegistry` runs the full set
in registration order and fails loudly (§8.7) at the first violation, naming the tick, the
invariant, and the offending rows rather than letting a corrupted world run on silently.

Only the invariants that are checkable against what exists in `core/` today are registered by
`default_registry()`. CLAUDE.md §2.5's closed energy/nutrient loop needs an income/loss ledger
(sunlight, respiration, feeding) that doesn't exist yet — #17 (metabolic budget), #18 (plants and
soil nutrients), #19 (feeding) and #21 (death and decomposition) own that bookkeeping and are
still open. Fabricating a ledger here ahead of them would be exactly the "two incompatible
versions of the same abstraction" CLAUDE.md §7.1 warns against, so full flow conservation is left
for those issues to register through this same harness once their systems exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from core.entities.store import EntityStore

Predicate = Callable[[EntityStore], np.ndarray]


@dataclass(frozen=True)
class Invariant:
    """A named, vectorized predicate.

    check: given a store, returns a (k,) int64 array of the row indices that violate the
        invariant -- empty if none do. Must be a pure read of the store's current arrays; the
        harness calls it once per tick and never mutates the store on its behalf.
    """

    name: str
    check: Predicate


class InvariantViolation(Exception):
    """A registered invariant found offending rows after a tick.

    tick: the tick count at which the violation was detected.
    invariant_name: the failing Invariant's name.
    offending_rows: (k,) int64 array of the violating row indices, k >= 1.
    """

    def __init__(self, tick: int, invariant_name: str, offending_rows: np.ndarray) -> None:
        self.tick = tick
        self.invariant_name = invariant_name
        self.offending_rows = offending_rows
        super().__init__(
            f"tick {tick}: invariant '{invariant_name}' violated at rows "
            f"{offending_rows.tolist()}"
        )


class InvariantRegistry:
    """An ordered set of invariants, evaluated together after a tick.

    Registration order is check order: check_all() raises at the first violation it finds rather
    than collecting every failure, so a single corrupted tick is reported once, loudly, instead
    of a wall of downstream symptoms.
    """

    def __init__(self) -> None:
        self._invariants: list[Invariant] = []

    def register(self, name: str, check: Predicate) -> None:
        self._invariants.append(Invariant(name, check))

    def check_all(self, store: EntityStore, tick: int) -> None:
        """Evaluate every registered invariant against `store`.

        Raises InvariantViolation at the first invariant (in registration order) that reports
        offending rows.
        """
        for invariant in self._invariants:
            offending_rows = invariant.check(store)
            if offending_rows.size:
                raise InvariantViolation(tick, invariant.name, offending_rows)


def no_alive_entity_occupies_a_free_row(store: EntityStore) -> np.ndarray:
    """Rows marked `alive` that are also on the free list.

    Only EntityStore.allocate()/release() are supposed to move rows between "alive" and "free"
    (§2.3 free-list row reuse), keeping the two states complementary by construction. A future
    domain service that claims `alive` as one of its owned columns (§2.3) and writes it directly
    -- e.g. marking death without calling release() -- would desync them: the row stays live
    while the free list believes it is available, so the next allocate() silently overwrites a
    live entity. This is exactly the reachable failure mode the check guards against, not a
    condition that structurally cannot occur.
    """
    return np.flatnonzero(store.alive & store.free_row_mask())


def no_alive_entity_has_negative_energy(store: EntityStore) -> np.ndarray:
    """Alive rows whose energy has gone negative.

    §2.5's metabolic pool is a hard budget: a trait's upkeep can drive an entity's energy to
    zero, never below it. Nothing in `core/` yet spends or grants energy (the full sunlight ->
    upkeep -> death ledger is #17-#19, #21, still open), but this half of the invariant --
    energy never goes negative -- holds regardless of which system does the spending, so it is
    checkable now and stays true once those systems land.
    """
    return np.flatnonzero(store.alive & (store.energy < 0))


def no_entity_leaves_world_bounds(
    min_x: float, max_x: float, min_y: float, max_y: float
) -> Predicate:
    """Build a predicate flagging alive rows whose (x, y) falls outside the given world bounds.

    Bounds are supplied by the caller (typically a Terrain's world_width/world_height, CLAUDE.md
    §2.6) rather than read from global state, so the same predicate is testable against an
    arbitrary rectangle without constructing a world.
    """

    def check(store: EntityStore) -> np.ndarray:
        out_of_bounds = (
            (store.x < min_x) | (store.x > max_x) | (store.y < min_y) | (store.y > max_y)
        )
        return np.flatnonzero(store.alive & out_of_bounds)

    return check


def default_registry(min_x: float, max_x: float, min_y: float, max_y: float) -> InvariantRegistry:
    """The invariants checkable against `core/` as it exists today, over a world of the given
    bounds.
    """
    registry = InvariantRegistry()
    registry.register("no_alive_entity_occupies_a_free_row", no_alive_entity_occupies_a_free_row)
    registry.register("no_alive_entity_has_negative_energy", no_alive_entity_has_negative_energy)
    registry.register(
        "no_entity_leaves_world_bounds", no_entity_leaves_world_bounds(min_x, max_x, min_y, max_y)
    )
    return registry
