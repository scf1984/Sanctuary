"""Invariant harness: vectorized predicates over global arrays, checked after every tick in
debug builds (CLAUDE.md §6).

The simulation is deliberately non-deterministic (§2.2), which rules out golden-output tests.
Correctness is enforced instead by invariants that must hold after every tick regardless of what
any given run's random draws did. `InvariantRegistry` runs the full set in registration order and
fails loudly (§8.7) at the first violation, naming the tick, the invariant, and what the invariant
found rather than letting a corrupted world run on silently.

**A check reports a `Violation`, not row indices** (#91). The harness's original contract returned
a `(k,)` array of offending rows, which silently assumed everything worth checking is an entity.
It is not: §6 lists "total nutrients are conserved" among the per-tick invariants, and the nutrient
pool lives on the plant field's grid cells (`core.ecology.plants`), which has no rows to report.
A check therefore returns `None` when it holds and a `Violation` when it does not, describing the
breach in its own terms — rows for the invariants that have rows, a nutrient shortfall for the one
that does not. The harness never interprets `Violation.detail`; it only reports it.

**Fields reach a check by closure, not by a world-context argument.** `nutrients_are_conserved`
binds the `Plants` instance when the invariant is built, exactly as `no_entity_leaves_world_bounds`
binds its rectangle. That keeps one predicate signature for every invariant and avoids a context
object listing every domain a check might one day read — most of which do not exist yet (§8.2).
It also already covers the cross-domain case #21 brings: a check that must weigh carcasses on the
store against nutrients in the soil closes over the field and receives the store.

Only the invariants that are checkable against what exists in `core/` today are registered by
`default_registry()`. §2.5's closed energy loop still needs an income/loss ledger for *energy*:
#17 landed the loss half as a pure drain (`core.ecology.service.Ecology`) and #18 the income half,
but #19 (feeding) and #21 (death and decomposition) own the transfer and return sides and are open,
so full energy-flow conservation is left for them. The nutrient half is closed today and is
registered whenever a world supplies a plant field.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from core.behaviour.movement import Movement
from core.ecology.plants import Plants
from core.entities.store import EntityStore
from core.selection import Selection


@dataclass(frozen=True)
class Violation:
    """What a failing invariant found, in that invariant's own terms.

    detail: human-readable description of the breach, used verbatim in the raised error. The
        harness never parses it, so each invariant is free to report whatever identifies the
        problem — offending rows, a conservation shortfall, a corrupt cell count.
    rows: (k,) int64 store rows that violate the invariant, empty for invariants over things that
        have no rows (the plant field's cells). Kept structured rather than folded into `detail`
        because an entity invariant's rows are what a debugger goes on to inspect.
    """

    detail: str
    rows: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))


Predicate = Callable[[EntityStore], Optional[Violation]]


@dataclass(frozen=True)
class Invariant:
    """A named, vectorized predicate.

    check: given a store, returns None if the invariant holds and a Violation if it does not.
        Must be a pure read of the state it inspects; the harness calls it once per tick and never
        mutates anything on its behalf.
    """

    name: str
    check: Predicate


class InvariantViolation(Exception):
    """A registered invariant failed after a tick.

    tick: the tick count at which the violation was detected.
    invariant_name: the failing Invariant's name.
    violation: the Violation the check reported.
    """

    def __init__(self, tick: int, invariant_name: str, violation: Violation) -> None:
        self.tick = tick
        self.invariant_name = invariant_name
        self.violation = violation
        super().__init__(f"tick {tick}: invariant '{invariant_name}' violated -- {violation.detail}")


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
        """Evaluate every registered invariant against `store` and whatever each closes over.

        Raises InvariantViolation at the first invariant (in registration order) that reports one.
        """
        for invariant in self._invariants:
            violation = invariant.check(store)
            if violation is not None:
                raise InvariantViolation(tick, invariant.name, violation)


def _rows_violation(rows: np.ndarray, what: str) -> Optional[Violation]:
    """None if `rows` is empty, else a Violation naming `what` and listing the rows."""
    if not rows.size:
        return None
    return Violation(f"{what} at rows {rows.tolist()}", rows)


def no_alive_entity_occupies_a_free_row(store: EntityStore) -> Optional[Violation]:
    """Rows marked `alive` that are also on the free list.

    Only EntityStore.allocate()/release() are supposed to move rows between "alive" and "free"
    (§2.3 free-list row reuse), keeping the two states complementary by construction. A future
    domain service that claims `alive` as one of its owned columns (§2.3) and writes it directly
    -- e.g. marking death without calling release() -- would desync them: the row stays live
    while the free list believes it is available, so the next allocate() silently overwrites a
    live entity. This is exactly the reachable failure mode the check guards against, not a
    condition that structurally cannot occur.
    """
    return _rows_violation(
        np.flatnonzero(store.alive & store.free_row_mask()), "alive entities on the free list"
    )


def no_alive_entity_has_negative_energy(store: EntityStore) -> Optional[Violation]:
    """Alive rows whose energy has gone negative.

    §2.5's metabolic pool is a hard budget: a trait's upkeep can drive an entity's energy to
    zero, never below it. Only #17's upkeep drain spends the pool so far -- #18's sunlight income
    reaches plants, not animals, and the transfer into an animal is #19's feeding, still open --
    but this half of the invariant, that energy never goes negative, holds regardless of which
    system does the spending, so it is checkable now and stays true once those systems land.
    """
    return _rows_violation(
        np.flatnonzero(store.alive & (store.energy < 0)), "alive entities with negative energy"
    )


def no_entity_leaves_world_bounds(
    min_x: float, max_x: float, min_y: float, max_y: float
) -> Predicate:
    """Build a predicate flagging alive rows whose (x, y) falls outside the given world bounds.

    Bounds are supplied by the caller (typically a Terrain's world_width/world_height, CLAUDE.md
    §2.6) rather than read from global state, so the same predicate is testable against an
    arbitrary rectangle without constructing a world.
    """

    def check(store: EntityStore) -> Optional[Violation]:
        out_of_bounds = (
            (store.x < min_x) | (store.x > max_x) | (store.y < min_y) | (store.y > max_y)
        )
        return _rows_violation(
            np.flatnonzero(store.alive & out_of_bounds), "alive entities outside world bounds"
        )

    return check


def no_alive_entity_is_more_than_dry(store: EntityStore) -> Optional[Violation]:
    """Flag alive rows whose `dehydration` has left [0, 1] (#156).

    The column is a *fraction* of a reserve, so both ends are physical rather than conventional: an
    animal cannot lose more water than it has, and it cannot hold more than full. It is asserted
    rather than clamped in `Hydration` because the bound is what makes the upkeep penalty bounded —
    past 1 a dry animal would be charged arbitrarily rather than merely fatally, and below 0 a
    *drink* would make an animal cheaper to run than a watered one, which is a free lunch reached
    by standing in a lake.

    Needs no closure: unlike the world bounds or the nutrient ledger, the range is a property of
    what the column *is* and not of any world's configuration, so there is nothing to bind.
    """
    living = Selection.from_mask(store.alive)
    if not len(living):
        return None
    mask = living.to_mask()
    deficit = store.dehydration[mask]
    outside = (deficit < 0.0) | (deficit > 1.0)
    return _rows_violation(
        np.flatnonzero(mask)[outside], "alive entities whose dehydration left [0, 1]"
    )


def no_entity_exceeds_its_top_speed(movement: Movement) -> Predicate:
    """Build a predicate flagging alive rows travelling faster than their expressed `speed` gene.

    §2.5 rejected a speed cap outright — top speed is a gene under selection, and an authored
    ceiling on an evolving trait is what "author the physics, not the outcomes" forbids. What
    replaced it is an *argument*: `Movement.step` writes velocity from the displacement that
    actually happened, and that displacement is never longer than the velocity it aimed for, which
    is on the segment between last tick's velocity and `top_speed × pace`. So the bound holds by
    induction from a standing start, with no clamp anywhere.

    This is that argument asserted rather than believed, which is exactly what §8.2 asks for when
    something genuinely cannot occur: a branch in the hot loop would be a defensive check against
    an impossible condition, and an invariant is where an impossible condition belongs. It would
    catch a future writer of `velocity_x` that set the column from an intention instead of an
    outcome — the one mistake the induction depends on nobody making.

    `relative_slack` covers the float32 round-trip on the velocity columns against the float64
    phenotype the bound is computed in, and nothing else. It is far below any real breach, which
    would be a whole step rather than a rounding of one.
    """
    relative_slack = 1e-5

    def check(store: EntityStore) -> Optional[Violation]:
        living = Selection.from_mask(store.alive)
        if not len(living):
            return None
        mask = living.to_mask()
        speed = np.hypot(
            store.velocity_x[mask].astype(np.float64), store.velocity_y[mask].astype(np.float64)
        )
        over = speed > movement.top_speed(living) * (1.0 + relative_slack)
        return _rows_violation(
            np.flatnonzero(mask)[over], "alive entities travelling above their top speed"
        )

    return check


def nutrients_are_conserved(plants: Plants, relative_tolerance: float = 1e-9) -> Predicate:
    """Build a predicate asserting the plant field's nutrient total never moves (§2.5, §6).

    plants: the field to watch. Its total at build time is the reference, so build the invariant
        after the world is seeded -- every later tick is compared against that opening total.
    relative_tolerance: fractional drift allowed against the opening total. The default matches
        what `tests/core/ecology/test_plants.py` measured over hundreds of ticks of growth,
        senescence and grazing (`rel=1e-9` holds there): float64's ~15 significant digits leave
        roughly six of headroom above the rounding of summing the grid every tick, while any real
        leak -- a system creating or destroying nutrients -- is orders of magnitude larger than
        that and still trips. Tightening it towards float64 epsilon would trip on arithmetic
        rather than on a bug, which is the failure the field is stored as float64 to avoid.

    This is the invariant #91 exists for: it reports a scalar drift over grid cells and has no
    offending rows to name, which is why a check now returns a Violation rather than a row array.
    """
    opening_total = plants.total_nutrients()

    def check(store: EntityStore) -> Optional[Violation]:
        total = plants.total_nutrients()
        if math.isclose(total, opening_total, rel_tol=relative_tolerance):
            return None
        return Violation(
            f"plant field holds {total!r} nutrient units against an opening total of "
            f"{opening_total!r} (drift {total - opening_total!r})"
        )

    return check


def default_registry(
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    plants: Optional[Plants] = None,
    movement: Optional[Movement] = None,
) -> InvariantRegistry:
    """The invariants checkable against `core/` as it exists today, over a world of the given
    bounds.

    plants: the world's plant field, if it has one. Supplying it adds nutrient conservation to the
        registry. It is optional because the entity invariants are checkable against a bare store
        and constructing a field needs a whole terrain, climate and water stack; a world that has
        plants and omits them here simply stops checking a conservation law it could have checked,
        which is why the parameter exists rather than the registry reaching for a global.
    movement: the world's movement service, if it has one. Supplying it adds the top-speed bound,
        which needs a phenotype and therefore a genetics stack — optional on the same terms as
        `plants`, and for the same reason.
    """
    registry = InvariantRegistry()
    registry.register("no_alive_entity_occupies_a_free_row", no_alive_entity_occupies_a_free_row)
    registry.register("no_alive_entity_has_negative_energy", no_alive_entity_has_negative_energy)
    registry.register(
        "no_alive_entity_is_more_than_dry", no_alive_entity_is_more_than_dry
    )
    registry.register(
        "no_entity_leaves_world_bounds", no_entity_leaves_world_bounds(min_x, max_x, min_y, max_y)
    )
    if plants is not None:
        registry.register("nutrients_are_conserved", nutrients_are_conserved(plants))
    if movement is not None:
        registry.register(
            "no_entity_exceeds_its_top_speed", no_entity_exceeds_its_top_speed(movement)
        )
    return registry
