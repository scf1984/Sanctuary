"""Movement: straight-line integration toward a target, priced against the metabolic pool
(CLAUDE.md §2.5, §2.6, issue #25).

This is where a decision becomes a consequence. `core.behaviour.service.Behaviour` decides what an
animal wants and `driven_by` hands over who is on the move; this module is what actually spends
energy to get there, and what makes terrain matter.

Three properties carry the design:

- **Elevation prices travel.** Climbing charges against the same pool everything else charges
  against, so a ridge is expensive to cross rather than merely slow. That is what turns §2.6's
  heightmap from scenery into the isolation barrier speciation needs (#16) — nobody places a
  barrier, and a mountain range becomes one because crossing it costs more than the far side is
  worth.
- **Effort is charged, not just distance** (§2.5). Cost per metre rises with `pace`, so a sprint is
  dearer than a stroll over the same ground. Pricing distance alone would make a chase merely long;
  it is the per-metre premium that makes a predator pay for every chase it loses and prey pay for
  every escape. Nothing here knows what fleeing *is* — a drive that wants urgency passes a higher
  pace, and #19's chase and #24's flight are then priced without this module changing.
- **The pool gates the step, it does not merely record it.** An animal that cannot pay for the
  whole step covers only what it can afford, and an empty one does not move at all. This is §2.5's
  "a starving animal can neither run nor hide" as a mechanism rather than as a mood: hunger closes
  off options instead of reading high.

**Movement never goes through an angle.** The prototype's `Vector.angle` computed ``atan2(x, y)``
— arguments reversed from the standard ``atan2(y, x)`` — which mirrored every direction about the
45-degree line. Integration needs a unit vector, and a unit vector is a division, so the whole
class of bug is absent rather than avoided.

Coefficients are per-world configuration, never constants in this module (CLAUDE.md §2.1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.ecology.service import Ecology
from core.entities.store import EntityStore
from core.genetics.service import Genetics
from core.genetics.vocabulary import GeneVocabulary
from core.selection import Selection
from core.services import ColumnRegistry, DomainService
from core.world.terrain import Terrain


@dataclass(frozen=True)
class MovementConfig:
    """Per-world locomotion cost table.

    speed_gene: the gene whose expressed value is top speed, in **world units per tick** — the
        tick being the only clock (§2.1), so a step is one tick's worth by construction and there
        is no timestep to pass in. Named here rather than assumed, as
        `MetabolismConfig.insulation_gene` is, because the vocabulary is per-world.
    size_gene: the gene whose expressed value scales every cost term below. A bigger body is more
        expensive to haul over the same ground and up the same hill, which is the counterweight
        that stops size running away on the benefits it buys elsewhere.
    transport_cost: joules per world unit travelled, per unit of expressed size, at zero pace.
        Must be positive: at zero, distance is free and nothing stops an animal crossing the world
        every tick, which removes the cost half of §2.5's hard budget for the one trait that
        spends the most.
    exertion_premium: extra fraction of `transport_cost` charged at full pace, so a metre at pace
        ``p`` costs ``transport_cost × (1 + exertion_premium × p)``. Must be non-negative —
        negative would make sprinting cheaper per metre than walking, inverting §2.5.
    climb_cost: joules per metre of elevation *gained*, per unit of expressed size. Descent
        charges nothing beyond its horizontal distance: raising a body against gravity is work in
        a way that lowering it is not, and that asymmetry is the whole of "uphill costs more than
        downhill". Must be non-negative; negative would mint energy out of walking uphill, which
        §2.5's closed loop forbids outright.
    walking_pace: the fraction of top speed an unhurried animal uses. Config rather than a literal
        at the call site because it is one half of the walk/sprint ratio `exertion_premium` prices,
        and tuning either one alone is what §2.1 means by constants drifting apart. Must be in
        (0, 1]: zero would mean an animal that never travels, and above one would let a pace buy
        speed the metabolic budget never charged for.
    """

    speed_gene: str
    size_gene: str
    transport_cost: float
    exertion_premium: float
    climb_cost: float
    walking_pace: float

    def __post_init__(self) -> None:
        if self.transport_cost <= 0:
            raise ValueError(
                f"transport_cost must be positive, got {self.transport_cost}; see the config "
                "docstring — free distance removes the cost half of the energy budget"
            )
        if self.exertion_premium < 0:
            raise ValueError(
                f"exertion_premium must be non-negative, got {self.exertion_premium}; "
                "negative makes sprinting cheaper per metre than walking"
            )
        if self.climb_cost < 0:
            raise ValueError(
                f"climb_cost must be non-negative, got {self.climb_cost}; "
                "negative mints energy out of walking uphill"
            )
        if not 0.0 < self.walking_pace <= 1.0:
            raise ValueError(f"walking_pace must be in (0, 1], got {self.walking_pace}")


class Movement(DomainService):
    """Owns the position columns: where every entity is, and what it cost to get there.

    Surface-locked (§2.6): ``z`` is the terrain elevation under ``(x, y)``, written on every step
    and by `settle`. The column is stored rather than derived on read because the spatial index and
    the viewer both consume it, and because free flight (§2.6's staged plan) unlocks it later
    without changing anyone's read.

    ecology: the owner of `energy` (#17). Every locomotion charge goes through `Ecology.spend`,
        because this service does not own that column and must not subtract from it directly
        (CLAUDE.md §2.3).
    genetics: consulted for expressed phenotype only — this service never writes a gene. Speed and
        size are read through `expressed`, so a species that does not express speed does not move,
        exactly as it does not pay for speed.
    terrain: the height field, sampled at both ends of every step to price the climb.
    """

    owns = ("x", "y", "z")

    # Narrows DomainService.store (typed `object`, the base being store-shape-agnostic) to the
    # concrete EntityStore whose position columns this service writes.
    store: EntityStore

    def __init__(
        self,
        store: EntityStore,
        registry: ColumnRegistry,
        ecology: Ecology,
        genetics: Genetics,
        terrain: Terrain,
        vocabulary: GeneVocabulary,
        config: MovementConfig,
    ) -> None:
        super().__init__(store, registry)
        self.ecology = ecology
        self.genetics = genetics
        self.terrain = terrain
        self.config = config
        # Raise KeyError naming the vocabulary version if either gene does not exist.
        self._speed_index = vocabulary.index_of(config.speed_gene)
        self._size_index = vocabulary.index_of(config.size_gene)

    def top_speed(self, selection: Selection) -> np.ndarray:
        """(len(selection),) float64, world units per tick: the furthest each entity could travel
        in one tick at full pace, from its expressed phenotype."""
        return self.genetics.expressed(selection)[:, self._speed_index].astype(np.float64)

    def settle(self, selection: Selection) -> None:
        """Drop `selection` onto the surface: write ``z`` from the terrain under ``(x, y)``.

        A freshly allocated row holds ``z = 0`` (`EntityStore.allocate`), which is underground
        everywhere the ground is higher than sea level. Seeding a population and #20's births both
        put an entity somewhere without moving it there, so the surface lock needs a way to hold at
        the instant of placement and not only after the first step.
        """
        mask = selection.to_mask()
        elevation = self.terrain.elevation_at(self.store.x[mask], self.store.y[mask])
        self.write("z", selection, elevation.astype(np.float32))

    def step(
        self,
        selection: Selection,
        target_x: np.ndarray,
        target_y: np.ndarray,
        pace: float,
    ) -> None:
        """Advance `selection` one tick toward ``(target_x, target_y)``, charging the effort.

        target_x, target_y: (len(selection),) world units, in ascending row order — the same order
            every service reads a selection in, so a target array from `Hunger.forage_target` lines
            up with the selection it was computed for without either side handling row indices.
        pace: fraction of top speed to travel at, in (0, 1]. `MovementConfig.walking_pace` for an
            unhurried animal; higher for urgency, which costs more per metre (see the module
            docstring). A scalar rather than a per-entity array because pace is a property of the
            *drive* that won this tick, and `driven_by` already partitions the population by drive
            — a fleeing set and a foraging set are two calls, not one call with a mixed column.

        Targets are consumed rather than clamped. Nothing here bounds them against the world
        rectangle, because a step never overshoots its target and never starts out of bounds, so a
        reachable target keeps the entity inside a region it was already inside; the case where
        that fails is a caller inventing an out-of-bounds target, which `Terrain.elevation_at`
        raises on rather than absorbing (§8.7), and which the invariant harness independently
        watches for (§6).
        """
        if not 0.0 < pace <= 1.0:
            raise ValueError(f"pace must be in (0, 1], got {pace}")

        mask = selection.to_mask()
        x = self.store.x[mask].astype(np.float64)
        y = self.store.y[mask].astype(np.float64)
        target_x = np.asarray(target_x, dtype=np.float64)
        target_y = np.asarray(target_y, dtype=np.float64)
        if target_x.shape != x.shape or target_y.shape != y.shape:
            # Checked rather than left to NumPy: a scalar or length-1 target broadcasts cleanly
            # and would march the entire selection at one animal's destination.
            raise ValueError(
                f"targets must have shape {x.shape} for {len(selection)} entities; "
                f"got {target_x.shape} and {target_y.shape}"
            )

        expressed = self.genetics.expressed(selection)
        size = expressed[:, self._size_index].astype(np.float64)
        reach = expressed[:, self._speed_index].astype(np.float64) * pace

        to_target_x = target_x - x
        to_target_y = target_y - y
        distance = np.hypot(to_target_x, to_target_y)
        # An animal standing on its target has no direction, and `Hunger.forage_target` returns
        # exactly that whenever nothing edible is in sight — a normal tick, not an edge case.
        moving = distance > 0.0
        unit_x = np.where(moving, to_target_x / np.where(moving, distance, 1.0), 0.0)
        unit_y = np.where(moving, to_target_y / np.where(moving, distance, 1.0), 0.0)

        # What the animal would do with an unlimited pool: go as far as its legs allow, or stop on
        # the target, whichever comes first.
        intended = np.minimum(reach, distance)
        ground = self.terrain.elevation_at(x, y)
        afford = self._affordable_fraction(
            selection, x, y, unit_x, unit_y, intended, ground, size, pace
        )
        travelled = intended * afford

        # Snapping instead of integrating the last fraction keeps an entity that reaches the edge
        # of the world exactly on it: `Terrain.elevation_at` rejects a position past the boundary,
        # and float error on `x + unit * distance` is enough to land there.
        arrives = travelled >= distance
        new_x = np.where(arrives, target_x, x + unit_x * travelled)
        new_y = np.where(arrives, target_y, y + unit_y * travelled)
        new_z = self.terrain.elevation_at(new_x, new_y)

        self.write("x", selection, new_x.astype(np.float32))
        self.write("y", selection, new_y.astype(np.float32))
        self.write("z", selection, new_z.astype(np.float32))
        self.ecology.spend(
            selection, self._cost(travelled, new_z - ground, size, pace).astype(np.float32)
        )

    def _affordable_fraction(
        self,
        selection: Selection,
        x: np.ndarray,
        y: np.ndarray,
        unit_x: np.ndarray,
        unit_y: np.ndarray,
        intended: np.ndarray,
        ground: np.ndarray,
        size: np.ndarray,
        pace: float,
    ) -> np.ndarray:
        """(n,) float64 in [0, 1]: how much of the intended step each entity can pay for.

        This is the gate that makes hunger close off options (§2.5). The bill for the whole step is
        priced first, and an animal that cannot cover it travels the fraction it can — so an empty
        pool pins a creature in place, and a nearly empty one shuffles.

        The fraction is a linear read of a cost that is not quite linear: shortening a step changes
        where it ends, and therefore how much of the hill it climbed. `step` re-prices the distance
        actually travelled against the elevation actually reached, so the charge is always the true
        cost of the move that happened; this only decides how far to go. An entity that still
        overpays on a concave slope is floored at zero by `Ecology.spend` — it spent everything it
        had getting there, which is the honest outcome and not a case to smooth over.
        """
        landing_z = self.terrain.elevation_at(x + unit_x * intended, y + unit_y * intended)
        full_cost = self._cost(intended, landing_z - ground, size, pace)
        # A zero-cost step is a zero-length one, or an animal whose species does not express size;
        # either way there is nothing it could fail to afford. The same guard appears twice because
        # the division is evaluated over the whole array before `where` selects, so the substitute
        # denominator is what keeps a free step from raising on divide-by-zero.
        payable = full_cost > 0.0
        chargeable = np.where(payable, full_cost, 1.0)
        fraction = np.clip(self.ecology.energy(selection) / chargeable, 0.0, 1.0)
        return np.where(payable, fraction, 1.0)

    def _cost(
        self, distance: np.ndarray, elevation_change: np.ndarray, size: np.ndarray, pace: float
    ) -> np.ndarray:
        """(n,) float64, joules: what covering `distance` while rising `elevation_change` costs.

        Hauling the body over the ground, plus raising it against gravity — both scaled by size,
        since a heavier animal pays more for either. Only the *gain* is charged: descent is
        braking rather than lifting, so it costs its horizontal distance and no more, and that
        asymmetry alone is what makes a ridge a barrier and a valley a corridor.

        Pace enters the horizontal term and not the climb: the premium is for moving urgently, and
        an animal that sprints up a hill has already paid for the sprint.
        """
        haul = self.config.transport_cost * distance * (1.0 + self.config.exertion_premium * pace)
        climb = self.config.climb_cost * np.maximum(elevation_change, 0.0)
        return size * (haul + climb)
