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
- **Effort is charged, not just distance** (§2.5). Cost per world unit rises with `pace`, so a
  sprint is dearer than a stroll over the same ground. Pricing distance alone would make a chase
  merely long; it is the per-unit premium that makes a predator pay for every chase it loses and
  prey pay for every escape. Nothing here knows what fleeing *is* — a drive that wants urgency
  passes a higher pace, and #19's chase and #24's flight are then priced without this module
  changing.
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

from core.behaviour.exertion import Exertion
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
    transport_cost: energy units per world unit travelled, per unit of expressed size, at zero pace.
        Must be positive: at zero, distance is free and nothing stops an animal crossing the world
        every tick, which removes the cost half of §2.5's hard budget for the one trait that
        spends the most.
    exertion_premium: extra fraction of `transport_cost` charged at full pace, so a world unit at
        pace ``p`` costs ``transport_cost × (1 + exertion_premium × p)``. Must be non-negative —
        negative would make sprinting cheaper per unit than walking, inverting §2.5.
    climb_cost: energy units per **world unit** of elevation *gained*, per unit of expressed size —
        the same length unit `transport_cost` is denominated in (#112), so ``climb_cost /
        transport_cost`` is a real statement: how many world units of flat ground cost what one
        world unit of climb does. Elevation used to be documented in a physical length unit while
        x and y were in world units, which left that ratio resting on a conversion factor nothing
        declared, nothing checked, and everything depended on — the same shape of defect as the
        prototype's degree-valued sight angle compared against a radian difference (§8.4).

        Descent charges nothing beyond its horizontal distance: raising a body against gravity is
        work in a way that lowering it is not, and that asymmetry is the whole of "uphill costs
        more than downhill". Must be non-negative; negative would mint energy out of walking
        uphill, which §2.5's closed loop forbids outright.
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
                "negative makes sprinting cheaper per world unit than walking"
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
    exertion: the owner of `exertion` (#107), told what each step took so that `Fatigue` has
        something to read. Same relationship as `ecology` above and for the same reason — the bill
        is handed over, never applied here.
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
        exertion: Exertion,
        genetics: Genetics,
        terrain: Terrain,
        vocabulary: GeneVocabulary,
        config: MovementConfig,
    ) -> None:
        super().__init__(store, registry)
        self.ecology = ecology
        self.exertion = exertion
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
            unhurried animal; higher for urgency, which costs more per world unit (see the module
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

        # The budget is in work rather than energy, because `_walk` accumulates work per unit of
        # body size (see `_work`). A species that does not express size pays nothing however far it
        # goes, so nothing bounds its walk.
        weightless = size <= 0.0
        budget = np.where(
            weightless, np.inf, self.ecology.energy(selection) / np.where(weightless, 1.0, size)
        )
        travelled, ascent = self._walk(
            x, y, unit_x, unit_y, intended, distance, target_x, target_y, budget, pace
        )

        new_x, new_y = self._landing(
            x, y, unit_x, unit_y, travelled, distance, target_x, target_y
        )
        new_z = self.terrain.elevation_at(new_x, new_y)

        self.write("x", selection, new_x.astype(np.float32))
        self.write("y", selection, new_y.astype(np.float32))
        self.write("z", selection, new_z.astype(np.float32))
        # The bill and the record of effort are the same quantity read two ways: `Ecology` is
        # charged the size-scaled cost, `Exertion` accumulates the per-size work, so a sprint up a
        # ridge is both expensive and tiring while a stroll over the same ground is neither (#107).
        work = self._work(travelled, ascent, pace)
        self.ecology.spend(selection, (size * work).astype(np.float32))
        self.exertion.accumulate(selection, work)

    def _landing(
        self,
        x: np.ndarray,
        y: np.ndarray,
        unit_x: np.ndarray,
        unit_y: np.ndarray,
        travel: np.ndarray,
        distance: np.ndarray,
        target_x: np.ndarray,
        target_y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Where a step of `travel` from ``(x, y)`` ends, snapped to the target on arrival.

        Snapping instead of integrating the last fraction keeps an entity that reaches the edge of
        the world exactly on it: `Terrain.elevation_at` rejects a position past the boundary, and
        float error on ``x + unit * distance`` is enough to land there — measured at 255 of 200,000
        random diagonal steps onto an edge.

        **Shared by the pricing pass and the move itself**, which is the whole point of it existing
        (#128). The two computed the landing point separately and only the move snapped, so a
        forager whose chosen patch sat on the world boundary raised out of the middle of a tick —
        rarely enough to look like a flake and often enough to kill a populated world in three
        ticks. One rule cannot disagree with itself.
        """
        arrives = travel >= distance
        return (
            np.where(arrives, target_x, x + unit_x * travel),
            np.where(arrives, target_y, y + unit_y * travel),
        )

    def _walk(
        self,
        x: np.ndarray,
        y: np.ndarray,
        unit_x: np.ndarray,
        unit_y: np.ndarray,
        intended: np.ndarray,
        distance: np.ndarray,
        target_x: np.ndarray,
        target_y: np.ndarray,
        budget: np.ndarray,
        pace: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """(travelled, ascent): how far each entity got, and how much it climbed getting there.

        **Every cell the step crosses is visited** (#113). Sampling only the two ends nets a
        descent against a climb, so a step that dropped into a ravine and hauled itself out read as
        level ground and was charged for none of it. That is not a rounding error in a cost: since
        impassable ground is expressed *entirely* through what it costs, terrain the cost function
        never sees is not a barrier at all.

        Elevation is bilinear (`Terrain.elevation_at`), so the profile along a straight line bends
        only where the line crosses a grid line. Sampling at exactly those crossings therefore
        captures every change of direction the field actually has — within one cell the profile is
        a single arc, and a rise and fall inside one cell is below the terrain's own resolution.

        **The iteration count is per tick, not per animal.** Each pass advances every still-moving
        entity to its own next crossing, with the finished ones masked off, so the loop runs as many
        times as the *longest* step in this selection needs rather than once per animal. That is
        what keeps a cell walk vectorized (§2.3), and it is why no speed cap is needed: a lineage
        that evolves an absurd top speed makes ticks slower, which #46's regression gates would say
        out loud, rather than making the answer quietly wrong (§8.7).

        **The budget is spent as it is walked**, which is what makes "where did it run out" a
        well-defined question: an animal with barely enough energy stops at the foot of a wall
        rather than partway up it in proportion to some flat average. Within the final cell the
        remaining budget is spent linearly, since one cell is the resolution of everything here.
        """
        cell_size = self.terrain.cell_size
        haul_rate = self.config.transport_cost * (1.0 + self.config.exertion_premium * pace)

        travelled = np.zeros_like(intended)
        ascent = np.zeros_like(intended)
        spent = np.zeros_like(intended)
        here_z = self.terrain.elevation_at(x, y)
        # Nothing moving has anything to walk, and an exhausted animal has nothing to spend.
        walking = (intended > 0.0) & (budget > 0.0)

        # A step of length d crosses at most d/cell_size grid lines on each axis, so 2*ceil(...)
        # bounds the crossings and the +2 covers the partial cell at each end. Derived from the
        # population each tick rather than fixed, per #113.
        longest = float(intended.max(initial=0.0))
        passes = 2 * int(np.ceil(longest / cell_size)) + 2

        for remaining_passes in range(passes, 0, -1):
            if not walking.any():
                break

            at_x = x + unit_x * travelled
            at_y = y + unit_y * travelled
            # Distance along the ray to the next grid line on each axis; an axis the step does not
            # move along is never crossed, and `inf` drops it out of the minimum below.
            to_boundary = np.minimum(
                self._to_next_grid_line(at_x, unit_x, cell_size),
                self._to_next_grid_line(at_y, unit_y, cell_size),
            )
            left = intended - travelled
            # On the final permitted pass, finish whatever is left in one segment rather than
            # stopping short. Only reachable if float slivers at a grid line consumed passes the
            # bound did not expect, and it costs accuracy within one cell rather than distance.
            step = left if remaining_passes == 1 else np.minimum(to_boundary, left)

            next_x, next_y = self._landing(
                x, y, unit_x, unit_y, travelled + step, distance, target_x, target_y
            )
            next_z = self.terrain.elevation_at(next_x, next_y)
            # Only the gain: descent is braking rather than lifting (see `_work`), and summing the
            # gains rather than the net difference is the whole of this issue.
            segment_ascent = np.maximum(next_z - here_z, 0.0)
            segment_work = haul_rate * step + self.config.climb_cost * segment_ascent

            affordable = budget - spent
            unaffordable = walking & (segment_work > affordable)
            # A fraction of a segment it cannot finish; the guard keeps a zero-cost segment (flat
            # ground with a free transport cost cannot happen, but a zero-length one can) out of the
            # division, which is evaluated over the whole array before `where` selects.
            chargeable = np.where(segment_work > 0.0, segment_work, 1.0)
            fraction = np.clip(affordable / chargeable, 0.0, 1.0)

            advanced = np.where(unaffordable, step * fraction, step)
            travelled = np.where(walking, travelled + advanced, travelled)
            ascent = np.where(
                walking,
                ascent + np.where(unaffordable, segment_ascent * fraction, segment_ascent),
                ascent,
            )
            spent = np.where(
                walking, spent + np.where(unaffordable, affordable, segment_work), spent
            )
            # An entity that reached its own next crossing stands on it, so the cell it is about to
            # enter starts from that elevation. One that ran out mid-cell is done and never reads
            # `here_z` again.
            here_z = np.where(walking & ~unaffordable, next_z, here_z)
            walking = walking & ~unaffordable & (travelled < intended)

        return travelled, ascent

    @staticmethod
    def _to_next_grid_line(
        position: np.ndarray, direction: np.ndarray, cell_size: float
    ) -> np.ndarray:
        """(n,) float64: distance along a unit ray to the next cell boundary on one axis.

        `inf` where the ray does not move along this axis, so a minimum against the other axis
        picks the crossing that actually happens. A position sitting exactly on a boundary returns
        a whole cell rather than zero, which is what stops the walk stalling on a grid line.
        """
        cells = position / cell_size
        forward = direction > 0.0
        backward = direction < 0.0
        boundary = np.where(
            forward, (np.floor(cells) + 1.0) * cell_size, (np.ceil(cells) - 1.0) * cell_size
        )
        moves = forward | backward
        return np.where(moves, (boundary - position) / np.where(moves, direction, 1.0), np.inf)

    def _work(self, distance: np.ndarray, ascent: np.ndarray, pace: float) -> np.ndarray:
        """(n,) float64: what covering `distance` while climbing `ascent` takes, per unit of body
        size.

        Hauling the body over the ground, plus raising it against gravity. `ascent` is the total
        climbed along the path (`_walk`), never the net difference between the ends: descent is
        braking rather than lifting, so it costs its horizontal distance and no more, and that
        asymmetry alone is what makes a ridge a barrier and a valley a corridor. Netting the two
        would hand back the climb out of every valley an animal fell into.

        Pace enters the horizontal term and not the climb: the premium is for moving urgently, and
        an animal that sprints up a hill has already paid for the sprint.

        **Size is deliberately not applied here.** Multiplied by size this is the energy bill, and
        left as it is it is the exertion an animal feels — a heavier animal pays more fuel for the
        same ground but is not thereby more tired, and #107 needs both readings of one quantity.
        """
        haul = self.config.transport_cost * distance * (1.0 + self.config.exertion_premium * pace)
        return haul + self.config.climb_cost * ascent
