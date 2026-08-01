"""Movement: velocity integrated toward what an animal wants, priced against the metabolic pool
(CLAUDE.md §2.5, §2.6, issues #25, #203, #204).

This is where a decision becomes a consequence. `core.behaviour.service.Behaviour` decides which
way an animal wants to go and how badly; this module is what actually spends energy to get there,
and what makes terrain matter.

Four properties carry the design:

- **Elevation prices travel.** Climbing charges against the same pool everything else charges
  against, so a ridge is expensive to cross rather than merely slow. That is what turns §2.6's
  heightmap from scenery into the isolation barrier speciation needs (#16) — nobody places a
  barrier, and a mountain range becomes one because crossing it costs more than the far side is
  worth.
- **Effort is charged, not just distance** (§2.5). Cost per world unit rises with `pace`, so a
  sprint is dearer than a stroll over the same ground. Pricing distance alone would make a chase
  merely long; it is the per-unit premium that makes a predator pay for every chase it loses and
  prey pay for every escape. Nothing here knows what fleeing *is*: `pace` is derived from how much
  better the chosen option was than doing nothing, so a drive gains the power to make an animal
  hurry by *scoring* — #19's chase and #24's flight are priced without this module changing.
- **Velocity is state, and it changes at a bounded rate** (#204). An animal is not repositioned
  toward a target each tick; its velocity turns and grows toward what it wants, at most `agility /
  size` per tick. So reversing costs time, a heavy body corners badly, and a pursuit is a pursuit
  rather than an arrival. This is what lets #179 resolve a chase by *contact* instead of by a kill
  formula: with momentum, whether the two meet is an outcome of the physics.
- **The pool gates the step, it does not merely record it.** An animal that cannot pay for the
  whole step covers only what it can afford, and an empty one does not move at all. This is §2.5's
  "a starving animal can neither run nor hide" as a mechanism rather than as a mood: hunger closes
  off options instead of reading high. Velocity is then written from what *happened*, not from what
  was intended, so an animal that ran out of energy mid-step ends the tick slower and one that ran
  into the world edge ends it stopped — neither needs a rule of its own.

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
from core.genetics.registry import GeneRegistry, Unit
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
    size_gene: the gene whose expressed value scales every cost term below, and which divides
        `agility_gene` to give the rate velocity may change at. A bigger body is more expensive to
        haul over the same ground and up the same hill *and* slower to turn, which together are the
        counterweight that stops size running away on the benefits it buys elsewhere. Mass
        resisting a change of direction is free physics rather than an authored penalty (#204).
    agility_gene: the gene whose expressed value is how fast velocity may change, in **world units
        per tick per tick** — a length, since the tick is unitless, exactly as `speed_gene` is.
        Divided by expressed size, so `agility / size` is the real acceleration and a lineage can
        trade being big against being nimble. Must charge a positive cost: turning quickly is pure
        benefit otherwise and runs away in every world, which is the rule §2.5 states for
        insulation and senescence resistance.
    haste_gene: the gene whose expressed value converts a utility advantage into a pace — how
        readily this animal turns a reason into speed. Read as a scale (see `Movement.pace`), and
        free, because hurrying already charges `exertion_premium` on every world unit: the
        selective consequence is immediate, which is the one case §2.5 exempts from the metabolic
        budget.
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
    walking_pace: the fraction of top speed an animal with **no particular reason to hurry** uses
        — the floor `Movement.pace` starts from, not a constant every animal moves at. Config
        rather than a literal at the call site because it is one half of the walk/sprint ratio
        `exertion_premium` prices, and tuning either one alone is what §2.1 means by constants
        drifting apart. Must be in (0, 1): zero would mean an animal that never travels, and one
        would leave nothing for haste to buy, so the premium would multiply a constant again —
        which is the whole of #203.
    """

    speed_gene: str
    size_gene: str
    agility_gene: str
    haste_gene: str
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
        if not 0.0 < self.walking_pace < 1.0:
            raise ValueError(
                f"walking_pace must be in (0, 1), got {self.walking_pace}; at 1 every animal is "
                "already flat out and `exertion_premium` multiplies a constant again (#203)"
            )


class Movement(DomainService):
    """Owns the position and velocity columns: where every entity is, how it is travelling, and
    what it cost to get there.

    Surface-locked (§2.6): ``z`` is the terrain elevation under ``(x, y)``, written on every step
    and by `settle`. The column is stored rather than derived on read because the spatial index and
    the viewer both consume it, and because free flight (§2.6's staged plan) unlocks it later
    without changing anyone's read. Velocity is two columns and not three for the same reason:
    nothing chooses a vertical speed, so integrating one would be state nothing writes.

    ecology: the owner of `energy` (#17). Every locomotion charge goes through `Ecology.spend`,
        because this service does not own that column and must not subtract from it directly
        (CLAUDE.md §2.3).
    exertion: the owner of `exertion` (#107), told what each step took so that `Fatigue` has
        something to read. Same relationship as `ecology` above and for the same reason — the bill
        is handed over, never applied here.
    genetics: consulted for expressed phenotype only — this service never writes a gene. Speed and
        size are read through `expressed`, so a species that does not express speed does not move,
        exactly as it does not pay for speed.
    terrain: the height field, sampled at every cell crossing along a step to price the climb
        (#113), and the rectangle a landing point is clipped into.
    """

    owns = ("x", "y", "z", "velocity_x", "velocity_y")

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
        genes: GeneRegistry,
        config: MovementConfig,
    ) -> None:
        super().__init__(store, registry)
        self.ecology = ecology
        self.exertion = exertion
        self.genetics = genetics
        self.terrain = terrain
        self.config = config
        # Raise KeyError naming the vocabulary version if a gene does not exist, and ValueError if
        # one is declared in a dimension this module would misread (#111). Top speed is world units
        # per tick and agility world units per tick per tick; the tick is unitless, so both are
        # lengths. Size is a bare scaling factor on every cost term, and haste is a bare scale on a
        # utility.
        self._speed_index = genes.index_of(config.speed_gene, unit=Unit.LENGTH)
        self._size_index = genes.index_of(config.size_gene, unit=Unit.DIMENSIONLESS)
        self._agility_index = genes.index_of(config.agility_gene, unit=Unit.LENGTH)
        self._haste_index = genes.index_of(config.haste_gene, unit=Unit.DIMENSIONLESS)
        # A gene that only ever buys performance and charges nothing runs away in every world —
        # §2.5's rule for insulation and senescence resistance, and agility is the same shape:
        # there is no world in which turning faster is worse. Refused at construction rather than
        # guarded per tick, at the one place the cost and the caller's intent are both in hand
        # (§8.7).
        if genes.spec(config.agility_gene).cost <= 0:
            raise ValueError(
                f"agility gene '{config.agility_gene}' must charge a positive cost; turning "
                "faster is pure benefit and is bounded by nothing else (§2.5)"
            )

    def top_speed(self, selection: Selection) -> np.ndarray:
        """(len(selection),) float64, world units per tick: the furthest each entity could travel
        in one tick at full pace, from its expressed phenotype."""
        return self.genetics.expressed(selection)[:, self._speed_index].astype(np.float64)

    def pace(self, selection: Selection, urge: np.ndarray) -> np.ndarray:
        """(len(selection),) float64 in [walking_pace, 1]: how hard each animal pushes.

        urge: (len(selection),) unit-free — how much better the option it chose was than standing
            still, which `Behaviour` records as `choice_urge`. Negative values are read as zero: the
            Boltzmann draw can pick an option worse than resting, and an animal that did so by
            accident has no reason to hurry over it.

        ```
        pace = 1 − (1 − walking_pace) × exp(−haste × urge)
        ```

        **A saturating map rather than a linear one, because utilities have no ceiling.** The sum
        over drives is unbounded above, so anything linear needs a cutoff, and a cutoff is a second
        constant to tune beside `walking_pace`. Exponential decay of the *remaining* headroom needs
        neither: no urge is enough to exceed top speed, and every increment of urge buys a fixed
        fraction of what is left. Past an urge of roughly 90 the headroom rounds away in float64
        and the pace is exactly 1 — flat out is a legitimate state, and the property that matters
        is that it is never exceeded, which is what `no_entity_exceeds_its_top_speed` asserts.

        **The scale is a gene rather than a measured constant** (#203 weighed both). Utilities are
        unnormalised, so "how much advantage counts as a lot" is not a fact about the world that
        could be measured once — it depends on the drive weights an animal carries, which are
        themselves genes (#23). Selection therefore calibrates it per lineage, and a world whose
        utilities drift in scale needs no retune. What it buys ecologically is temperament: a high
        lineage bolts at every provocation and pays the premium constantly, a low one is placid and
        is caught.

        Note what is *not* here: no drive names a pace, and no branch asks what an animal is doing.
        A drive makes an animal hurry by scoring the option higher, which is exactly what §2.5
        promised when it said a drive "passes a higher number" — the number is its own urgency, and
        it arrives through the utility sum rather than through this module knowing about it.
        """
        return self._pace(
            self.genetics.expressed(selection)[:, self._haste_index].astype(np.float64), urge
        )

    def _pace(self, haste: np.ndarray, urge: np.ndarray) -> np.ndarray:
        """`pace` from an already-expressed haste column, so `step` needs only one phenotype read.

        `Genetics.expressed` rebuilds the whole `(n, n_genes)` block per call, and a second call
        inside one tick is a block nobody needed — the same argument #114 makes for sampling
        candidate positions once and sharing them across drives.
        """
        wanted = np.maximum(np.asarray(urge, dtype=np.float64), 0.0)
        headroom = 1.0 - self.config.walking_pace
        return 1.0 - headroom * np.exp(-haste * wanted)

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
        urge: np.ndarray,
    ) -> None:
        """Advance `selection` one tick, steering toward ``(target_x, target_y)``, and charge it.

        target_x, target_y: (len(selection),) world units, in ascending row order — the same order
            every service reads a selection in, so a target array from `Behaviour.chosen_target`
            lines up with the selection it was computed for without either side handling row
            indices. It is the point being **steered toward**, not a destination: momentum means an
            animal cannot stop dead on a mark, so a fast one overshoots and a turning one arcs.
            An animal handed its own position is asking to come to a halt, which is how a chosen
            rest arrives here (#114) and is the only thing this module needs to know about resting.
        urge: (len(selection),) unit-free, how much better the chosen option was than standing
            still — converted to a per-entity pace by `pace`. An array rather than the scalar it
            replaced, which is #203: with one pace for the whole world nothing could ever hurry and
            `exertion_premium` multiplied a constant.

        Targets are consumed rather than clamped, but the **landing point** is clipped into the
        world: velocity is what decides where a step ends, so a target near the edge no longer
        bounds it. Clipping the landing rather than the target is also what makes a wall stop an
        animal instead of deflecting it, since velocity is rewritten from the displacement that
        actually happened.
        """
        mask = selection.to_mask()
        x = self.store.x[mask].astype(np.float64)
        y = self.store.y[mask].astype(np.float64)
        target_x = np.asarray(target_x, dtype=np.float64)
        target_y = np.asarray(target_y, dtype=np.float64)
        urge = np.asarray(urge, dtype=np.float64)
        if target_x.shape != x.shape or target_y.shape != y.shape or urge.shape != x.shape:
            # Checked rather than left to NumPy: a scalar or length-1 target broadcasts cleanly
            # and would march the entire selection at one animal's destination.
            raise ValueError(
                f"targets and urge must have shape {x.shape} for {len(selection)} entities; "
                f"got {target_x.shape}, {target_y.shape} and {urge.shape}"
            )

        expressed = self.genetics.expressed(selection)
        size = expressed[:, self._size_index].astype(np.float64)
        pace = self._pace(expressed[:, self._haste_index].astype(np.float64), urge)

        velocity_x, velocity_y = self._accelerate(
            selection,
            x,
            y,
            target_x,
            target_y,
            expressed[:, self._speed_index].astype(np.float64) * pace,
            expressed[:, self._agility_index].astype(np.float64),
            size,
        )

        # Where this velocity would put the animal, kept inside the world. Everything downstream
        # walks toward that landing point, so the cell-crossing pass and the boundary snap in
        # `_landing` are unchanged by momentum — only where the step points has moved upstream.
        landing_x = np.clip(x + velocity_x, 0.0, self.terrain.world_width)
        landing_y = np.clip(y + velocity_y, 0.0, self.terrain.world_height)
        along_x = landing_x - x
        along_y = landing_y - y
        distance = np.hypot(along_x, along_y)
        moving = distance > 0.0
        unit_x = np.where(moving, along_x / np.where(moving, distance, 1.0), 0.0)
        unit_y = np.where(moving, along_y / np.where(moving, distance, 1.0), 0.0)

        # The budget is in work rather than energy, because `_walk` accumulates work per unit of
        # body size (see `_work`). A species that does not express size pays nothing however far it
        # goes, so nothing bounds its walk.
        weightless = size <= 0.0
        budget = np.where(
            weightless, np.inf, self.ecology.energy(selection) / np.where(weightless, 1.0, size)
        )
        travelled, ascent = self._walk(
            x, y, unit_x, unit_y, distance, landing_x, landing_y, budget, pace
        )

        new_x, new_y = self._landing(
            x, y, unit_x, unit_y, travelled, distance, landing_x, landing_y
        )
        new_z = self.terrain.elevation_at(new_x, new_y)

        self.write("x", selection, new_x.astype(np.float32))
        self.write("y", selection, new_y.astype(np.float32))
        self.write("z", selection, new_z.astype(np.float32))
        # Velocity is written from the displacement that *happened*, never from the one intended.
        # That is what makes running out of energy and running into a wall need no rules of their
        # own: an animal that could only afford half its step carries half the speed into the next
        # tick, and one that reached the edge carries none. It also bounds velocity by top speed
        # without a cap — the new value is never longer than the intended one, which is never
        # longer than the larger of the old velocity and `top_speed × pace` — which is the
        # invariant `no_entity_exceeds_its_top_speed` asserts (§6).
        self.write("velocity_x", selection, (unit_x * travelled).astype(np.float32))
        self.write("velocity_y", selection, (unit_y * travelled).astype(np.float32))
        # The bill and the record of effort are the same quantity read two ways: `Ecology` is
        # charged the size-scaled cost, `Exertion` accumulates the per-size work, so a sprint up a
        # ridge is both expensive and tiring while a stroll over the same ground is neither (#107).
        work = self._work(travelled, ascent, pace)
        self.ecology.spend(selection, (size * work).astype(np.float32))
        self.exertion.accumulate(selection, work)

    def _accelerate(
        self,
        selection: Selection,
        x: np.ndarray,
        y: np.ndarray,
        target_x: np.ndarray,
        target_y: np.ndarray,
        wanted_speed: np.ndarray,
        agility: np.ndarray,
        size: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """(velocity_x, velocity_y): last tick's velocity moved toward what the animal wants.

        The desired velocity is `wanted_speed` pointed at the target, or **zero** for an animal
        standing on its own target — which is a chosen rest (#114), and which is why stopping is
        expressed here rather than by a branch: an animal that wants zero velocity decelerates by
        the same rule that turns it, so coming to a halt takes as long as getting going.

        The change is capped in *magnitude*, not per axis. A per-axis cap would make an animal turn
        faster along the diagonals than along the axes, which is the grid leaking into the physics
        — the same defect §2.5 rejects 8-way movement for.

        ```
        limit = agility / size
        ```

        **Divided by size, so mass resists a change of direction.** That is what gives `size` its
        first downside beyond upkeep and creates a genuine speed-against-agility axis for selection
        to work on: a big fast animal that corners badly and a small slow one that jinks are both
        viable, which is the difference between a predator and its prey being *different kinds of
        animal* rather than two speed values (#204). A species expressing no size has nothing to
        resist the change, matching the unlimited energy budget `step` gives it for the same
        reason.
        """
        toward_x = target_x - x
        toward_y = target_y - y
        bearing = np.hypot(toward_x, toward_y)
        steering = bearing > 0.0
        scale = np.where(steering, wanted_speed / np.where(steering, bearing, 1.0), 0.0)
        mask = selection.to_mask()
        velocity_x = self.store.velocity_x[mask].astype(np.float64)
        velocity_y = self.store.velocity_y[mask].astype(np.float64)

        change_x = toward_x * scale - velocity_x
        change_y = toward_y * scale - velocity_y
        change = np.hypot(change_x, change_y)
        weightless = size <= 0.0
        limit = np.where(weightless, np.inf, agility / np.where(weightless, 1.0, size))
        turning = change > 0.0
        held = np.minimum(1.0, np.where(turning, limit / np.where(turning, change, 1.0), 1.0))
        return velocity_x + change_x * held, velocity_y + change_y * held

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
        distance: np.ndarray,
        target_x: np.ndarray,
        target_y: np.ndarray,
        budget: np.ndarray,
        pace: np.ndarray,
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

        travelled = np.zeros_like(distance)
        ascent = np.zeros_like(distance)
        spent = np.zeros_like(distance)
        here_z = self.terrain.elevation_at(x, y)
        # Nothing moving has anything to walk, and an exhausted animal has nothing to spend.
        walking = (distance > 0.0) & (budget > 0.0)

        # A step of length d crosses at most d/cell_size grid lines on each axis, so 2*ceil(...)
        # bounds the crossings and the +2 covers the partial cell at each end. Derived from the
        # population each tick rather than fixed, per #113.
        longest = float(distance.max(initial=0.0))
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
            left = distance - travelled
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
            walking = walking & ~unaffordable & (travelled < distance)

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

    def _work(self, distance: np.ndarray, ascent: np.ndarray, pace: np.ndarray) -> np.ndarray:
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
