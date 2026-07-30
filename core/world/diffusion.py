"""Cost-aware diffusion: spreading a signal over ground that is expensive to cross (#93).

A field of *something worth walking to* — standing crop today, prey and mates later — diffused
outward so that every cell holds how much of it is reachable from there, discounted by how far
away it is **and by what the ground between costs**. A forager then reads a heading straight off
the gradient rather than ranking a list of candidate patches.

**Why one operator rather than a distance discount plus a terrain check.** The discrete version of
this question ranked patches by `biomass / (1 + distance / forage_reluctance)`, which is a distance
discount and nothing else: a meadow on the far side of a ridge scored exactly as well as one on open
ground the same number of world units away. Diffusion makes the discount fall out of the spreading
itself, and once the spreading is cost-aware the climb discount is the same mechanism rather than a
second coefficient bolted alongside the first. §2.1's warning about constants that must be tuned as
a table is the whole argument: two coefficients that describe one preference will drift apart.

**Only elevation gain impedes**, matching what a step actually costs (§2.5): reaching a source
uphill is discounted, reaching one downhill is not. An animal in a valley therefore perceives the
slope above it less readily than one on the rim perceives the valley floor — which is the same
asymmetry that makes a ridge a barrier and a valley a corridor, arriving in perception rather than
being asserted separately.

**The signal routes around obstacles rather than being attenuated through them.** Because spreading
is a walk over the neighbour graph, what arrives behind a wall is whatever came round the end of it,
so a barrier with a gap is a detour and one without is a wall. That is what makes a fence (#27) an
intervention rather than a multiplier, and it is why this is not simply "sample the field and divide
by relief".

**Nothing here moves biomass.** The field says what is *readable* from where; `Plants.graze` is what
actually removes a mouthful, and it operates on the source rather than on this.

**Known limitation: a rim of one range-length reads slightly rich.** A cell on the boundary has
fewer directions to leave by, so a walk starting there is confined and revisits nearby source more
often than an interior walk does. The stationary distribution is still uniform — the operator is
symmetric and doubly stochastic, so a *uniform* field stays uniform exactly — but before it settles,
a cell within about one `range` of the edge reads higher than an equivalent interior cell. Measured
on flat ground with `range = 4`: a source two cells from a corner leaves the corner at 1.036 against
0.925 at the source's own cell, so a forager in that corner reads no reason to leave it. Filed as
#140; it is a property of any reflecting boundary rather than of this parameterisation, and
the alternative — letting signal drain off the map — trades it for a rim that reads spuriously
*empty*, which is the worse error because the world's edge would then repel.

Terrain never changes, so the per-edge conductance is computed once at construction and every tick
pays only for the passes. Those are quadratic in `range` (`_passes_for`), which is the one cost of
this design worth stating out loud: doubling how far food advertises itself quadruples what the
field costs per tick.

## What this deliberately does not do yet

`core.ecology.cues` wants exactly this operator — scent currently diffuses through a mountain
unattenuated — and converting it is #139 rather than done here, because
`CueField.sample_excluding_self` subtracts the *exact* diagonal of its blur, computed as an outer
product of two 1-D diagonals. That factorisation exists only because a separable blur is separable;
a per-edge conductance is not, and an approximate self-response would leave every cannibal lineage
faintly afraid of itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.world.terrain import Terrain, bilinear_sample

def _passes_for(diffusion_range: float, cell_size: float) -> int:
    """How many averaging passes spread a signal `diffusion_range` world units.

    One pass is one step of a walk over the neighbour graph, so `n` passes spread a source over
    roughly `sqrt(n)` cells — diffusion widens with the square root of time, not linearly. Inverting
    that is what makes `range` mean the distance it says it does.

    The `5 / 4` is the laziness correction: a cell keeps a fifth of its own weight each pass (see
    `_conductance`), so only four fifths of a pass goes into spreading and the same width needs
    proportionally more of them. Checked against the operator rather than trusted — a declared range
    of 2, 4, 8 and 16 measures an RMS spread of 2.00, 4.00, 8.00 and 16.00 world units.

    This replaced a decay-per-hop parameterisation that did not work: with a source re-injected
    every pass, dilution over the growing area dominated the decay, and a declared range of 16 cells
    produced a measured e-folding length of 1.33. The knob moved the answer by a factor of 2.3 across
    an 8x change in its value, which is a coefficient that does not control what it is named for.
    """
    return max(1, int(round(1.25 * (diffusion_range / cell_size) ** 2)))


@dataclass(frozen=True)
class DiffusionConfig:
    """Per-world tuning for one diffused field — never constants in `core/` (§2.1).

    range: world units. How far a signal carries over flat ground, as the RMS spread of what a point
        source becomes — measured, not nominal (`_passes_for`). It is what decides whether foragers
        stay local and strip ground bare before moving on (small) or spread their pressure out
        (large), so it is the coefficient that used to be a drive's `forage_reluctance`. Must be
        positive; at zero nothing carries and every cell would read only itself.

        **Costs passes quadratically**, since diffusion widens with the square root of time. A range
        of 4 cells is 20 whole-grid passes per tick and a range of 16 is 320, so this is a knob with
        a real price rather than a free preference.
    climb_penalty: per world unit of elevation *gained* between neighbouring cells. Zero is legal
        and means a world whose relief does not impede perception. Must be non-negative — negative
        would carry a signal further uphill than over the flat, which is the barrier inverted.
    """

    range: float
    climb_penalty: float

    def __post_init__(self) -> None:
        if self.range <= 0:
            raise ValueError(
                f"range must be positive, got {self.range}; at zero nothing carries and "
                "every cell reads only itself"
            )
        if self.climb_penalty < 0:
            raise ValueError(
                f"climb_penalty must be non-negative, got {self.climb_penalty}; "
                "negative carries a signal further uphill than over flat ground"
            )


class CostAwareDiffusion:
    """A terrain's per-edge conductance, resolved once, applied to any field over its grid.

    conductance: ``(5, height, width)`` float32, unit-free in [0, 1] — how readily a signal passes
        into each cell from itself and from its four neighbours, already normalised so a cell's five
        weights sum to one. Plane order is self, north, south, west, east, matching the shifts in
        `_neighbours`. Edges of the world conduct nothing inward from outside, which makes the
        boundary reflecting rather than a sink: a cell at the map edge is not spuriously deaf.
    passes: how many relaxation passes one `spread` costs. Derived from `range` and the cell size
        rather than configured, because it is a numerical property of the kernel and not an
        ecological choice.
    """

    def __init__(self, terrain: Terrain, config: DiffusionConfig) -> None:
        self.terrain = terrain
        self.config = config
        self.passes = _passes_for(config.range, terrain.cell_size)
        self.conductance = _conductance(terrain, config.climb_penalty)

    def spread(self, source: np.ndarray) -> np.ndarray:
        """``(height, width)`` float32: `source` diffused over the grid it is defined on.

        source: ``(height, width)``, non-negative, in whatever unit the caller's field carries —
            energy units for standing crop.

        Each pass replaces every cell with a conductance-weighted average of what its neighbours
        held, which is one step of a walk over the neighbour graph. After `passes` steps a cell
        holds how much of the source is reachable from it, spread over roughly `range` world units
        and routed around whatever the conductance made expensive. Whole-array operations
        throughout: no per-cell and no per-source loop (§2.3).

        **Nothing is ever amplified.** Each cell takes a weighted average of its neighbours, which
        can never exceed the largest of them, so the field is bounded by the source's own maximum
        however many passes run. Total is *not* conserved and is not meant to be — the conductance
        is deliberately asymmetric, since only a climb impedes — but a gradient can only ever point
        at something that is really there, which is the property that matters.
        """
        if source.shape != self.terrain.heights.shape:
            raise ValueError(
                f"source must cover the terrain grid {self.terrain.heights.shape}, "
                f"got {source.shape}"
            )
        field = np.asarray(source, dtype=np.float32)
        for _ in range(self.passes):
            field = (self.conductance * _neighbours(field)).sum(axis=0)
        return field

    def gradient_at(
        self, field: np.ndarray, x: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """(gx, gy), each ``(n,)`` float64: which way the field rises, at continuous positions.

        A direction rather than a target: the operator answers "which way is better from here",
        and how far to commit to that answer is the caller's (#114 will make it a heading among
        options; today `Hunger` walks one step along it).

        Central differences one cell either side, bilinearly sampled so a forager standing between
        cells gets the interpolation everything else over this grid uses (`bilinear_sample`). The
        probe is clamped inside the world, so a cell at the boundary differences against itself and
        reads no push outward rather than raising — foragers do reach the edge exactly
        (`Movement._landing`), and a gradient undefined there would fail on the one position that
        is guaranteed to occur.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        step = self.terrain.cell_size
        width = self.terrain.world_width
        height = self.terrain.world_height

        east = self._sample(field, np.clip(x + step, 0.0, width), y)
        west = self._sample(field, np.clip(x - step, 0.0, width), y)
        north = self._sample(field, x, np.clip(y + step, 0.0, height))
        south = self._sample(field, x, np.clip(y - step, 0.0, height))
        # Divided by the probe's own span rather than by 2·step, so a clamped probe at the edge is
        # a one-sided difference over the distance it actually spanned instead of a half-size
        # reading of the same slope.
        return (
            _rise(east - west, np.clip(x + step, 0.0, width) - np.clip(x - step, 0.0, width)),
            _rise(north - south, np.clip(y + step, 0.0, height) - np.clip(y - step, 0.0, height)),
        )

    def _sample(self, field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return bilinear_sample(
            field, x, y, self.terrain.cell_size, self.terrain.world_width, self.terrain.world_height
        )


def _rise(difference: np.ndarray, span: np.ndarray) -> np.ndarray:
    """(n,) float64: a slope, and zero where the probe had nowhere to span (a 1×1 world)."""
    spanned = span > 0.0
    return np.where(spanned, difference / np.where(spanned, span, 1.0), 0.0)


def _neighbours(field: np.ndarray) -> np.ndarray:
    """``(5, height, width)``: what each cell itself, then its north, south, west and east
    neighbour, holds.

    **The cell itself is the first plane, and that is not an optimisation.** A four-neighbour step
    always changes the parity of ``row + column``, so a walk of an even number of steps can only
    ever land on cells of the source's own parity: the field comes out as a checkerboard with every
    other cell at exactly zero, and a gradient read across one of those zeros points nowhere.
    Keeping weight on the cell itself makes the walk *lazy*, which destroys that parity structure —
    the standard fix, and the reason `_passes_for` carries a laziness correction.

    Edge cells repeat themselves outward under the roll. Paired with an edge conductance of zero
    (`_conductance`) the wrapped value is never read, so this is about keeping one rectangular array
    rather than about what lies outside the world.
    """
    return np.stack(
        (
            field,
            np.roll(field, -1, axis=0),
            np.roll(field, 1, axis=0),
            np.roll(field, 1, axis=1),
            np.roll(field, -1, axis=1),
        )
    )


def _conductance(terrain: Terrain, climb_penalty: float) -> np.ndarray:
    """``(4, height, width)`` float32: normalised passability from each cell to each neighbour.

    Passability falls exponentially in the *rise* from a cell to its neighbour, which is the
    direction a forager would walk to reach what that neighbour holds. Descent is free, exactly as
    it is in `core.behaviour.movement`: raising a body against gravity is work in a way that
    lowering it is not, and using the same asymmetry in both places is what stops perception and
    locomotion disagreeing about what a ridge is.

    Normalising each cell's four weights to sum to one is what makes the relaxation contractive,
    and therefore what makes `spread` converge rather than accumulate without bound. It also makes
    the operator blind to a world's absolute elevation: only differences between neighbours enter.
    """
    heights = terrain.heights.astype(np.float32)
    rise = np.maximum(_neighbours(heights) - heights, 0.0)
    # Plane 0 is the cell itself, whose rise against itself is zero, so it conducts fully — which
    # is what makes the walk lazy (see `_neighbours`).
    passable = np.exp(-climb_penalty * rise, dtype=np.float32)

    # The world's rim conducts nothing inward from outside: a signal reflects rather than draining
    # off the edge of the map, which would make the border read as emptier than it is and push
    # grazers inward by an artifact of the grid.
    #
    # **The reflected weight returns to the cell itself, not to its other neighbours**, and that is
    # not cosmetic. Spreading it sideways breaks the operator's symmetry at the rim — an interior
    # cell sends a fifth of itself to its edge neighbour while that neighbour sends a quarter back —
    # so the walk stops being doubly stochastic and piles up along the boundary. Measured: with
    # sideways redistribution, a source two cells from a corner left the *corner* holding more than
    # the source's own cell, 0.836 against 0.755, so a forager standing there read a gradient
    # pointing further into the corner and would have walked into it and stayed. Returning the
    # weight to self is the standard reflecting walk — an outward step is refused and the walker
    # stays put — which keeps the flat-ground operator symmetric and its stationary distribution
    # uniform.
    for plane, rim in ((1, np.s_[-1, :]), (2, np.s_[0, :]), (3, np.s_[:, 0]), (4, np.s_[:, -1])):
        passable[0][rim] += passable[plane][rim]
        passable[plane][rim] = 0.0

    # Every cell conducts to itself, so the total is never zero and no guard is needed here.
    return passable / passable.sum(axis=0)
