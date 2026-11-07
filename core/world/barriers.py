"""Barriers: edges of the grid nothing can cross (CLAUDE.md §2.6, issue #27).

A fence is the intervention §2.5 calls the most rewarding in the game — isolate a population, watch
it diverge — and it is the player's constructed counterpart to a mountain range. What makes that
work is already built and was built *for* this: `CostAwareDiffusion` spreads over a **neighbour
graph** rather than by a kernel, so a signal that cannot pass an edge arrives behind the barrier
only by coming round the end of it. §2.5 says so outright — *"that is what will make a fence an
intervention rather than a multiplier"* — and this is the thing that was missing.

## A barrier is an edge, not a cell

A fence stands *between* two cells; it does not occupy one. Represented as blocked cells instead, a
one-cell-thick wall would take a cell out of the world — ground that grows nothing, that nothing can
stand on, and that an animal walking along the fence would have to path around. Ecologically a fence
line is not a strip of desert, and arithmetically a blocked cell would have to be excluded from the
nutrient ledger, from grazing, and from every field's normalisation.

Two boolean grids hold every edge exactly once:

    blocked_north[r, c]   the edge between (r, c) and (r - 1, c)
    blocked_west[r, c]    the edge between (r, c) and (r, c - 1)

A cell's south edge is its neighbour's north edge, and its east edge its neighbour's west, so there
is no second place to write the same fact and no way for the two to disagree. Row 0 has no north
neighbour and column 0 no west one, so those entries exist and stay false — kept rather than
special-cased, because a `(h, w)` array indexes the same way as every other field in the world.

## Terrain is a cost; a fence is a refusal

§2.5 settles that *"impassable ground is expressed entirely through what it costs"*, and a fence is
the one thing in the world that is **not**. That is a deliberate distinction rather than an
exception to forget:

- A ridge is expensive. An animal desperate enough — starving, with the far side visibly better —
  can pay for it and cross, and whether it does is exactly the selection pressure terrain exists to
  apply. That is why `climb_cost` is a price and not a wall.
- A fence is a refusal. Priced instead of refused, a fence would be a *very steep hill*: a
  sufficiently desperate animal crosses it, sufficiently many animals cross it eventually, and the
  isolation the player paid for leaks. Worse, the animal that tried would empty its pool against the
  wire and die there, so a fence would read as a killing field rather than a boundary.

So `Movement` stops an animal *at* a blocked edge without charging it for the crossing it did not
make. Where that is along a straight line is `reach`, the one crossing finder: `Movement` cuts each
step there before pricing it, and `Behaviour` reads each candidate heading there (#278), so the
ground an animal scores is the ground it would stand on.

## A sealed fence partitions the grid into regions

`enclose` always draws a closed rectangle, so every fence that exists seals what it surrounds, and
the grid falls into **regions**: sets of cells joined by open edges. Nothing in one region can reach
another, so "are these two places on the same side of every fence" is one integer comparison
instead of a walk (#169). `regions` labels them, recomputed only when `revision` moves, and these
ask it:

- two animals on opposite sides of the wire cannot touch, so cannot mate or strike (#276);
- each region carries its own scent field, spread over its bounding box (`region_boxes`, #139).

Connectivity is through the **four** edge neighbours, which is what `reach` walks for `Movement`: it
crosses one edge at a time, so a diagonal through a fence corner is two crossings and is refused
if either is. A fence with a gap in it would leave its two sides one region, and every reader then
behaves exactly as if it were not there — which is right, because an animal can walk round.

## What a fence does not stop

**Terrain-shaped scent.** Since #139 no smell crosses a fence, because each sealed region carries
its own field. But a region's field is spread over its bounding box, not over its own cells, so an
irregular region's plume shortcuts across the area of another: animals outside a pen still smell
each other *across* it, though nobody in it. Scent stays blind to terrain, deferred to wind (#97).
"""

from __future__ import annotations

import numpy as np

from core.world.terrain import Terrain, to_next_grid_line

# How far short of a blocked edge a body is stopped, as a fraction of a cell. Small enough to be
# invisible ecologically and large enough to survive the float32 the position columns are stored in:
# a cell is 1 world unit in the shipped world, so this is a thousandth of one, against a float32
# resolution near 20 of about 2e-6.
WIRE_MARGIN = 1e-3


class Barriers:
    """Which grid edges are impassable, over one world's terrain.

    blocked_north: `(h, w) bool` — the edge between `(r, c)` and `(r - 1, c)` is blocked.
    blocked_west: `(h, w) bool` — the edge between `(r, c)` and `(r, c - 1)` is blocked.

    Mutable, and deliberately: a fence is built and removed while a world runs, which is the whole
    point of it being an intervention. Everything derived from it — the diffusion conductance —
    therefore has to be rebuilt when it changes, and `revision` is what lets a reader notice.
    """

    # Declared rather than left to inference, as `Plants` and `Carrion` are and for the same reason:
    # `np.zeros(shape)` with a statically unknown shape resolves to the stubs' 1-D overload.
    blocked_north: np.ndarray
    blocked_west: np.ndarray

    def __init__(self, terrain: Terrain) -> None:
        self.terrain = terrain
        self.blocked_north = np.zeros(terrain.heights.shape, dtype=np.bool_)
        self.blocked_west = np.zeros(terrain.heights.shape, dtype=np.bool_)
        # Bumped by every edit. Whatever caches a derivation of these grids compares it rather than
        # being told to invalidate — a stamp a writer must remember to set is a stamp that is
        # eventually not set, and the failure is silent (§8.7): the field simply keeps routing
        # through a fence that is standing.
        self.revision = 0
        self._regions = np.zeros(terrain.heights.shape, dtype=np.int32)
        self._n_regions = 1
        self._boxes = _bounding_boxes(self._regions, 1)
        self._wires = _wire_table(self.blocked_north, self.blocked_west)
        self._regions_revision = self.revision

    @property
    def any_blocked(self) -> bool:
        """Whether anything is fenced at all, so an unfenced world pays for nothing."""
        return bool(self.blocked_north.any() or self.blocked_west.any())

    @property
    def regions(self) -> np.ndarray:
        """`(h, w) int32`: the sealed region each cell lies in, numbered from 0 (module docstring).

        Two cells share a label exactly when a path of open edges joins them. Numbered in order of
        each region's first cell in row-major order, so an unfenced world is all 0 and the region
        holding cell `(0, 0)` is always 0. Recomputed when `revision` has moved since the last
        call, and otherwise returned from the cache: every reader asks once per tick and fences go
        up far less often than that.

        Read-only, because a reader that wrote into it would change every other reader's answer
        until the next fence edit.
        """
        self._refresh_regions()
        return self._regions

    @property
    def n_regions(self) -> int:
        """How many sealed regions the grid holds; 1 for an unfenced world."""
        self._refresh_regions()
        return self._n_regions

    @property
    def region_boxes(self) -> np.ndarray:
        """`(n_regions, 4) int64`: each region's bounding box in cells, by label.

        Each row is `[row_low, row_high, col_low, col_high]`, half-open, so `regions[r0:r1, c0:c1]`
        is the box. A sealed rectangle's box is the rectangle itself; an unfenced world's one box
        is the whole grid. Read-only, as `regions` is.
        """
        self._refresh_regions()
        return self._boxes

    def _refresh_regions(self) -> None:
        if self._regions_revision != self.revision:
            self._regions, self._n_regions = _label_regions(self.blocked_north, self.blocked_west)
            self._boxes = _bounding_boxes(self._regions, self._n_regions)
            self._wires = _wire_table(self.blocked_north, self.blocked_west)
            self._regions.flags.writeable = False
            self._regions_revision = self.revision

    def same_region(
        self, from_x: np.ndarray, from_y: np.ndarray, to_x: np.ndarray, to_y: np.ndarray
    ) -> np.ndarray:
        """`bool`, broadcast over the arguments: whether each `(from)` and `(to)` share a region.

        World units. A position is in the region of the cell containing it, and `Movement` stops
        an animal short of the wire (#27), so an animal is always in the region it was penned in.
        """
        regions = self.regions
        from_row, from_col = self.terrain.cell_indices(from_x, from_y)
        to_row, to_col = self.terrain.cell_indices(to_x, to_y)
        return np.asarray(regions[from_row, from_col] == regions[to_row, to_col], dtype=np.bool_)

    def region_of(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """`int32`, shaped like `x`: the sealed region each world position lies in."""
        rows, cols = self.terrain.cell_indices(x, y)
        return self.regions[rows, cols]

    def enclose(self, min_x: float, min_y: float, max_x: float, max_y: float) -> int:
        """Fence the perimeter of the world-unit rectangle; return how many edges that blocked.

        The *perimeter*, not the area: enclosing is what isolates a population, and blocking every
        edge inside the rectangle would not fence a herd in, it would freeze one in place.

        Returns the count so a caller can tell a fence that did something from one that did not —
        a rectangle outside the world, or one thinner than a cell, blocks nothing, and §8.7 prefers
        that be visible to the player rather than silently absorbed.
        """
        rows, cols = self.blocked_north.shape
        cell = self.terrain.cell_size
        # Half-open in cells: the rectangle covers cells [low, high), so its north edge is the
        # north edge of the first row inside it and its south edge is the north edge of the first
        # row outside. Clipped to the grid, so a rectangle running off the map fences the part of
        # its perimeter that exists rather than raising — a fence against the world's rim is a
        # legitimate thing to draw, and the rim already stops everything anyway.
        low_col = int(np.clip(np.floor(min_x / cell + 0.5), 0, cols))
        high_col = int(np.clip(np.floor(max_x / cell + 0.5), 0, cols))
        low_row = int(np.clip(np.floor(min_y / cell + 0.5), 0, rows))
        high_row = int(np.clip(np.floor(max_y / cell + 0.5), 0, rows))
        if low_col >= high_col or low_row >= high_row:
            return 0

        before = int(self.blocked_north.sum() + self.blocked_west.sum())
        self.blocked_north[low_row, low_col:high_col] = True
        if high_row < rows:
            self.blocked_north[high_row, low_col:high_col] = True
        self.blocked_west[low_row:high_row, low_col] = True
        if high_col < cols:
            self.blocked_west[low_row:high_row, high_col] = True

        self.revision += 1
        return int(self.blocked_north.sum() + self.blocked_west.sum()) - before

    def clear(self) -> None:
        """Take every fence down. The counterpart to `enclose`, so a player can undo an isolation
        they no longer want — and what #33's standing policies will hold open or closed."""
        self.blocked_north[...] = False
        self.blocked_west[...] = False
        self.revision += 1

    def reach(
        self,
        x: np.ndarray,
        y: np.ndarray,
        unit_x: np.ndarray,
        unit_y: np.ndarray,
        distance: np.ndarray,
    ) -> np.ndarray:
        """float64 world units, the arguments' broadcast shape: how far along each straight ray a
        body gets before a blocked edge stops it, or `distance` where none lies on the ray.

        x, y: where each ray starts, world units, inside the world. unit_x, unit_y: its direction,
        a unit vector. distance: its length, world units, ending inside the world. Broadcast
        together, so `Behaviour` can hand one start per animal, `(n, 1)`, for its `(n, k)` rays.

        **The one crossing finder** (#278). `Movement` asks it where a step ends, and `Behaviour`
        where a candidate heading is read, so the ground an animal scores is exactly the ground it
        would be standing on. Two finders would be two answers to one question, and a candidate a
        hair past the point the walk stops would read the far side of the wire.

        **It walks the cell edges one crossing at a time.** A cell is the square centred on an
        elevation node, so its edges sit on the half-integer lattice (#27): a ray tested only at
        its ends steps clean over a wire in its middle, and one cutting a pen's corner has both ends
        in one region. At each crossing it asks `crossing_blocked` about a probe a margin past it,
        which is unambiguously in the cell being entered.

        **It stops `WIRE_MARGIN` of a cell short of the line, never on it.** A cell owns the
        half-open span around its centre, so a position exactly on a boundary indexes to the cell
        *beyond* it, and stopping there would read as already through. Measured before the margin
        existed, a 39x39 pen lost 3 of 20 penned animals in 100 ticks and gained strangers at the
        same rate.

        **Only rays that could meet a wire are walked**, found by two box queries on a summed-area
        table of the blocked edges, each four lookups. A straight ray visits only cells inside the
        box its two end cells span, and crosses only edges those cells own, so a ray whose box holds
        no blocked edge cannot meet one. The first query (`near_wire`) asks it of each *start*,
        with a box as wide as the longest ray, which is per animal rather than per candidate; the
        second asks it of each surviving ray's own box. Walking every candidate instead is what #169 measured at
        2.7-7.6 s per tick (docs/spikes/fence-perception.md).
        """
        shape = np.broadcast_shapes(
            np.shape(x), np.shape(y), np.shape(unit_x), np.shape(unit_y), np.shape(distance)
        )
        free = np.array(np.broadcast_to(distance, shape), dtype=np.float64)
        if not self.any_blocked or free.size == 0:
            return free
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        # Which starts have any wire within the longest ray's reach, at the starts' own shape.
        near = self.near_wire(x, y, float(free.max()))
        rays = np.flatnonzero(np.broadcast_to(near, shape) & (free > 0.0))
        if rays.size == 0:
            return free
        at = np.unravel_index(rays, shape)

        def gather(values: np.ndarray) -> np.ndarray:
            return np.broadcast_to(np.asarray(values, dtype=np.float64), shape)[at]

        ray_x, ray_y = gather(x), gather(y)
        ray_ux, ray_uy = gather(unit_x), gather(unit_y)
        length = free.reshape(-1)[rays]
        # Which of those rays have a wire inside the box their own two end cells span. The end is
        # clipped into the world, since it may sit a float hair outside it.
        rows, cols = self.terrain.cell_indices(ray_x, ray_y)
        end_rows, end_cols = self.terrain.cell_indices(
            np.clip(ray_x + ray_ux * length, 0.0, self.terrain.world_width),
            np.clip(ray_y + ray_uy * length, 0.0, self.terrain.world_height),
        )
        crossing = self._holds_wire(
            np.minimum(rows, end_rows),
            np.maximum(rows, end_rows),
            np.minimum(cols, end_cols),
            np.maximum(cols, end_cols),
        )
        if crossing.any():
            walked = rays[crossing]
            free.reshape(-1)[walked] = self._walk_to_wire(
                ray_x[crossing], ray_y[crossing], ray_ux[crossing], ray_uy[crossing],
                length[crossing],
            )
        return free

    def near_wire(self, x: np.ndarray, y: np.ndarray, within: float) -> np.ndarray:
        """`bool`, shaped like `x`: whether a blocked edge may lie within `within` world units of
        each position. Measured per axis in whole cells, so it may say yes for a wire slightly
        further off, and never no for one inside that range.

        `reach`'s first filter, and `Behaviour`'s before it builds any ray at all: an animal with
        no wire within one `look_ahead` has no candidate a wire could stop, and most animals in a
        fenced world are such animals.
        """
        if not self.any_blocked:
            return np.zeros(np.shape(x), dtype=np.bool_)
        self._refresh_regions()
        rows, cols = self.terrain.cell_indices(x, y)
        radius = int(np.ceil(within / self.terrain.cell_size))
        return self._holds_wire(rows - radius, rows + radius, cols - radius, cols + radius)

    def _holds_wire(
        self, row_low: np.ndarray, row_high: np.ndarray, col_low: np.ndarray, col_high: np.ndarray
    ) -> np.ndarray:
        """`bool`, shaped like the bounds: whether any cell in rows `[row_low, row_high]` and
        columns `[col_low, col_high]`, inclusive and clipped to the grid, owns a blocked edge."""
        height, width = self.blocked_north.shape
        row_low = np.clip(row_low, 0, height)
        row_high = np.clip(row_high + 1, 0, height)
        col_low = np.clip(col_low, 0, width)
        col_high = np.clip(col_high + 1, 0, width)
        wires = self._wires
        count = (
            wires[row_high, col_high]
            - wires[row_low, col_high]
            - wires[row_high, col_low]
            + wires[row_low, col_low]
        )
        return np.asarray(count > 0, dtype=np.bool_)

    def _walk_to_wire(
        self,
        x: np.ndarray,
        y: np.ndarray,
        unit_x: np.ndarray,
        unit_y: np.ndarray,
        distance: np.ndarray,
    ) -> np.ndarray:
        """`reach` for rays that may meet a wire: every cell edge each one crosses, in order.

        The iteration count is per call, not per ray: each pass advances every unfinished ray to
        its own next crossing and drops the finished ones, so the loop runs as many times as the
        longest ray needs and each pass costs only what is still walking.
        """
        cell = self.terrain.cell_size
        margin = WIRE_MARGIN * cell
        world_width, world_height = self.terrain.world_width, self.terrain.world_height
        end_x = x + unit_x * distance
        end_y = y + unit_y * distance
        free = distance.copy()
        travelled = np.zeros_like(distance)
        walking = np.arange(distance.size)
        # A ray of length d crosses at most ceil(d / cell) edges on each axis; +2 covers the
        # partial cell at each end.
        passes = 2 * int(np.ceil(float(distance.max()) / cell)) + 2

        for remaining_passes in range(passes, 0, -1):
            if walking.size == 0:
                break
            ux, uy = unit_x[walking], unit_y[walking]
            d, t = distance[walking], travelled[walking]
            start_x, start_y = x[walking], y[walking]
            at_x = start_x + ux * t
            at_y = start_y + uy * t
            # Cell edges are the node lattice shifted by half a cell (`Terrain.cell_indices`).
            to_edge = np.minimum(
                to_next_grid_line(at_x + 0.5 * cell, ux, cell),
                to_next_grid_line(at_y + 0.5 * cell, uy, cell),
            )
            left = d - t
            # On the last permitted pass, finish in one segment: only reachable if float slivers
            # at a line consumed passes the bound did not expect.
            step = left if remaining_passes == 1 else np.minimum(to_edge, left)

            # Asked from the segment's midpoint to a probe just past its end. Neither end of the
            # segment will do as the first point: each sits on a cell edge, which indexes to the
            # cell on its far side, so a ray travelling toward lower indices would be asked about
            # the edge it has just crossed rather than the one ahead. The midpoint is unambiguously
            # in the cell being crossed, and the probe in the cell being entered. The probe snaps
            # to the ray's end once past it, so it never leaves the ray, and both are clipped into
            # the world: a float hair outside it is enough for `cell_indices` to refuse, and the
            # rim stops everything anyway.
            middle = t + 0.5 * step
            probe = t + step + margin
            past = probe >= d
            probe_x = np.where(past, end_x[walking], start_x + ux * probe)
            probe_y = np.where(past, end_y[walking], start_y + uy * probe)
            refused = self.crossing_blocked(
                np.clip(start_x + ux * middle, 0.0, world_width),
                np.clip(start_y + uy * middle, 0.0, world_height),
                np.clip(probe_x, 0.0, world_width),
                np.clip(probe_y, 0.0, world_height),
            )
            reached = t + np.where(refused, np.maximum(step - margin, 0.0), step)
            travelled[walking] = reached
            done = refused | (reached >= d)
            free[walking[done]] = np.where(refused[done], reached[done], d[done])
            walking = walking[~done]

        return free

    def crossing_blocked(
        self, from_x: np.ndarray, from_y: np.ndarray, to_x: np.ndarray, to_y: np.ndarray
    ) -> np.ndarray:
        """`(n,) bool`: whether the move from each `(from)` to its `(to)` crosses a blocked edge.

        Both points are world units. Only the *cell* each falls in matters, because a barrier lives
        on a cell edge: a move within one cell crosses nothing, and a move between neighbours
        crosses exactly the edge between them.

        **Diagonal moves are treated as blocked if either component edge is**, which is the
        conservative reading and the right one: a fence corner has no gap in it, and permitting the
        diagonal would let animals leak through the join one at a time. `Movement` walks axis
        crossings one at a time (#113) so it never asks about a diagonal, but this method is the
        general query and a caller sampling coarser must not find a hole.
        """
        from_row, from_col = self.terrain.cell_indices(from_x, from_y)
        to_row, to_col = self.terrain.cell_indices(to_x, to_y)

        # The *southern* of two rows owns the edge between them, since `blocked_north[r]` is the
        # edge above row r; likewise the eastern of two columns owns the edge between them. Taking
        # the max index therefore reads the one entry holding the fact, whichever way the move runs.
        edge_row = np.maximum(from_row, to_row)
        edge_col = np.maximum(from_col, to_col)

        # A diagonal has two component edges and no defined order, so it is blocked if *either* is:
        # a fence corner has no gap in it, and permitting the diagonal would let animals leak
        # through the join. Both columns of the horizontal edge are checked for the same reason.
        crossed_north = (from_row != to_row) & (
            self.blocked_north[edge_row, from_col] | self.blocked_north[edge_row, to_col]
        )
        crossed_west = (from_col != to_col) & (
            self.blocked_west[from_row, edge_col] | self.blocked_west[to_row, edge_col]
        )
        return np.asarray(crossed_north | crossed_west, dtype=np.bool_)


def _wire_table(blocked_north: np.ndarray, blocked_west: np.ndarray) -> np.ndarray:
    """`(h + 1, w + 1) int64`, read-only: a summed-area table of the blocked edges, by owning cell.

    `table[r, c]` counts the blocked edges owned by the cells above and left of `(r, c)`, so the
    count in any box of cells is four lookups. `reach` asks it whether a ray could meet a wire.
    """
    owned = blocked_north.astype(np.int64) + blocked_west.astype(np.int64)
    table = np.zeros((owned.shape[0] + 1, owned.shape[1] + 1), dtype=np.int64)
    table[1:, 1:] = owned.cumsum(axis=0).cumsum(axis=1)
    table.flags.writeable = False
    return table


def _bounding_boxes(regions: np.ndarray, count: int) -> np.ndarray:
    """`(count, 4) int64`, read-only: `[row_low, row_high, col_low, col_high)` of each label."""
    rows, cols = regions.shape
    labels = regions.ravel()
    row_of = np.repeat(np.arange(rows, dtype=np.int64), cols)
    col_of = np.tile(np.arange(cols, dtype=np.int64), rows)
    boxes = np.empty((count, 4), dtype=np.int64)
    boxes[:, 0] = rows
    boxes[:, 1] = -1
    boxes[:, 2] = cols
    boxes[:, 3] = -1
    np.minimum.at(boxes[:, 0], labels, row_of)
    np.maximum.at(boxes[:, 1], labels, row_of)
    np.minimum.at(boxes[:, 2], labels, col_of)
    np.maximum.at(boxes[:, 3], labels, col_of)
    boxes[:, 1] += 1
    boxes[:, 3] += 1
    boxes.flags.writeable = False
    return boxes


def _label_regions(blocked_north: np.ndarray, blocked_west: np.ndarray) -> tuple[np.ndarray, int]:
    """`(h, w) int32` region labels and their count: the connected components of the open edges.

    Every cell starts labelled by its own flat index, and labels are then lowered to the smallest
    one reachable, alternating two sweeps until nothing changes: each **run** of a row between
    blocked west edges takes its minimum, then each run of a column between blocked north edges
    does. A run is open along its whole length, so it is one region and its minimum is a label the
    whole run can reach. Every path is a sequence of row moves and column moves, so the sweeps
    converge on each region's smallest index, in as many rounds as the path that needs most turns —
    two or three for a rectangle, where a flood fill one cell per pass would need the region's
    diameter.

    Vectorised throughout (§2.3): one `minimum.reduceat` per sweep over the flattened grid, since
    the runs of a row-major array are contiguous, and the transpose for the column runs.
    """
    rows, cols = blocked_north.shape
    if not (blocked_north.any() or blocked_west.any()):
        return np.zeros((rows, cols), dtype=np.int32), 1

    # A run starts at the first cell of every line and at every blocked edge along it.
    row_starts = blocked_west.copy()
    row_starts[:, 0] = True
    col_starts = blocked_north.T.copy()
    col_starts[:, 0] = True
    row_first = np.flatnonzero(row_starts)
    col_first = np.flatnonzero(col_starts)
    # Which run each cell belongs to, so a run's minimum is gathered back onto its cells.
    row_run = np.cumsum(row_starts.ravel()) - 1
    col_run = np.cumsum(col_starts.ravel()) - 1

    labels = np.arange(rows * cols, dtype=np.int64)
    while True:
        lowered = np.minimum.reduceat(labels, row_first)[row_run]
        across = np.ascontiguousarray(lowered.reshape(rows, cols).T).ravel()
        across = np.minimum.reduceat(across, col_first)[col_run]
        lowered = np.ascontiguousarray(across.reshape(cols, rows).T).ravel()
        if np.array_equal(lowered, labels):
            break
        labels = lowered

    # Renumber 0..k-1 in order of each region's smallest index, which is its first cell.
    unique, compact = np.unique(labels, return_inverse=True)
    return compact.reshape(rows, cols).astype(np.int32), int(unique.size)
