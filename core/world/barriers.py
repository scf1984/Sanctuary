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
make, and the walk it already does per cell crossing (#113) is where that lands — a fence needs no
second pass, because the code that would have to know about it is already visiting every edge.

## A sealed fence partitions the grid into regions

`enclose` always draws a closed rectangle, so every fence that exists seals what it surrounds, and
the grid falls into **regions**: sets of cells joined by open edges. Nothing in one region can reach
another, so "are these two places on the same side of every fence" is one integer comparison
instead of a walk (#169). `regions` labels them, recomputed only when `revision` moves, and these
ask it:

- a candidate heading across the wire offers nothing to any drive (#169), since the animal cannot
  get there;
- two animals on opposite sides of the wire cannot touch, so cannot mate or strike (#276);
- each region carries its own scent field, spread over its bounding box (`region_boxes`, #139).

Connectivity is through the **four** edge neighbours, which is what `Movement` walks (#113): it
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

from core.world.terrain import Terrain


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
