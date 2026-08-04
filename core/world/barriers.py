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

## What a fence does not stop

**Scent.** `core.ecology.cues` blurs with a separable box kernel rather than the neighbour-graph
walk, precisely because `sample_excluding_self` subtracts the exact diagonal of that blur and the
factorisation only works for a separable kernel. So an animal smells its neighbours through a fence,
and converting the cue field to a cost-aware operator is filed separately (#139). Worth knowing
before reading a fenced world's fear scores: the barrier is real for food, water and movement, and
not yet for smell.
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

    @property
    def any_blocked(self) -> bool:
        """Whether anything is fenced at all, so an unfenced world pays for nothing."""
        return bool(self.blocked_north.any() or self.blocked_west.any())

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
