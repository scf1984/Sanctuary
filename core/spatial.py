"""Uniform-grid spatial index for neighbour queries, z-capable from day one (CLAUDE.md §2.6, §4).

Sensing, feeding, predation, and mating all need "what is near me" every tick. This module answers
that with a grid over a store's ``(x, y, z)`` columns, bucketed at a cell size sized to the query
radius, so a neighbour search only has to look at a small fixed neighbourhood of cells regardless
of how many entities exist elsewhere in the world — never the whole population.

Movement is surface-locked until #43 lands (CLAUDE.md §2.6), so today every entity's z happens to
sit at terrain height. That is a fact about *movement*, not about this index: cells are bucketed on
all three axes unconditionally, and queries measure true 3D Euclidean distance, so nothing here
hardcodes ``z == terrain height`` in a way that would need to change the day flight or diving
movement starts writing non-surface z values.

Pair-enumeration semantics: this index has no notion of a single, globally-deduplicated pair list.
``neighbors_of(observers, radius)`` answers one observer ``Selection`` at a time and returns the
union of every *other* indexed entity within ``radius`` of *any* observer in that selection,
excluding the observers themselves. The underlying relation is symmetric — if A is within radius of
B, B is within radius of A — but each direction is a separate call: querying with
``observers=selection_containing_A`` returns B, and a separate call with
``observers=selection_containing_B`` returns A. Nothing here collapses "A-B" against "B-A" into one
answer, because doing so would require an ordering over pairs this index does not impose; a caller
that wants deduplicated pairs (e.g. a mating system pairing each entity at most once) must impose
that ordering itself.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from core.selection import Selection

_CellKey = tuple[int, int, int]


class SpatialIndex:
    """Grid-bucketed neighbour queries over a store's position columns.

    cell_size: float, world units, > 0. Callers size this to the maximum sensing range across the
        population they intend to query (CLAUDE.md issue #11) — this module has no notion of genes
        or sensing and never computes that maximum itself; whichever future behaviour service owns
        sensing-range genes computes it and passes the result in here.

    Internal state — a row's bucket, not its position, is this index's source of truth between
    calls to rebuild()/update():
      _cell_of_row: row -> the (ix, iy, iz) cell key it was last bucketed under. Only present for
          rows currently indexed.
      _buckets: (ix, iy, iz) -> the set of rows currently bucketed there. A cell with no rows is
          removed rather than left as an empty set, so bucket count reflects occupied cells only.
    """

    def __init__(self, cell_size: float) -> None:
        if not cell_size > 0:
            raise ValueError("cell_size must be positive")
        self.cell_size = cell_size
        self._cell_of_row: dict[int, _CellKey] = {}
        self._buckets: dict[_CellKey, set[int]] = {}

    def rebuild(self, store: Any, population: Selection) -> None:
        """Discard the current grid and rebucket every row in ``population`` from scratch.

        store: any object exposing ``x``, ``y``, ``z`` as ``(capacity,) float32`` world-unit
            columns, e.g. an EntityStore. Rows outside ``population`` are not indexed and will
            never appear in a neighbour result, even if they hold live data in ``store``.
        """
        self._cell_of_row = {}
        self._buckets = {}
        rows = population.to_indices()
        for row, cell in zip(rows.tolist(), self._cell_keys(store, rows)):
            self._cell_of_row[row] = cell
            self._buckets.setdefault(cell, set()).add(row)

    def update(self, store: Any, population: Selection) -> None:
        """Reconcile the grid with the current position/population state.

        Only rows that changed — newly entered ``population``, no longer in it, or moved to a
        different cell — cause a bucket mutation; a row that neither moved cell nor changed
        membership costs one dict lookup. This is the incremental update issue #11 asks for in
        place of a full rebuild every tick, since most entities stay in the same cell most ticks.
        """
        rows = population.to_indices()
        current = set(rows.tolist())

        for row in [r for r in self._cell_of_row if r not in current]:
            self._discard(row)

        if len(rows) == 0:
            return
        for row, cell in zip(rows.tolist(), self._cell_keys(store, rows)):
            old_cell = self._cell_of_row.get(row)
            if old_cell == cell:
                continue
            if old_cell is not None:
                self._buckets[old_cell].discard(row)
                if not self._buckets[old_cell]:
                    del self._buckets[old_cell]
            self._cell_of_row[row] = cell
            self._buckets.setdefault(cell, set()).add(row)

    def neighbors_of(self, store: Any, observers: Selection, radius: float) -> Selection:
        """Every indexed row within ``radius`` of any row in ``observers``, excluding observers.

        radius: float, world units, > 0. Need not equal ``cell_size``; a radius larger than
        ``cell_size`` widens how many surrounding cells are searched, which costs more candidate
        rows to filter but stays correct.

        A row not currently indexed (never passed to rebuild()/update(), or since removed) is
        never returned and, if present in ``observers``, contributes no candidates for that
        observer — its position is unknown to this index.
        """
        if not radius > 0:
            raise ValueError("radius must be positive")

        capacity = observers.capacity
        result = np.zeros(capacity, dtype=np.bool_)
        observer_rows = observers.to_indices()
        if len(observer_rows) == 0:
            return Selection.from_mask(result)

        cell_radius = max(1, math.ceil(radius / self.cell_size))
        offsets = range(-cell_radius, cell_radius + 1)
        observer_cells = {
            self._cell_of_row[row] for row in observer_rows.tolist() if row in self._cell_of_row
        }
        candidate_rows: set[int] = set()
        for cx, cy, cz in observer_cells:
            for dx in offsets:
                for dy in offsets:
                    for dz in offsets:
                        bucket = self._buckets.get((cx + dx, cy + dy, cz + dz))
                        if bucket:
                            candidate_rows.update(bucket)
        candidate_rows.difference_update(observer_rows.tolist())
        if not candidate_rows:
            return Selection.from_mask(result)

        candidates = np.array(sorted(candidate_rows), dtype=np.int64)
        # Dense candidate-by-observer distance matrix: candidates are already narrowed to a small
        # cell neighbourhood by the grid above, and observer counts are typically small (one
        # predator, one mate search), so this stays a small matrix rather than an O(n^2) full scan.
        dx = store.x[candidates][:, None] - store.x[observer_rows][None, :]
        dy = store.y[candidates][:, None] - store.y[observer_rows][None, :]
        dz = store.z[candidates][:, None] - store.z[observer_rows][None, :]
        within_radius = (dx**2 + dy**2 + dz**2 <= radius**2).any(axis=1)
        result[candidates[within_radius]] = True
        return Selection.from_mask(result)

    def _discard(self, row: int) -> None:
        cell = self._cell_of_row.pop(row)
        bucket = self._buckets[cell]
        bucket.discard(row)
        if not bucket:
            del self._buckets[cell]

    def _cell_keys(self, store: Any, rows: np.ndarray) -> list[_CellKey]:
        ix = np.floor(store.x[rows] / self.cell_size).astype(np.int64)
        iy = np.floor(store.y[rows] / self.cell_size).astype(np.int64)
        iz = np.floor(store.z[rows] / self.cell_size).astype(np.int64)
        return list(zip(ix.tolist(), iy.tolist(), iz.tolist()))
