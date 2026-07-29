"""Water: flow accumulation and depression pooling derived from the terrain heightmap.

Water is a consequence of terrain, not authored (CLAUDE.md §2.6): every cell's elevation decides
whether it drains through a channel, pools into a lake, or stays dry, so a terraforming
intervention changes hydrology for free by changing the height field underneath it. Water is a
static field derived from a `Terrain` snapshot — nothing here advances per tick; it is recomputed
whenever the terrain changes.
"""

from __future__ import annotations

import heapq

import numpy as np

from core.world.terrain import Terrain

# 8-connected neighbour offsets (row, col): the standard D8 flow model in terrain hydrology.
_NEIGHBOR_OFFSETS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)


class Water:
    """Derived hydrology: filled heights, flow routing, drainage, and lake depth.

    depth:              (height, width) float32, world units — the unit elevation is in, since
                         depth is a difference of two elevations (#112). Standing water above
                         the original terrain; 0 where the ground is dry.
    flow_direction:      (height, width) int8. Index into `_NEIGHBOR_OFFSETS` naming the downhill
                         neighbour each cell drains into; -1 where a cell has no lower neighbour,
                         which happens at map-edge outlets (water leaves the world there) and at
                         the exact center of a perfectly flat plateau.
    flow_accumulation:   (height, width) float32, cell-count. Number of cells, including itself,
                         whose drainage passes through this cell — read by
                         `core.ecology.plants` as its soil-moisture proxy, and a proxy for
                         channel width that a future river-rendering system can threshold.

    Constructed either by `generate()` from a `Terrain`, or directly from these three arrays plus
    `cell_size` when restoring from a snapshot — mirroring `Terrain`'s own constructor, so a
    snapshot system (owned by a separate issue) can round-trip this state without this module
    knowing anything about persistence.
    """

    def __init__(
        self,
        depth: np.ndarray,
        flow_direction: np.ndarray,
        flow_accumulation: np.ndarray,
        cell_size: float,
    ) -> None:
        depth = np.asarray(depth, dtype=np.float32)
        flow_direction = np.asarray(flow_direction, dtype=np.int8)
        flow_accumulation = np.asarray(flow_accumulation, dtype=np.float32)
        if depth.ndim != 2:
            raise ValueError("depth must be a 2D grid")
        if flow_direction.shape != depth.shape or flow_accumulation.shape != depth.shape:
            raise ValueError("depth, flow_direction, and flow_accumulation must share a shape")
        if cell_size <= 0:
            raise ValueError("cell_size must be positive")

        self.depth = depth
        self.flow_direction = flow_direction
        self.flow_accumulation = flow_accumulation
        self.cell_size = float(cell_size)

    @classmethod
    def generate(cls, terrain: Terrain) -> Water:
        """Derive drainage and pooling from a terrain height field.

        Depressions are filled to their spill elevation with a priority-flood fill (Barnes,
        Lehman & Mulla 2014), which guarantees a monotonically non-increasing path from every
        cell to the map edge. One consequence of that guarantee, proved in `_fill_depressions`,
        is that the terrain's global maximum is never flooded — the concrete property the water
        pooling test asserts.
        """
        filled = _fill_depressions(terrain.heights)
        depth = filled - terrain.heights
        # Filling only ever raises a cell (filled >= original); clip the sub-epsilon negative
        # values float subtraction can otherwise leave behind at cells filled to themselves.
        depth = np.maximum(depth, 0.0)
        flow_direction = _flow_directions(filled, terrain.heights)
        flow_accumulation = _flow_accumulation(filled, terrain.heights, flow_direction)
        return cls(depth, flow_direction, flow_accumulation, terrain.cell_size)

    def depth_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Bilinearly interpolated standing-water depth (world units) at world positions."""
        return _bilinear_sample(self.depth, x, y, self.cell_size)

    def is_drinkable_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Whether a position has standing water a thirsty creature could drink.

        No salinity or contamination model exists yet, so "there is water here" and "it is
        drinkable" are presently the same predicate. This is exposed as its own method, distinct
        from `depth_at`, because the thirst drive (CLAUDE.md §2.5) asks "can I drink here", not
        "how deep is it" — when a non-drinkable-water case exists (e.g. a sea), it changes this
        one method instead of every caller re-deriving the threshold.
        """
        return self.depth_at(x, y) > 0.0


def _fill_depressions(heights: np.ndarray) -> np.ndarray:
    """Priority-flood fill: raise every interior cell to the lowest elevation reachable from the
    map edge without ever descending, so every depression fills exactly to its spill point.

    The map boundary is treated as open drainage — the edge of the simulated world, not a wall —
    so boundary cells seed the flood at their own (unraised) elevation.

    Proof that a cell `p` at the global maximum is never flooded: every cell's filled value is
    bounded above by `max(heights)`, by induction over the flood order (a boundary seed starts at
    `original <= max(heights)`, and each subsequent cell's filled value is
    `max(original, predecessor_filled)`, both terms `<= max(heights)`). So every neighbour of `p`
    has `filled <= max(heights) == original[p]`, which makes `p`'s own value
    `max(original[p], min_neighbour_filled) == original[p]` — unraised.
    """
    rows, cols = heights.shape
    filled = np.asarray(heights, dtype=np.float64).copy()
    visited = np.zeros((rows, cols), dtype=bool)

    heap: list[tuple[float, int, int]] = []
    for r in range(rows):
        for c in (0, cols - 1):
            heapq.heappush(heap, (float(filled[r, c]), r, c))
            visited[r, c] = True
    for c in range(1, cols - 1):
        for r in (0, rows - 1):
            heapq.heappush(heap, (float(filled[r, c]), r, c))
            visited[r, c] = True

    while heap:
        elevation, r, c = heapq.heappop(heap)
        for dr, dc in _NEIGHBOR_OFFSETS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or visited[nr, nc]:
                continue
            visited[nr, nc] = True
            filled[nr, nc] = max(filled[nr, nc], elevation)
            heapq.heappush(heap, (float(filled[nr, nc]), nr, nc))

    return filled.astype(np.float32)


def _flow_directions(filled: np.ndarray, original: np.ndarray) -> np.ndarray:
    """Steepest-descent neighbour on the filled surface, vectorized across the 8 D8 directions.

    Ties on filled elevation (a flat, pooled lake surface) are broken toward the lower original
    elevation, so flow crossing a lake still drains toward its deepest point rather than stalling
    on the first tied neighbour scanned. A cell keeps direction -1 only if no neighbour is lower
    on both keys: a genuine map-edge outlet, or the exact low point of a perfectly flat plateau.
    """
    rows, cols = filled.shape
    filled = filled.astype(np.float64)
    original = original.astype(np.float64)
    padded_filled = np.pad(filled, 1, constant_values=np.inf)
    padded_original = np.pad(original, 1, constant_values=np.inf)

    best_filled = filled.copy()
    best_original = original.copy()
    direction = np.full((rows, cols), -1, dtype=np.int64)

    for idx, (dr, dc) in enumerate(_NEIGHBOR_OFFSETS):
        neighbor_filled = padded_filled[1 + dr : 1 + dr + rows, 1 + dc : 1 + dc + cols]
        neighbor_original = padded_original[1 + dr : 1 + dr + rows, 1 + dc : 1 + dc + cols]
        better = (neighbor_filled < best_filled) | (
            (neighbor_filled == best_filled) & (neighbor_original < best_original)
        )
        direction = np.where(better, idx, direction)
        best_filled = np.where(better, neighbor_filled, best_filled)
        best_original = np.where(better, neighbor_original, best_original)

    return direction.astype(np.int8)


def _flow_accumulation(
    filled: np.ndarray, original: np.ndarray, flow_direction: np.ndarray
) -> np.ndarray:
    """Cells draining through each cell, including itself.

    Processing order is highest-filled-first, tied by highest-original-first — the same key
    `_flow_directions` uses to route flow — so every cell's incoming contributions are finalized
    before it hands its total to whichever neighbour it drains into.
    """
    rows, cols = filled.shape
    order = np.lexsort(
        (-original.ravel().astype(np.float64), -filled.ravel().astype(np.float64))
    )
    accumulation = np.ones((rows, cols), dtype=np.float64)
    for flat_index in order:
        r, c = divmod(int(flat_index), cols)
        idx = flow_direction[r, c]
        if idx == -1:
            continue
        dr, dc = _NEIGHBOR_OFFSETS[idx]
        accumulation[r + dr, c + dc] += accumulation[r, c]
    return accumulation.astype(np.float32)


def _bilinear_sample(
    field: np.ndarray, x: np.ndarray, y: np.ndarray, cell_size: float
) -> np.ndarray:
    """Bilinearly interpolate `field` at continuous world positions, matching
    `Terrain._sample`'s semantics (same grid convention, same out-of-bounds behaviour) without
    depending on a `Terrain` instance, since `Water` must be reconstructable from its own arrays
    alone for snapshot restore.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rows, cols = field.shape
    world_width = (cols - 1) * cell_size
    world_height = (rows - 1) * cell_size
    in_bounds_x = np.all((x >= 0) & (x <= world_width))
    in_bounds_y = np.all((y >= 0) & (y <= world_height))
    if not (in_bounds_x and in_bounds_y):
        raise ValueError("position outside water field bounds")

    gx = x / cell_size
    gy = y / cell_size
    col0 = np.floor(gx).astype(np.int64)
    row0 = np.floor(gy).astype(np.int64)
    col1 = np.minimum(col0 + 1, cols - 1)
    row1 = np.minimum(row0 + 1, rows - 1)
    fx = gx - col0
    fy = gy - row0

    top = field[row0, col0] * (1 - fx) + field[row0, col1] * fx
    bottom = field[row1, col0] * (1 - fx) + field[row1, col1] * fx
    return top * (1 - fy) + bottom * fy
