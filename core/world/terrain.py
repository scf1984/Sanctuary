"""Heightmap terrain: a height field over the world grid plus derived slope and aspect.

Elevation is real from the start (CLAUDE.md §2.6): it drives movement cost, line-of-sight
occlusion, downhill water flow, and temperature by altitude, so climate zones fall out of
relief instead of being painted on. Generation is a pure function of a world seed so a world
can be recreated for testing even though the running simulation is not deterministic (§2.2).

**This module defines the world's one length unit** (#112). `cell_size` is world units per cell
edge and elevation is in those same units, so vertical and horizontal distance are directly
comparable and nothing needs a conversion factor between them. The unit is deliberately **not** a
physical one: grounding it invites Earth-calibrated constants — Earth's tropospheric lapse rate
was the first — that carry no meaning in a world whose relief is whatever its config says, and
cannot be tuned freely once written down. Every length in `core/` is in this unit, and
`tests/test_length_units.py` fails if any of them claims otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TerrainConfig:
    """Parameters for reproducible terrain generation.

    width, height: grid resolution in cells. World extent in world units is
        ``(width - 1) * cell_size`` by ``(height - 1) * cell_size`` — callers derive this from
        climate-zone variety and animal home range (§2.6), never from a fixed constant here.
    min_elevation, max_elevation: output height range, **world units** — the same unit as x, y,
        cell_size and speed (#112). Required rather than defaulted, because relief only means
        something against the extent above, which only the caller knows: the range this replaced
        defaulted to 0–1000 against a default `cell_size` of 1.0, i.e. relief a thousand cells
        tall, and nothing in the type system or the tests noticed. Choose it as a *ratio* to
        extent — around a tenth gives terrain a creature must climb but not a wall of cliffs —
        and rescale it when the world grows, since it is the ratio and not the magnitude that
        decides whether a ridge isolates a population (§2.6, #16).
    cell_size: world units per cell edge. Defaulting to 1.0 is a normalisation rather than a
        guess about scale — one cell *is* one world unit — which is why this length may carry a
        default where the two above may not.
    seed: generation is a pure function of this value; the same seed always yields the same
        height field.
    octaves: number of fractal noise layers combined to build relief. More octaves add
        finer detail on top of the same large-scale shape.
    persistence: amplitude falloff per octave, in (0, 1). Higher values weight finer octaves
        more heavily, producing rougher terrain.
    """

    width: int
    height: int
    min_elevation: float
    max_elevation: float
    cell_size: float = 1.0
    seed: int = 0
    octaves: int = 6
    persistence: float = 0.5

    def __post_init__(self) -> None:
        if self.width < 2 or self.height < 2:
            raise ValueError("terrain grid must be at least 2x2 cells")
        if self.cell_size <= 0:
            raise ValueError("cell_size must be positive")
        if self.octaves < 1:
            raise ValueError("octaves must be at least 1")
        if not 0.0 < self.persistence < 1.0:
            raise ValueError("persistence must be in (0, 1)")
        if self.max_elevation <= self.min_elevation:
            raise ValueError("max_elevation must exceed min_elevation")


class Terrain:
    """Owns the world's height field and the slope/aspect derived from it.

    heights: (height, width) float32, world units — the same unit as `cell_size`, so a slope is a
             ratio of two quantities in one unit and `climb_cost` is comparable to
             `transport_cost` (#112).
    slope:   (height, width) float32, radians from horizontal; 0 is flat.
    aspect:  (height, width) float32, radians of the downhill direction measured
             counterclockwise from +x (grid column axis); 0 on perfectly flat cells, where the
             direction is undefined.
    """

    def __init__(self, heights: np.ndarray, cell_size: float) -> None:
        if heights.ndim != 2:
            raise ValueError("heights must be a 2D grid")
        self.heights = np.asarray(heights, dtype=np.float32)
        self.cell_size = float(cell_size)
        self.slope, self.aspect = _slope_and_aspect(self.heights, self.cell_size)

    @classmethod
    def generate(cls, config: TerrainConfig) -> Terrain:
        """Build a height field from fractal value noise seeded by ``config.seed``."""
        relief = _fractal_value_noise(
            width=config.width,
            height=config.height,
            octaves=config.octaves,
            persistence=config.persistence,
            seed=config.seed,
        )
        span = relief.max() - relief.min()
        normalized = (relief - relief.min()) / span if span > 0 else np.zeros_like(relief)
        elevation_range = config.max_elevation - config.min_elevation
        heights = config.min_elevation + normalized * elevation_range
        return cls(heights, config.cell_size)

    @property
    def world_width(self) -> float:
        """Extent along x in world units."""
        return (self.heights.shape[1] - 1) * self.cell_size

    @property
    def world_height(self) -> float:
        """Extent along y in world units."""
        return (self.heights.shape[0] - 1) * self.cell_size

    def cell_indices(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Grid (row, col) of the cell containing each world position.

        Every field laid over this grid asks the same question — standing crop (#18), carrion
        (#185), reachable water (#156) — so it is answered once here, where the grid is defined,
        rather than re-derived per field. Promoted on the third repetition rather than the first
        (§8.3); before it, one field owned a private helper and the others reached into it.

        Raises ValueError outside the world, matching `bilinear_sample` — a position off the map is
        a bug in whatever moved the entity, and defaulting it to an edge cell would let animals
        graze, rot and drink at a border strip forever (§8.7).
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if not (
            np.all((x >= 0) & (x <= self.world_width))
            and np.all((y >= 0) & (y <= self.world_height))
        ):
            raise ValueError("position outside terrain bounds")

        # floor(v + 0.5) rather than np.round: round() is banker's rounding, so a position exactly
        # on a cell boundary would land in different cells depending on the boundary's parity.
        cols = np.floor(x / self.cell_size + 0.5).astype(np.int64)
        rows = np.floor(y / self.cell_size + 0.5).astype(np.int64)
        return rows, cols

    def elevation_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Bilinearly interpolated elevation (world units) at continuous world positions."""
        return self._sample(self.heights, x, y)

    def slope_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Bilinearly interpolated slope (radians from horizontal) at world positions."""
        return self._sample(self.slope, x, y)

    def aspect_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Bilinearly interpolated downhill-direction angle (radians) at world positions."""
        return self._sample(self.aspect, x, y)

    def _sample(self, field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return bilinear_sample(field, x, y, self.cell_size, self.world_width, self.world_height)


def bilinear_sample(
    field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    cell_size: float,
    world_width: float,
    world_height: float,
) -> np.ndarray:
    """Bilinearly interpolate a ``(rows, cols)`` grid field at continuous world positions.

    Shared by any grid-shaped field over the terrain grid (elevation, slope, aspect, and
    `core.world.climate`'s temperature field) so the interpolation and bounds-check logic exists
    in exactly one place rather than being re-derived per field.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    in_bounds_x = np.all((x >= 0) & (x <= world_width))
    in_bounds_y = np.all((y >= 0) & (y <= world_height))
    if not (in_bounds_x and in_bounds_y):
        raise ValueError("position outside terrain bounds")

    rows, cols = field.shape
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


def _slope_and_aspect(heights: np.ndarray, cell_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Finite-difference gradient of the height field, converted to slope and downhill aspect."""
    dz_dy, dz_dx = np.gradient(heights.astype(np.float64), cell_size)
    slope = np.arctan(np.hypot(dz_dx, dz_dy))
    aspect = np.where(
        np.hypot(dz_dx, dz_dy) > 0,
        np.arctan2(-dz_dy, -dz_dx),
        0.0,
    )
    return slope.astype(np.float32), aspect.astype(np.float32)


def _fractal_value_noise(
    width: int, height: int, octaves: int, persistence: float, seed: int
) -> np.ndarray:
    """Sum of upsampled random grids at doubling frequency (value-noise fBm), unit-free.

    The first octave is the coarsest (few cells, upsampled smoothly) and carries the most
    weight, so large-scale relief dominates; each subsequent octave doubles the resolution and
    loses weight by ``persistence``, layering progressively finer detail on top. Reversing this
    order would give fine-grained noise full weight and large-scale shape almost none, producing
    grainy static instead of continuous terrain.
    """
    rng = np.random.default_rng(seed)
    result = np.zeros((height, width), dtype=np.float64)
    amplitude = 1.0
    total_amplitude = 0.0
    for octave in range(octaves):
        divisor = 2 ** (octaves - 1 - octave)
        coarse_h = max(2, height // divisor + 1)
        coarse_w = max(2, width // divisor + 1)
        coarse = rng.random((coarse_h, coarse_w))
        result += amplitude * _upsample_bilinear(coarse, height, width)
        total_amplitude += amplitude
        amplitude *= persistence
    return result / total_amplitude


def _upsample_bilinear(coarse: np.ndarray, out_height: int, out_width: int) -> np.ndarray:
    """Bilinearly resample a small grid up to (out_height, out_width)."""
    coarse_h, coarse_w = coarse.shape
    row_pos = np.linspace(0, coarse_h - 1, out_height)
    col_pos = np.linspace(0, coarse_w - 1, out_width)
    row0 = np.floor(row_pos).astype(np.int64)
    col0 = np.floor(col_pos).astype(np.int64)
    row1 = np.minimum(row0 + 1, coarse_h - 1)
    col1 = np.minimum(col0 + 1, coarse_w - 1)
    fy = (row_pos - row0)[:, None]
    fx = (col_pos - col0)[None, :]

    top_left = coarse[row0[:, None], col0[None, :]]
    top_right = coarse[row0[:, None], col1[None, :]]
    bottom_left = coarse[row1[:, None], col0[None, :]]
    bottom_right = coarse[row1[:, None], col1[None, :]]
    top = top_left * (1 - fx) + top_right * fx
    bottom = bottom_left * (1 - fx) + bottom_right * fx
    return top * (1 - fy) + bottom * fy
