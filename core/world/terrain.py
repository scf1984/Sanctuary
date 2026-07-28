"""Heightmap terrain: a height field over the world grid plus derived slope and aspect.

Elevation is real from the start (CLAUDE.md §2.6): it drives movement cost, line-of-sight
occlusion, downhill water flow, and temperature by altitude, so climate zones fall out of
relief instead of being painted on. Generation is a pure function of a world seed so a world
can be recreated for testing even though the running simulation is not deterministic (§2.2).
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
    cell_size: world units per cell edge.
    seed: generation is a pure function of this value; the same seed always yields the same
        height field.
    octaves: number of fractal noise layers combined to build relief. More octaves add
        finer detail on top of the same large-scale shape.
    persistence: amplitude falloff per octave, in (0, 1). Higher values weight finer octaves
        more heavily, producing rougher terrain.
    min_elevation, max_elevation: output height range, meters.
    """

    width: int
    height: int
    cell_size: float = 1.0
    seed: int = 0
    octaves: int = 6
    persistence: float = 0.5
    min_elevation: float = 0.0
    max_elevation: float = 1000.0

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

    heights: (height, width) float32, meters.
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

    def elevation_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Bilinearly interpolated elevation (meters) at continuous world positions."""
        return self._sample(self.heights, x, y)

    def slope_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Bilinearly interpolated slope (radians from horizontal) at world positions."""
        return self._sample(self.slope, x, y)

    def aspect_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Bilinearly interpolated downhill-direction angle (radians) at world positions."""
        return self._sample(self.aspect, x, y)

    def _sample(self, field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        in_bounds_x = np.all((x >= 0) & (x <= self.world_width))
        in_bounds_y = np.all((y >= 0) & (y <= self.world_height))
        if not (in_bounds_x and in_bounds_y):
            raise ValueError("position outside terrain bounds")

        rows, cols = field.shape
        gx = x / self.cell_size
        gy = y / self.cell_size
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
