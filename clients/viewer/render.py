"""Pure rendering math: terrain shading, water tinting, species colour, tick interpolation.

Kept free of any rendering library so it is testable without a display (CLAUDE.md §3.2: the
viewer is read-only and its coupling to the core must stay narrow). `app.py` is the only module
that touches pygame; everything here is plain NumPy in, NumPy out.
"""

from __future__ import annotations

import colorsys

import numpy as np

from core.world.terrain import Terrain
from core.world.water import Water

# Diagonal lighting from the upper-left, the conventional default for cartographic relief
# shading. Expressed in Terrain.aspect's own convention (radians, counterclockwise from +x) so
# no separate compass-bearing conversion is needed.
_LIGHT_ALTITUDE = np.radians(45.0)
_LIGHT_AZIMUTH = np.radians(135.0)

# Hypsometric tint stops (low -> mid -> high), matching the conventional green-to-brown-to-white
# elevation ramp used on physical relief maps.
_ELEVATION_STOPS = np.array([0.0, 0.5, 1.0])
_ELEVATION_COLORS = np.array(
    [
        [60.0, 110.0, 60.0],
        [140.0, 120.0, 80.0],
        [245.0, 245.0, 245.0],
    ]
)

_WATER_COLOR = np.array([40.0, 90.0, 200.0])
# Depth (world units, as `Water.depth` reports) at which standing water reaches its most
# saturated tint; deeper water clips to
# the same color rather than growing darker without bound.
_WATER_REFERENCE_DEPTH = 3.0

_UNSET_SPECIES_COLOR = np.array([128, 128, 128], dtype=np.uint8)
# Irrational turn fraction: successive hashed hues land far apart on the color wheel however
# many species ids are drawn, unlike `id % n`, which collides as soon as ids exceed n.
_GOLDEN_RATIO_CONJUGATE = 0.6180339887498949


def elevation_shading(terrain: Terrain) -> np.ndarray:
    """Hypsometric-tinted, hillshaded terrain color, (height, width, 3) uint8.

    Color comes from a fixed low-to-high ramp over the terrain's own elevation range; brightness
    comes from a hillshade so ridges and depressions read as relief rather than flat color bands
    — the concrete failure mode (CLAUDE.md's Why: "creatures walking through ridges" is invisible
    without this) that motivates this being anything more than a flat colormap.
    """
    heights = terrain.heights.astype(np.float64)
    span = heights.max() - heights.min()
    normalized = (heights - heights.min()) / span if span > 0 else np.zeros_like(heights)

    color = np.empty(heights.shape + (3,), dtype=np.float64)
    for channel in range(3):
        color[..., channel] = np.interp(
            normalized, _ELEVATION_STOPS, _ELEVATION_COLORS[:, channel]
        )

    brightness = _hillshade(terrain)
    shaded = color * brightness[..., None]
    return np.clip(shaded, 0.0, 255.0).astype(np.uint8)


def _hillshade(terrain: Terrain) -> np.ndarray:
    """Relative brightness, (height, width) float64 in [0.3, 1.0], from slope and aspect.

    Standard hillshade formula (zenith/azimuth form), floored at 0.3 rather than 0 so that
    shadowed slopes stay legible instead of reading as pure black — this is a diagnostic view,
    not an artistic render.
    """
    zenith = np.pi / 2.0 - _LIGHT_ALTITUDE
    slope = terrain.slope.astype(np.float64)
    aspect = terrain.aspect.astype(np.float64)
    cos_incidence = np.cos(zenith) * np.cos(slope) + np.sin(zenith) * np.sin(slope) * np.cos(
        aspect - _LIGHT_AZIMUTH
    )
    return 0.3 + 0.7 * np.clip(cos_incidence, 0.0, 1.0)


def apply_water_overlay(base_rgb: np.ndarray, water: Water) -> np.ndarray:
    """`base_rgb` (height, width, 3) uint8 with standing water blended in over `water.depth`.

    Blend strength grows with depth up to `_WATER_REFERENCE_DEPTH`, so a shallow puddle still
    reads as water without every lake looking identically saturated regardless of how deep it is.
    """
    depth = water.depth.astype(np.float64)
    alpha = np.clip(depth / _WATER_REFERENCE_DEPTH, 0.0, 1.0) * 0.85
    alpha = np.where(depth > 0.0, np.maximum(alpha, 0.5), 0.0)
    blended = base_rgb.astype(np.float64) * (1.0 - alpha[..., None]) + _WATER_COLOR * alpha[
        ..., None
    ]
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)


def species_colors(species_id: np.ndarray) -> np.ndarray:
    """(n,3) uint8 RGB, one deterministic color per entity keyed by its species id.

    Unset entities (species_id == -1) render as neutral gray rather than being hashed into the
    same palette as a real species.
    """
    species_id = np.asarray(species_id)
    colors = np.empty((species_id.shape[0], 3), dtype=np.uint8)
    for i, sid in enumerate(species_id.tolist()):
        colors[i] = _UNSET_SPECIES_COLOR if sid < 0 else _color_for_id(sid)
    return colors


def _color_for_id(species_id: int) -> np.ndarray:
    hue = (species_id * _GOLDEN_RATIO_CONJUGATE) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return (np.array([r, g, b]) * 255.0).astype(np.uint8)


def interpolate_positions(
    previous: tuple[np.ndarray, np.ndarray, np.ndarray],
    current: tuple[np.ndarray, np.ndarray, np.ndarray],
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Blend two tick-boundary position snapshots for smooth rendering between ticks (§2.1).

    `alpha` in [0, 1]: 0 renders exactly at `previous`, 1 exactly at `current`. Tick size is a
    simulation concern and must stay independent of how smooth this looks (CLAUDE.md §2.1), so
    this is the only place frame-rate-driven blending happens.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    return tuple(prev + (curr - prev) * alpha for prev, curr in zip(previous, current))


def world_to_screen(
    x: np.ndarray,
    y: np.ndarray,
    world_width: float,
    world_height: float,
    screen_width: int,
    screen_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map world-unit (x, y) to top-down integer pixel coordinates for a `screen_width x
    screen_height` window covering exactly `(world_width, world_height)` world units.
    """
    px = (x / world_width) * screen_width if world_width > 0 else np.zeros_like(x)
    py = (y / world_height) * screen_height if world_height > 0 else np.zeros_like(y)
    return px.astype(np.int32), py.astype(np.int32)
