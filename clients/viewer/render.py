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


def live_positions(
    previous: tuple[np.ndarray, np.ndarray, np.ndarray],
    previous_row_ids: np.ndarray,
    current: tuple[np.ndarray, np.ndarray, np.ndarray],
    current_row_ids: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Where to draw each live entity this frame, blended between two tick-boundary snapshots.

    `alpha` in [0, 1]: 0 renders exactly at `previous`, 1 exactly at `current`. Tick size is a
    simulation concern and must stay independent of how smooth this looks (CLAUDE.md §2.1), so
    this is the only place frame-rate-driven blending happens.

    Returns `(x, y, z, drawn)`: three `(n_live,)` arrays of world-unit coordinates, plus the
    `(capacity,)` bool mask that selected them, so a caller can filter any other column — species
    id, energy, a future overlay — onto exactly the same rows.

    **Occupancy is an argument, not an afterthought.** `EntityStore.release` clears `alive` and the
    id mapping but deliberately leaves `x`, `y` and `z` untouched, since `allocate` overwrites
    whatever its caller seeds. A snapshot of positions is therefore full *capacity*, not
    population, and drawing it whole paints every row that has ever been used — a corpse frozen at
    the spot it died, in its species colour, forever (#119). That was invisible only because
    nothing had ever died: the demo world allocated exactly its capacity and never bred.

    Two distinct rows are excluded, and one id array answers both because ids are never reused:

    - **Not occupied now** (`current_row_ids < 0`) — nothing to draw.
    - **Not the same entity as at `previous`** — there is no position to blend *from*. Such a row
      is drawn at its current position instead. This covers a newborn in a fresh row, whose
      previous entry holds nothing meaningful, *and* a newborn in a recycled one, whose previous
      entry holds its predecessor's death site — the second is why this compares ids rather than an
      `alive` flag, which reads True at both ends of that interval and hides the reuse entirely.
      Without it a newborn streaks across the screen from wherever the last occupant fell.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")

    drawn = current_row_ids >= 0
    # Blend only where the row held this same entity at both ends of the interval. Elsewhere the
    # previous coordinate belongs to somebody else, so `alpha` is replaced by 1 and the entity is
    # drawn where it is now.
    continuous = current_row_ids[drawn] == previous_row_ids[drawn]
    blend = np.where(continuous, alpha, 1.0)

    previous_x, previous_y, previous_z = (axis[drawn] for axis in previous)
    current_x, current_y, current_z = (axis[drawn] for axis in current)
    return (
        previous_x + (current_x - previous_x) * blend,
        previous_y + (current_y - previous_y) * blend,
        previous_z + (current_z - previous_z) * blend,
        drawn,
    )


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
