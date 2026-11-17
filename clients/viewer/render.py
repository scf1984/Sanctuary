"""Pure rendering math: terrain shading, water tinting, species colour, tick interpolation.

Kept free of any rendering library so it is testable without a display. `app.py` is the only
module that touches pygame; everything here is plain NumPy in, NumPy out.

**It reads nothing but what `api` hands it** (#35). What a number *means* ecologically (how hungry
an animal is, what a full biomass cell is) is decided behind the boundary and arrives as a value.
This module decides only how that value looks.
"""

from __future__ import annotations

import colorsys
from typing import Optional

import numpy as np

import api

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
# Depth (world units, as `TerrainView.water_depth` reports) at which standing water reaches its most
# saturated tint; deeper water clips to
# the same color rather than growing darker without bound.
_WATER_REFERENCE_DEPTH = 3.0

# Plant-field tints. Every one is deliberately outside the palettes already on screen: the
# elevation ramp owns desaturated green, tan and white, and water owns deep blue, so an overlay
# drawn in any of those would be read as terrain rather than as data laid over it.
_LAYER_COLORS = {
    "biomass": np.array([120.0, 225.0, 40.0]),  # lime, against the ramp's dark green
    "carrion": np.array([227.0, 73.0, 72.0]),  # red, the documented status/alarm hue
    "soil_nutrients": np.array([190.0, 70.0, 200.0]),  # violet, used by nothing else
    "moisture": np.array([40.0, 210.0, 210.0]),  # cyan, against water's deep blue
    "potential_growth": np.array([245.0, 180.0, 40.0]),  # amber
}

FIELD_LAYERS: tuple[str, ...] = api.FIELD_LAYERS
"""The world fields that can be drawn, in the order a viewer should cycle them.

The boundary decides which fields exist, and this module only tints them. A test holds the colour
table to exactly that list, so a layer can never be offered without a tint or tinted without being
offered (CLAUDE.md §4: a rule declared as data must be the thing that is consulted).

**Only one is drawn at a time**, which is why these five tints need no mutual CVD separation: they
are never on screen together, and the readout names the active one. What each *does* need is to
read against the terrain underneath it, which is what keeps them far apart in hue from the
hypsometric greens and browns and from the water's blue.

`carrion` is red deliberately, and it is the one layer that is not a resource: it marks where
something was killed (#179). Red is the documented alarm hue and nothing else in this view uses
it, so a spreading red patch is unambiguous — which is the whole point of being able to see
predation happen at all.
"""

# Validated as an ordinal ramp against this viewer's own terrain (mean #6a6a64) with the dataviz
# skill's checker: monotone lightness, every adjacent gap >= 0.06 L, light end 3.96:1 on that
# surface, hue spread 6 degrees. One hue, light to dark, which is the rule for a magnitude.
#
# Orange rather than the default blue because the *surface* has already spent blue on standing
# water: a blue animal over a blue lake is the one place this view most needs to stay readable.
#
# Quantised to these five steps rather than interpolated between them, for two reasons. Every
# colour that reaches the screen is then a validated step rather than a blend nothing checked; and
# a thousand dots read as bands, where a continuous ramp reads as mush at four pixels across.
CONDITION_RAMP = np.array(
    [
        [250, 212, 196],
        [244, 165, 130],
        [235, 104, 52],
        [184, 69, 29],
        [125, 44, 15],
    ],
    dtype=np.uint8,
)

# A light casing around every dot. No single fill colour clears 3:1 against all of this terrain --
# measured, the dark ramp steps vanish on dark ground (1.02:1) and the light steps on bright ground
# (1.80:1) -- which is exactly why map marks are cased. A light ring plus a ramp that runs dark
# gives every dot two tones, and at least one of the two separates from whatever is behind it.
_CASING_COLOR = np.array([252, 252, 251], dtype=np.uint8)

CONDITION_MODES: tuple[str, ...] = ("species", *api.CONDITIONS)
"""What an entity's colour can encode, in the order a viewer should cycle them.

`species` is the viewer's own: it colours by an id the frame already carries. Every other mode is
a deficit `Session.condition` computes, because what "hungry" is measured against is a fact of the
world rather than of the view.

**Every mode but the first is a deficit, and they all darken toward death.** Hunger is the energy
an animal is short of, thirst is the water it has lost, age is how far through a life it is. Read
the other way up -- colouring the *reserve* -- the animal in trouble would be the faintest mark on
screen, which is precisely backwards for an instrument whose job is spotting trouble (§3.3).

`species` stays first because it is what the view meant before this, and it is still the right
answer once there is more than one species to tell apart (#16).
"""

# Peak blend strength, matching the water overlay so the two read as the same kind of mark. Short
# of 1.0 because relief must stay visible underneath: an overlay that hid the hillshade would make
# "the ridge is bare" and "there is no ridge" look identical.
_FIELD_MAX_ALPHA = 0.85

_UNSET_SPECIES_COLOR = np.array([128, 128, 128], dtype=np.uint8)
# Irrational turn fraction: successive hashed hues land far apart on the color wheel however
# many species ids are drawn, unlike `id % n`, which collides as soon as ids exceed n.
_GOLDEN_RATIO_CONJUGATE = 0.6180339887498949


def elevation_shading(terrain: api.TerrainView) -> np.ndarray:
    """Hypsometric-tinted, hillshaded terrain color, (height, width, 3) uint8.

    Color comes from a fixed low-to-high ramp over the terrain's own elevation range; brightness
    comes from a hillshade so ridges and depressions read as relief rather than flat color bands
    — the concrete failure mode (CLAUDE.md's Why: "creatures walking through ridges" is invisible
    without this) that motivates this being anything more than a flat colormap.
    """
    heights = terrain.elevation.to_numpy().astype(np.float64)
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


def _hillshade(terrain: api.TerrainView) -> np.ndarray:
    """Relative brightness, (height, width) float64 in [0.3, 1.0], from slope and aspect.

    Standard hillshade formula (zenith/azimuth form), floored at 0.3 rather than 0 so that
    shadowed slopes stay legible instead of reading as pure black — this is a diagnostic view,
    not an artistic render.
    """
    zenith = np.pi / 2.0 - _LIGHT_ALTITUDE
    slope = terrain.slope.to_numpy().astype(np.float64)
    aspect = terrain.aspect.to_numpy().astype(np.float64)
    cos_incidence = np.cos(zenith) * np.cos(slope) + np.sin(zenith) * np.sin(slope) * np.cos(
        aspect - _LIGHT_AZIMUTH
    )
    return 0.3 + 0.7 * np.clip(cos_incidence, 0.0, 1.0)


def apply_water_overlay(base_rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """`base_rgb` (height, width, 3) uint8 with standing water blended in over `depth`, a
    (height, width) grid in world units.

    Blend strength grows with depth up to `_WATER_REFERENCE_DEPTH`, so a shallow puddle still
    reads as water without every lake looking identically saturated regardless of how deep it is.
    """
    depth = np.asarray(depth, dtype=np.float64)
    alpha = np.clip(depth / _WATER_REFERENCE_DEPTH, 0.0, 1.0) * 0.85
    alpha = np.where(depth > 0.0, np.maximum(alpha, 0.5), 0.0)
    blended = base_rgb.astype(np.float64) * (1.0 - alpha[..., None]) + _WATER_COLOR * alpha[
        ..., None
    ]
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)


def apply_field_overlay(
    base_rgb: np.ndarray, field: np.ndarray, reference: float, color: np.ndarray
) -> np.ndarray:
    """`base_rgb` (height, width, 3) uint8 with a per-cell `field` blended in against `reference`.

    `field` is (height, width) in whatever unit the layer holds; `reference` is a value in that
    same unit at which the tint saturates. A cell reading `reference` or more is drawn at full
    strength, so anything above it clips rather than tinting further — nutrients are conserved and
    therefore concentrate (#18), and a cell holding several times an ordinary share must read as
    "full" rather than overflowing the blend into some other colour.

    **`reference` is supplied, never measured off `field`.** That is the whole point of the
    parameter. Normalising against the field's current range would rescale the ramp every frame,
    so a world slowly starving to death would render identically to a lush one — which is exactly
    the failure §3.3 says this instrument exists to catch. The scale comes from the world's own
    physics, as `FieldView.reference`, and is fixed for the session.
    """
    if reference <= 0.0:
        raise ValueError(f"reference must be positive, got {reference}")
    if field.shape != base_rgb.shape[:2]:
        raise ValueError(
            f"field and base_rgb must share a grid shape, got {field.shape} and "
            f"{base_rgb.shape[:2]}"
        )

    alpha = np.clip(field.astype(np.float64) / reference, 0.0, 1.0) * _FIELD_MAX_ALPHA
    blended = base_rgb.astype(np.float64) * (1.0 - alpha[..., None]) + color * alpha[..., None]
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)


def field_overlay(base_rgb: np.ndarray, view: api.FieldView) -> np.ndarray:
    """`base_rgb` with one world field blended in, tinted by its layer and scaled by the reference
    the world reported for it. The scale is never measured here (see `apply_field_overlay`)."""
    return apply_field_overlay(
        base_rgb, view.values.to_numpy(), view.reference, _LAYER_COLORS[view.layer]
    )


def condition_colors(deficit: np.ndarray) -> np.ndarray:
    """(n, 3) uint8: one colour per animal, from its deficit in [0, 1] (`Session.condition`).

    **Quantised to `CONDITION_RAMP`**, so darker always means closer to death and every colour on
    screen is a validated ramp step. See `CONDITION_MODES` for why it is the deficit rather than
    the reserve. Clipped again here, because indexing off the end of the ramp is this module's
    failure to prevent whatever arrives.
    """
    step = np.clip(
        (np.clip(deficit, 0.0, 1.0) * len(CONDITION_RAMP)).astype(np.int64),
        0,
        len(CONDITION_RAMP) - 1,
    )
    return CONDITION_RAMP[step]


def casing_color() -> tuple[int, int, int]:
    """The ring drawn around every entity dot, so a mark reads against any terrain.

    See `_CASING_COLOR`: no single fill clears 3:1 against this view's whole terrain range, which
    is why the mark is two-toned rather than one.
    """
    return tuple(int(channel) for channel in _CASING_COLOR)


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
    previous: api.Frame, current: api.Frame, alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Where to draw each animal in `current` this frame, blended from where it was in `previous`.

    `alpha` in [0, 1]: 0 renders exactly at `previous`, 1 exactly at `current`. Tick size is a
    simulation concern and must stay independent of how smooth this looks (CLAUDE.md §2.1), so
    this is the only place frame-rate-driven blending happens.

    Returns `(x, y, z)`, each `(n,)` world units in `current`'s order, so any other column handed
    over with that frame (species, a condition) lines up with it.

    **Blended by id, because ids are never reused** (§2.3). An animal in both frames moves smoothly
    from one position to the other. One that is new in `current` has no position to blend from and
    is drawn where it is, whether it was just born or the client never saw it. This is #119's rule.
    It used to be enforced by comparing row ids, because a recycled row hid a newborn behind its
    predecessor's death site. A frame carries no rows at all, so that mistake cannot be made here.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")

    current_ids = current.ids.to_numpy()
    previous_ids = previous.ids.to_numpy()
    now = [axis.to_numpy().astype(np.float64) for axis in (current.x, current.y, current.z)]
    if not previous_ids.size or not current_ids.size:
        return now[0], now[1], now[2]

    # For each current animal, where (if anywhere) it sits in the previous frame: one sort and one
    # binary search, rather than a dictionary built per frame.
    order = np.argsort(previous_ids)
    slot = np.minimum(
        np.searchsorted(previous_ids, current_ids, sorter=order), previous_ids.size - 1
    )
    source = order[slot]
    blend = np.where(previous_ids[source] == current_ids, alpha, 1.0)

    blended = []
    for before, after in zip((previous.x, previous.y, previous.z), now):
        start = before.to_numpy().astype(np.float64)[source]
        blended.append(start + (after - start) * blend)
    return blended[0], blended[1], blended[2]


def drag_rectangle(
    start: tuple[int, int], now: tuple[int, int]
) -> tuple[int, int, int, int]:
    """`(left, top, width, height)` in pixels for a drag from `start` to `now`.

    Normalised, so a box dragged up-and-left is the same box as one dragged down-and-right — a
    player should not have to know which corner the code considers first.

    Here rather than in `app.py` for #110's reason: geometry belongs somewhere a test can collect
    it. A rectangle drawn from the wrong corner is invisible on inspection and obvious in use.
    """
    left, right = sorted((start[0], now[0]))
    top, bottom = sorted((start[1], now[1]))
    return left, top, right - left, bottom - top


def barrier_segments(
    barriers: api.BarrierView,
    cell_size: float,
    world_width: float,
    world_height: float,
    screen_width: int,
    screen_height: int,
) -> list[tuple[int, int, int, int]]:
    """`(x0, y0, x1, y1)` pixel segments, one per blocked edge, for drawing a fence.

    A barrier lives on a cell *edge* (#27), so it draws as a line between two cells rather than as
    a filled cell — drawing the cell would put the fence half a cell off and make a pen look one
    cell smaller than it is.

    Returns segments rather than blitting, so the layout stays testable without a display, and so
    the caller decides colour and width (§3.3).
    """
    scale_x = screen_width / world_width if world_width > 0 else 0.0
    scale_y = screen_height / world_height if world_height > 0 else 0.0
    segments = []

    # Cell r spans [r, r + 1) cells (#162), so the edge *above* row r sits at world y = r * cell.
    # Same for the edge west of column c.
    for row, col in zip(*np.nonzero(barriers.blocked_north.to_numpy())):
        y = row * cell_size * scale_y
        x = col * cell_size * scale_x
        segments.append((int(x), int(y), int(x + cell_size * scale_x), int(y)))
    for row, col in zip(*np.nonzero(barriers.blocked_west.to_numpy())):
        x = col * cell_size * scale_x
        y = row * cell_size * scale_y
        segments.append((int(x), int(y), int(x), int(y + cell_size * scale_y)))
    return segments


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

def screen_to_world(
    px: np.ndarray,
    py: np.ndarray,
    world_width: float,
    world_height: float,
    screen_width: int,
    screen_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """The inverse of `world_to_screen`: where in the world a pixel points.

    Returns zeros for a degenerate world rather than dividing by its extent, matching what
    `world_to_screen` does in the same case — the two have to agree or a round trip would land
    somewhere neither of them meant.
    """
    x = (px / screen_width) * world_width if screen_width > 0 else np.zeros_like(px)
    y = (py / screen_height) * world_height if screen_height > 0 else np.zeros_like(py)
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)


def pick_entity(
    x: float,
    y: float,
    entity_x: np.ndarray,
    entity_y: np.ndarray,
    ids: np.ndarray,
    radius: float,
) -> Optional[int]:
    """The id of the animal nearest `(x, y)`, or None if nothing is within `radius`.

    entity_x, entity_y, ids: parallel arrays, a frame's columns in practice. **The id is returned
    rather than an index into those arrays**, because the index means nothing once the next frame
    arrives in a different order, and the id still names the same animal.

    None rather than a nearest-anything, so clicking empty ground clears the panel instead of
    leaving the last animal selected — which would be a readout quietly describing something the
    player is no longer pointing at.
    """
    if entity_x.shape[0] == 0:
        return None
    gap = np.hypot(entity_x - x, entity_y - y)
    nearest = int(np.argmin(gap))
    if gap[nearest] > radius:
        return None
    return int(ids[nearest])


def describe_entity(entity_id: int, report: Optional[api.EntityReport]) -> tuple[str, ...]:
    """A readable dump of one animal, line by line, for the viewer to blit (§3.3, #195).

    Returns text rather than a structure because the formatting *is* the part worth testing: this
    module is collectable in CI and `app.py` is not, so anything that decides how a number reads
    has to live here (#110). What the numbers *are* is `Session.inspect`'s.

    Shows each gene's **stored and expressed value side by side**, which is the only way the
    expression mode is visible at all — a magnitude gene folding across zero and a unit-interval
    gene squashing are both invisible from either number alone.
    """
    if report is None:
        return (f"entity {entity_id}", "gone — no animal holds this id any more")

    lines = [
        f"entity {report.id}   species {report.species}",
        (
            # A negative age is the gestation clock (#20), not a corrupt reading.
            f"gestating — born in {-report.age} ticks"
            if report.age < 0
            else f"age {report.age} ticks"
        ),
        f"energy {report.energy:.1f}   upkeep {report.upkeep:.3f}/tick"
        f"   exertion {report.exertion:.3f}",
        f"position ({report.x:.1f}, {report.y:.1f}, {report.z:.1f})",
        # Speed against top speed and pace against `walking_pace` are the two readings momentum and
        # haste are only visible through (#203, #204): an animal held below its own pace is being
        # limited by agility or by an empty pool, and neither shows in a position.
        f"speed {report.speed:.2f} of {report.top_speed:.2f} top"
        f"   urge {report.urge:+.3f} → pace {report.pace:.2f}",
        "",
    ]

    total = sum(share.contribution for share in report.drives)
    lines.append("drives, by share of the last decision:")
    for share in report.drives:
        fraction = share.contribution / total if total > 0 else 0.0
        lines.append(f"  {share.name:<10} {share.contribution:8.3f}  {100 * fraction:5.1f}%")
    lines.append("")

    lines.append("genes — expressed, and as stored:")
    for gene in report.genes:
        lines.append(f"  {gene.name:<22} {gene.expressed:10.4f}  {gene.stored:10.4f}")
    return tuple(lines)
