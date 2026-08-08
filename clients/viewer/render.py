"""Pure rendering math: terrain shading, water tinting, species colour, tick interpolation.

Kept free of any rendering library so it is testable without a display (CLAUDE.md §3.2: the
viewer is read-only and its coupling to the core must stay narrow). `app.py` is the only module
that touches pygame; everything here is plain NumPy in, NumPy out.
"""

from __future__ import annotations

import colorsys
from typing import TYPE_CHECKING, Optional

import numpy as np

from core.selection import Selection
from core.world.terrain import Terrain
from core.world.water import Water

if TYPE_CHECKING:  # pragma: no cover - import cycle at runtime only
    from core.world.assembly import World

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

FIELD_LAYERS: tuple[str, ...] = tuple(_LAYER_COLORS)
"""The world fields that can be drawn, in the order a viewer should cycle them.

Derived from the colour table rather than written out beside it, so a layer can never be offered
without a tint or tinted without being offered (CLAUDE.md §4: a rule declared as data must be the
thing that is consulted).

**Only one is drawn at a time**, which is why these five tints need no mutual CVD separation: they
are never on screen together, and the readout names the active one. What each *does* need is to
read against the terrain underneath it, which is what keeps them far apart in hue from the
hypsometric greens and browns and from the water's blue.

`carrion` is red deliberately, and it is the one layer that is not a resource: it marks where
something was killed (#179). Red is the documented alarm hue and nothing else in this view uses
it, so a spreading red patch is unambiguous — which is the whole point of being able to see
predation happen at all.
"""

_LAYER_SOURCE = {
    "biomass": ("plants", "biomass"),
    "carrion": ("carrion", "mass"),
    "soil_nutrients": ("plants", "soil_nutrients"),
    "moisture": ("plants", "moisture"),
    "potential_growth": ("plants", "potential_growth"),
}
"""Where each layer's array lives on a built world: `(service attribute, field attribute)`.

One table rather than a naming convention, because the fields genuinely live on different services
and `getattr(plants, layer)` stopped being true the moment a layer came from anywhere but `Plants`.
Declared beside the colours and consulted by `field_layer`, so a layer offered without a source
raises at the first frame rather than drawing the wrong field (§8.7).
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

CONDITION_MODES: tuple[str, ...] = ("species", "hunger", "thirst", "age")
"""What an entity's colour can encode, in the order a viewer should cycle them.

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
    the failure §3.3 says this instrument exists to catch. The caller computes the scale once from
    the world's own physics (`plant_overlay_references`) and holds it for the run.
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


def layer_references(world) -> dict[str, float]:
    """The value at which each layer's tint saturates, in that layer's own unit.

    Computed once from the world's static physics and then held for the run, because a scale that
    moves is a scale that hides change (see `apply_field_overlay`). Every one of these is a
    constant for the lifetime of the world: `potential_growth` is static by construction, and the
    rest are config values.

    The numbers are chosen so that **1.0 means something ecological**, not merely "the largest
    value seen so far":

    - `biomass` — `potential_growth.max() / senescence_rate`, the standing crop a cell settles at
      when soil never limits it. `grow()` adds `min(potential_growth, affordable)` and removes
      `biomass × senescence_rate`, so with nutrients to spare those balance exactly there. A cell
      therefore reads full when it is as green as its own light, warmth and water allow, and reads
      dark when something else — nutrients, or grazing — is holding it below that.
    - `carrion` — `strike_power`, the most one strike can take out of an equally-sized victim.
      A cell reads full when a specialist predator has made a kill in it.

      **This was `founder_energy` — a whole body — and rendering the view proved it wrong.** No
      single strike deposits a whole body: damage is `strike_power × size ratio × animal_share ** p`
      and the shipped world settles near a 0.3 flesh allocation, so a typical strike leaves 5.8
      energy units and the busiest cell measured 23.3. Against 180 that is 13% alpha at the
      *brightest* cell — a layer that is always empty, for a mechanic that is running. Scaled to
      the force limit instead, the same cell reads 39% and a kill is something the eye can find.
    - `soil_nutrients` — the per-cell pool every cell started with. Above 1.0 is ground that has
      accumulated more than its original share, below is ground that has given it up.
    - `moisture` — 1.0, since it is already a fraction.
    - `potential_growth` — its own maximum, which is fixed because the field is.
    """
    plants = world.plants
    light_ceiling = float(plants.potential_growth.max())
    if light_ceiling <= 0.0:
        raise ValueError(
            "no cell can grow anything: potential growth is zero across the whole grid, so there "
            "is no scale to draw a biomass ramp against"
        )
    if plants.config.initial_soil_nutrients <= 0.0:
        raise ValueError(
            "initial_soil_nutrients is zero, so a soil ramp has nothing to normalise against"
        )

    return {
        "biomass": light_ceiling / plants.config.senescence_rate,
        "carrion": world.config.predation.strike_power,
        "soil_nutrients": plants.config.initial_soil_nutrients,
        "moisture": 1.0,
        "potential_growth": light_ceiling,
    }


def field_layer(world, layer: str) -> np.ndarray:
    """The `(height, width)` array a named layer draws, fetched from wherever it lives.

    Fetched per frame rather than held, because these fields change every tick — biomass is grazed
    and regrows, carrion is deposited and rots.
    """
    service, field = _LAYER_SOURCE[layer]
    return getattr(getattr(world, service), field)


def field_overlay(
    base_rgb: np.ndarray, world, layer: str, references: dict[str, float]
) -> np.ndarray:
    """`base_rgb` with one named world field blended in. `layer` must be one of `FIELD_LAYERS`.

    `references` comes from `layer_references` and is passed in rather than recomputed, so that the
    scale is fixed for the run even though the field under it changes every tick.
    """
    return apply_field_overlay(
        base_rgb, field_layer(world, layer), references[layer], _LAYER_COLORS[layer]
    )


def condition_colors(world, drawn: np.ndarray, mode: str) -> np.ndarray:
    """(n_drawn, 3) uint8: one colour per drawn entity, encoding `mode`.

    `drawn` is the occupancy mask `live_positions` returned, so this selects exactly the rows being
    painted — never capacity, which would colour corpses (#119).

    Every mode but `species` is a **deficit quantised to `CONDITION_RAMP`**, so darker always
    means closer to death and every colour on screen is a validated ramp step. See
    `CONDITION_MODES` for why it is the deficit rather than the reserve.

    Hunger is measured against `HungerConfig.satiation_energy` — the pool level the drive itself
    treats as "wants nothing" — rather than against the population's current maximum, for the
    reason `apply_field_overlay` gives about references: a scale read off the population rescales
    every frame, so a starving herd and a fed one would render identically.
    """
    if mode == "species":
        return species_colors(world.store.species_id[drawn])

    if mode == "hunger":
        deficit = 1.0 - world.store.energy[drawn] / world.config.hunger.satiation_energy
    elif mode == "thirst":
        deficit = world.store.dehydration[drawn]
    elif mode == "age":
        # Against the oldest an animal in this world has ever been, which is the only scale the
        # world offers: there is no lifespan gene and no death clock (§2.5), so "old" is a fact
        # about the population rather than a constant anybody chose.
        oldest = max(int(world.store.age.max()), 1)
        deficit = world.store.age[drawn] / oldest
    else:
        raise ValueError(f"unknown condition mode {mode!r}; expected one of {CONDITION_MODES}")

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
    barriers, world_width: float, world_height: float, screen_width: int, screen_height: int
) -> list[tuple[int, int, int, int]]:
    """`(x0, y0, x1, y1)` pixel segments, one per blocked edge, for drawing a fence.

    A barrier lives on a cell *edge* (#27), so it draws as a line between two cells rather than as
    a filled cell — drawing the cell would put the fence half a cell off and make a pen look one
    cell smaller than it is.

    Returns segments rather than blitting, so the layout stays testable without a display, and so
    the caller decides colour and width (§3.3).
    """
    cell = barriers.terrain.cell_size
    scale_x = screen_width / world_width if world_width > 0 else 0.0
    scale_y = screen_height / world_height if world_height > 0 else 0.0
    segments = []

    # A cell centred on a node owns the half-open span around it, so the edge *above* row r sits at
    # world y = (r - 0.5) * cell. Same for the edge west of column c.
    for row, col in zip(*np.nonzero(barriers.blocked_north)):
        y = (row - 0.5) * cell * scale_y
        x = (col - 0.5) * cell * scale_x
        segments.append((int(x), int(y), int(x + cell * scale_x), int(y)))
    for row, col in zip(*np.nonzero(barriers.blocked_west)):
        x = (col - 0.5) * cell * scale_x
        y = (row - 0.5) * cell * scale_y
        segments.append((int(x), int(y), int(x), int(y + cell * scale_y)))
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
    rows: np.ndarray,
    radius: float,
) -> Optional[int]:
    """The store row of the entity nearest `(x, y)`, or None if nothing is within `radius`.

    entity_x, entity_y, rows: parallel arrays over whatever population the caller chose to make
    selectable — the living, in practice. **`rows` is returned rather than an index into those
    arrays**, because the caller has already filtered and the two numbers are different: handing
    back a position index would inspect whichever animal happened to occupy that row.

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
    return int(rows[nearest])


def describe_entity(world: "World", row: int) -> tuple[str, ...]:
    """A readable dump of one entity, line by line, for the viewer to blit (§3.3, #195).

    Returns text rather than a structure because the formatting *is* the part worth testing: this
    module is collectable in CI and `app.py` is not, so anything that decides what a number means
    has to live here (#110). `app.py` draws the lines and nothing else.

    Shows each gene's **stored and expressed value side by side**, which is the only way the
    expression mode is visible at all — a magnitude gene folding across zero and a unit-interval
    gene squashing are both invisible from either number alone.
    """
    store = world.store
    if not store.alive[row]:
        return (f"row {row}", "empty — this row holds no entity",)

    selection = Selection.from_indices(np.array([row], dtype=np.int64), store.capacity)
    age = int(store.age[row])
    lines = [
        f"entity {int(store.row_ids()[row])}   species {int(store.species_id[row])}   row {row}",
        (
            # A negative age is the gestation clock (#20), not a corrupt row.
            f"gestating — born in {-age} ticks"
            if age < 0
            else f"age {age} ticks"
        ),
        f"energy {float(store.energy[row]):.1f}"
        f"   upkeep {float(world.ecology.upkeep(selection)[0]):.3f}/tick"
        f"   exertion {float(store.exertion[row]):.3f}",
        f"position ({float(store.x[row]):.1f}, {float(store.y[row]):.1f},"
        f" {float(store.z[row]):.1f})",
        # Speed against top speed and pace against `walking_pace` are the two readings momentum and
        # haste are only visible through (#203, #204): an animal held below its own pace is being
        # limited by agility or by an empty pool, and neither shows in a position.
        f"speed {float(np.hypot(store.velocity_x[row], store.velocity_y[row])):.2f}"
        f" of {float(world.movement.top_speed(selection)[0]):.2f} top"
        f"   urge {float(store.choice_urge[row]):+.3f}"
        f" → pace {float(world.movement.pace(selection, store.choice_urge[[row]])[0]):.2f}",
        "",
    ]

    contributions = {
        name: float(values[0]) for name, values in world.behaviour.breakdown(selection).items()
    }
    total = sum(contributions.values())
    ranked = sorted(contributions.items(), key=lambda item: -item[1])
    lines.append("drives, by share of the last decision:")
    for name, value in ranked:
        share = value / total if total > 0 else 0.0
        lines.append(f"  {name:<10} {value:8.3f}  {100 * share:5.1f}%")
    lines.append("")

    stored = world.genetics.genes_at(np.array([row], dtype=np.int64))[0]
    expressed = world.genetics.expressed_at(np.array([row], dtype=np.int64))[0]
    lines.append("genes — expressed, and as stored:")
    for i, gene in enumerate(world.config.genes):
        lines.append(f"  {gene.name:<22} {float(expressed[i]):10.4f}  {float(stored[i]):10.4f}")
    return tuple(lines)
