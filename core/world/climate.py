"""Temperature field derived from altitude, latitude and time, and climate zones derived from it.

Elevation and world position drive temperature (CLAUDE.md §2.6), so climate zones fall out of
terrain rather than being painted regions. This module owns the abstraction contract from the
issue that introduced it: the temperature field is queryable per continuous position, and zone
labels are derived from that field after the fact, for display and metrics only — no simulation
logic may branch on a zone name. Systems that need the field for real effects (thermoregulation
cost in metabolism, plant productivity) query `Climate.temperature_at` directly, never a label.

**Weather is derived, never simulated (#159).** With `ClimateConfig.seasons` set, the field is a
pure function of ``(x, y, tick, seed)``: the static altitude-and-latitude field, plus a seasonal
cycle whose swing grows with distance from the equator, plus smooth noise. Nothing is integrated
from one tick to the next, so there is no weather state: a snapshot stores none (§3.2), offline
catch-up evaluates the field at whatever tick it lands on rather than stepping through the ones it
skipped (§2.4), and one batch of a thousand ticks sees exactly the sky a thousand batches of one
do. `advance_to` is the tick system that writes the field for the tick about to run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.world.terrain import Terrain, bilinear_sample

# Ordered (label, upper bound in temperature units, exclusive) pairs. A cell's zone is the first bucket whose
# upper bound exceeds its temperature. Boundaries are round, illustrative values for grouping
# a continuous field into legible bands — display and metrics only (see module docstring).
# They sit where they do relative to the demo world's equator and neutral temperatures, not
# at any physical landmark: in temperature units nothing freezes at 0 (#238).
_ZONE_BANDS: tuple[tuple[str, float], ...] = (
    ("polar", -10.0),
    ("tundra", 0.0),
    ("temperate", 20.0),
    ("tropical", float("inf")),
)

# Mixed into the world seed so the weather lattice never draws the same numbers as terrain
# generation, which seeds a generator from the bare seed. Its value is arbitrary; that it differs
# from anything else seeded off the world seed is not.
_WEATHER_STREAM = 159


@dataclass(frozen=True)
class SeasonsConfig:
    """How the temperature field moves in time (#159). Every field is required: each is a statement
    about this world's calendar or sky, and none has a value that is right everywhere.

    year_length: ticks per seasonal cycle. **A ratio-table entry (§2.1), not a free constant**: a
        year is one lifespan, about six generations, so roughly 1.5 generations live through each
        season. That ratio decides what seasonality *is* — much longer than a generation and
        lineages adapt to it genetically, much shorter and only individuals can cope with it —
        and a world sets it from its own generation time rather than from a calendar.
    amplitude_fraction: the seasonal swing at a given distance from the equator, as a fraction of
        the temperature `latitude_gradient` already drops over that distance. The swing therefore
        grows toward the poles, as on a real planet, and the equator has no season.

        **Denominated in the latitude gradient so it cannot invert it.** #159 measured the trap: a
        swing that grows with latitude at or past the equator-to-pole gradient makes summer poles
        as hot as the tropics, and past it hotter — the climate turns inside out. Stated as a
        fraction of that gradient, the crossover is exactly 1.0 and is refused here, and the swing
        moves with `latitude_gradient` rather than drifting away from it. That makes it the third
        member of the `lapse_rate`/`latitude_gradient` set by construction rather than by a test
        that hopes nobody retunes one alone.
    weather_amplitude: temperature units. Standard deviation of the weather anomaly at its lattice
        knots. It is the size of a cold snap or a heat wave, not a gradient, so it is absolute.
    weather_scale: world units between weather lattice points — how large one weather system is.
        The noise is evaluated on that coarse lattice and upsampled, because it is smooth at that
        scale by construction and #159 measured gathering it per terrain cell at 34x the cost.
    weather_timescale: ticks between independent weather states — how long a spell lasts. The
        field blends smoothly from one state to the next in between.
    """

    year_length: int
    amplitude_fraction: float
    weather_amplitude: float
    weather_scale: float
    weather_timescale: int

    def __post_init__(self) -> None:
        if self.year_length < 2:
            raise ValueError("year_length must be at least two ticks, or there is no cycle")
        if self.amplitude_fraction < 0:
            raise ValueError("amplitude_fraction must be non-negative")
        if self.amplitude_fraction >= 1.0:
            raise ValueError(
                f"amplitude_fraction {self.amplitude_fraction} would invert the latitude "
                "gradient: at 1.0 a summer pole is as hot as the equator, and past it hotter "
                "(#159). It must stay below 1.0"
            )
        if self.weather_amplitude < 0:
            raise ValueError("weather_amplitude must be non-negative")
        if self.weather_scale <= 0:
            raise ValueError("weather_scale must be positive")
        if self.weather_timescale < 1:
            raise ValueError("weather_timescale must be at least one tick")


@dataclass(frozen=True)
class ClimateConfig:
    """Parameters mapping terrain position to temperature, in temperature units (#238).

    equator_y: world-unit y coordinate of peak (equatorial) temperature. Temperature falls off
        with distance from this line in both directions, so it models one hemisphere's worth of
        latitude band across the world's y extent rather than a full planet. Seasons follow the
        same folding: both sides of the line share one calendar, rather than summer on one side
        being winter on the other.
    equator_temperature: temperature at the equator at `sea_level_elevation`. Required, with no
        default: it is an absolute temperature, and an absolute temperature only means something
        against the others it is read beside — `MetabolismConfig.neutral_temperature`,
        `HydrationConfig.neutral_temperature`, `PlantsConfig`'s growth band — all of which are
        already required. A default on one side of that relation would let the two drift apart
        silently. Same principle #112 applied to absolute lengths (#238).
    latitude_gradient: temperature units lost per world unit of distance from `equator_y`.
    sea_level_elevation: elevation (world units) at which the altitude lapse rate is zero. A
        default of 0.0 is a reference point rather than a guess about scale, so unlike
        `TerrainConfig`'s elevation range it may carry one.
    lapse_rate: temperature units lost per **world unit** of elevation above `sea_level_elevation`
        (#112). It was denominated in a physical length unit and defaulted to Earth's tropospheric
        lapse rate, which is exactly the Earth-calibrated constant this world has no use for —
        read against elevation in world units, that default cooled a peak by hundredths of a
        temperature unit, and altitude stopped driving climate at all.

        Re-derived from the world's own geometry instead: a peak should be about as cold as a
        pole, or §2.6's "climate zones are consequences of terrain" is decoration. That means
        ``lapse_rate * relief ~= latitude_gradient * half_extent``, and with relief chosen at
        roughly a tenth of extent (see `TerrainConfig.max_elevation`) it lands at five times
        `latitude_gradient`. The two are therefore **one pair, tuned together** — §2.1's warning
        about constants drifting apart applies to them exactly, and a test pins the factor so
        that moving one alone fails rather than quietly halving how much terrain matters.
    seasons: how the field moves in time (#159), or None for a climate that never changes. None is
        for fixtures that isolate one mechanism; it is not a claim that a world has no weather.
    """

    equator_y: float
    equator_temperature: float
    latitude_gradient: float = 0.05
    sea_level_elevation: float = 0.0
    lapse_rate: float = 0.25
    seasons: SeasonsConfig | None = None

    def __post_init__(self) -> None:
        if self.latitude_gradient < 0:
            raise ValueError("latitude_gradient must be non-negative")
        if self.lapse_rate < 0:
            raise ValueError("lapse_rate must be non-negative")


class Climate:
    """Owns the world's temperature field, derived from terrain, world position and the tick.

    temperature: (height, width) float32, temperature units — one value per terrain grid cell,
        aligned with `terrain.heights`. The field for the tick most recently passed to
        `advance_to` (tick 0 until then). Written **in place**, so a reader holding this array
        reads the current sky rather than the one it was handed at assembly.
    """

    def __init__(self, terrain: Terrain, config: ClimateConfig, seed: int | None = None) -> None:
        if config.seasons is not None and seed is None:
            raise ValueError(
                "a climate with seasons needs the world seed: its weather is a function of "
                "(x, y, tick, seed), and without the seed two worlds would share one sky (#159)"
            )
        self.terrain = terrain
        self.config = config
        self._seed = seed
        self._base = _temperature_field(terrain, config)
        self.temperature = self._base.astype(np.float32)
        self._seasonal_profile = None
        self._lattice = None
        self._knots: dict[int, np.ndarray] = {}
        if config.seasons is not None:
            # Per row, the temperature units one full summer adds: `amplitude_fraction` of what
            # the latitude gradient drops between the equator and that row (see SeasonsConfig).
            row_y = np.arange(terrain.heights.shape[0], dtype=np.float64) * terrain.cell_size
            self._seasonal_profile = (
                config.seasons.amplitude_fraction
                * config.latitude_gradient
                * np.abs(row_y - config.equator_y)
            )[:, None]
            self._lattice = _UpsamplingLattice(terrain, config.seasons.weather_scale)
            self.advance_to(0)

    def advance_to(self, tick: int) -> None:
        """Write the field for `tick` into `temperature`. A tick system, run first in the tick."""
        if self.config.seasons is None:
            return
        self.temperature[...] = self.field_at(tick)

    def field_at(self, tick: int) -> np.ndarray:
        """(height, width) float32: the temperature field at `tick`, as a new array.

        A pure function of the terrain, the config, the seed and `tick`. Calling it in any order,
        or skipping ticks, gives the same answer — which is what makes batching and catch-up
        unable to change the weather (§2.4).
        """
        seasons = self.config.seasons
        if seasons is None:
            return self._base.astype(np.float32)
        assert self._seasonal_profile is not None and self._lattice is not None
        # Phase zero is the spring equinox, so a world founded at tick 0 starts at its mean climate
        # and the first quarter-year warms toward summer.
        season = np.sin(2.0 * np.pi * tick / seasons.year_length)
        field = self._base + self._seasonal_profile * season
        if seasons.weather_amplitude > 0.0:
            field = field + seasons.weather_amplitude * self._lattice.upsample(
                self._weather_lattice(tick)
            )
        return field.astype(np.float32)

    def _weather_lattice(self, tick: int) -> np.ndarray:
        """The coarse weather anomaly at `tick`, in standard deviations, blended between knots.

        Knot ``k`` is an independent standard-normal lattice drawn from a generator seeded by
        ``(seed, stream, k)`` alone, so any knot can be produced without the ones before it. Between
        knots the field moves along a smoothstep, so a spell builds and fades rather than switching.
        Blending two independent draws shrinks the spread midway (to 1/sqrt(2) at the midpoint);
        that is accepted rather than renormalised, because a renormalised blend would swing harder
        between knots than at them.
        """
        assert self.config.seasons is not None
        timescale = self.config.seasons.weather_timescale
        knot, offset = divmod(tick, timescale)
        fraction = offset / timescale
        blend = fraction * fraction * (3.0 - 2.0 * fraction)
        return (1.0 - blend) * self._knot(knot) + blend * self._knot(knot + 1)

    def _knot(self, knot: int) -> np.ndarray:
        # A cache of derived values only: dropping it changes nothing but the cost. Two knots are
        # live at a time, so it never holds more than the pair in use.
        assert self._seed is not None and self._lattice is not None
        if knot not in self._knots:
            if len(self._knots) >= 2:
                self._knots.pop(min(self._knots))
            rng = np.random.default_rng((self._seed, _WEATHER_STREAM, knot))
            self._knots[knot] = rng.standard_normal(self._lattice.shape)
        return self._knots[knot]

    def temperature_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Bilinearly interpolated temperature (temperature units) at continuous world positions."""
        return bilinear_sample(
            self.temperature,
            x,
            y,
            self.terrain.cell_size,
            self.terrain.world_width,
            self.terrain.world_height,
        )

    def zone_labels(self) -> np.ndarray:
        """Per-cell climate zone name, shape (height, width), dtype '<U9'.

        Derived from `temperature` for display and metrics only — see module docstring.
        """
        return zone_at(self.temperature)


class _UpsamplingLattice:
    """A coarse lattice over the world, and the fixed weights that bilinearly spread it to the grid.

    The weights depend only on the grid geometry, so they are computed once; each tick then pays two
    gathers and two weighted sums over the grid, which is what keeps weather cheap (#159 measured
    ~1.4 ms at 256x256 against ~47 ms for noise gathered per cell).
    """

    def __init__(self, terrain: Terrain, spacing: float) -> None:
        rows, cols = terrain.heights.shape
        self.shape = (
            int(np.ceil(terrain.world_height / spacing)) + 2,
            int(np.ceil(terrain.world_width / spacing)) + 2,
        )
        self._rows = _axis_weights(rows, terrain.cell_size, spacing)
        self._cols = _axis_weights(cols, terrain.cell_size, spacing)

    def upsample(self, lattice: np.ndarray) -> np.ndarray:
        """(height, width) float64: `lattice` bilinearly interpolated onto the terrain grid."""
        c0, c1, wc = self._cols
        r0, r1, wr = self._rows
        along_cols = lattice[:, c0] * (1.0 - wc) + lattice[:, c1] * wc
        return along_cols[r0, :] * (1.0 - wr)[:, None] + along_cols[r1, :] * wr[:, None]


def _axis_weights(
    n_cells: int, cell_size: float, spacing: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each grid line along one axis: the lattice index either side, and the weight of the far one."""
    position = np.arange(n_cells, dtype=np.float64) * cell_size / spacing
    lower = np.floor(position).astype(np.int64)
    return lower, lower + 1, position - lower


def zone_at(temperature: np.ndarray) -> np.ndarray:
    """Classify a temperature field (temperature units) into named climate zones.

    Labels only exist for display and metrics; nothing in `core/` may branch on the string this
    returns (the abstraction contract this module owns — see module docstring).
    """
    labels = np.empty(temperature.shape, dtype="<U9")
    lower = -np.inf
    for name, upper in _ZONE_BANDS:
        labels[(temperature >= lower) & (temperature < upper)] = name
        lower = upper
    return labels


def _temperature_field(terrain: Terrain, config: ClimateConfig) -> np.ndarray:
    """(height, width) float64: a latitude band per row, cooled by elevation above sea level.

    The part of the field that does not move with time. Held at float64 so the seasonal and weather
    terms are added before the one rounding to the stored float32.
    """
    row_y = np.arange(terrain.heights.shape[0], dtype=np.float64) * terrain.cell_size
    latitude_temperature = config.equator_temperature - config.latitude_gradient * np.abs(
        row_y - config.equator_y
    )
    elevation_above_sea_level = np.clip(
        terrain.heights.astype(np.float64) - config.sea_level_elevation, 0.0, None
    )
    altitude_drop = config.lapse_rate * elevation_above_sea_level
    return latitude_temperature[:, None] - altitude_drop
