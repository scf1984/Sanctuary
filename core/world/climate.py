"""Temperature field derived from altitude and latitude, and climate zones derived from it.

Elevation and world position drive temperature (CLAUDE.md §2.6), so climate zones fall out of
terrain rather than being painted regions. This module owns the abstraction contract from the
issue that introduced it: the temperature field is queryable per continuous position, and zone
labels are derived from that field after the fact, for display and metrics only — no simulation
logic may branch on a zone name. Systems that need the field for real effects (thermoregulation
cost in metabolism, plant productivity) query `Climate.temperature_at` directly, never a label.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.world.terrain import Terrain, bilinear_sample

# Ordered (label, upper bound °C, exclusive) pairs. A cell's zone is the first bucket whose
# upper bound exceeds its temperature. Boundaries are round, illustrative values for grouping
# a continuous field into legible bands — display and metrics only (see module docstring).
_ZONE_BANDS: tuple[tuple[str, float], ...] = (
    ("polar", -10.0),
    ("tundra", 0.0),
    ("temperate", 20.0),
    ("tropical", float("inf")),
)


@dataclass(frozen=True)
class ClimateConfig:
    """Parameters mapping terrain position to temperature, degrees Celsius.

    equator_y: world-unit y coordinate of peak (equatorial) temperature. Temperature falls off
        with distance from this line in both directions, so it models one hemisphere's worth of
        latitude band across the world's y extent rather than a full planet.
    equator_temperature: temperature at the equator at `sea_level_elevation`.
    latitude_gradient: degrees C lost per world unit of distance from `equator_y`.
    sea_level_elevation: elevation (meters) at which the altitude lapse rate is zero.
    lapse_rate: degrees C lost per meter of elevation above `sea_level_elevation`. Defaults to
        Earth's ~6.5 °C/km tropospheric lapse rate as a plausible starting point, not a
        measured claim about this simulation.
    """

    equator_y: float
    equator_temperature: float = 30.0
    latitude_gradient: float = 0.05
    sea_level_elevation: float = 0.0
    lapse_rate: float = 0.0065

    def __post_init__(self) -> None:
        if self.latitude_gradient < 0:
            raise ValueError("latitude_gradient must be non-negative")
        if self.lapse_rate < 0:
            raise ValueError("lapse_rate must be non-negative")


class Climate:
    """Owns the world's temperature field, derived from terrain elevation and world position.

    temperature: (height, width) float32, degrees Celsius — one value per terrain grid cell,
        aligned with `terrain.heights`.
    """

    def __init__(self, terrain: Terrain, config: ClimateConfig) -> None:
        self.terrain = terrain
        self.config = config
        self.temperature = _temperature_field(terrain, config)

    def temperature_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Bilinearly interpolated temperature (deg C) at continuous world positions."""
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


def zone_at(temperature: np.ndarray) -> np.ndarray:
    """Classify a temperature field (deg C) into named climate zones.

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
    """Vectorized temperature: a latitude band per row, cooled by elevation above sea level."""
    row_y = np.arange(terrain.heights.shape[0], dtype=np.float64) * terrain.cell_size
    latitude_temperature = config.equator_temperature - config.latitude_gradient * np.abs(
        row_y - config.equator_y
    )
    elevation_above_sea_level = np.clip(
        terrain.heights.astype(np.float64) - config.sea_level_elevation, 0.0, None
    )
    altitude_drop = config.lapse_rate * elevation_above_sea_level
    temperature = latitude_temperature[:, None] - altitude_drop
    return temperature.astype(np.float32)
