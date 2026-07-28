import numpy as np
import pytest

from core.world.climate import Climate, ClimateConfig, zone_at
from core.world.terrain import Terrain, TerrainConfig


def make_terrain(**overrides):
    params = dict(
        width=33,
        height=33,
        cell_size=10.0,
        seed=42,
        octaves=5,
        persistence=0.5,
        min_elevation=0.0,
        max_elevation=2000.0,
    )
    params.update(overrides)
    return Terrain.generate(TerrainConfig(**params))


def make_climate_config(terrain, **overrides):
    params = dict(equator_y=terrain.world_height / 2)
    params.update(overrides)
    return ClimateConfig(**params)


class TestClimateConfig:
    def test_rejects_negative_latitude_gradient(self):
        with pytest.raises(ValueError):
            ClimateConfig(equator_y=0.0, latitude_gradient=-0.1)

    def test_rejects_negative_lapse_rate(self):
        with pytest.raises(ValueError):
            ClimateConfig(equator_y=0.0, lapse_rate=-0.1)


class TestTemperatureField:
    def test_temperature_falls_with_altitude(self):
        # Flat latitude (all rows at the equator) isolates the altitude effect.
        heights = np.array([[0.0, 1000.0, 2000.0]] * 3, dtype=np.float32)
        terrain = Terrain(heights, cell_size=1.0)
        config = ClimateConfig(equator_y=terrain.world_height / 2, latitude_gradient=0.0)
        climate = Climate(terrain, config)

        row = climate.temperature[1]
        assert row[0] > row[1] > row[2]

    def test_temperature_varies_with_latitude(self):
        # Flat terrain isolates the latitude effect.
        heights = np.zeros((5, 3), dtype=np.float32)
        terrain = Terrain(heights, cell_size=10.0)
        config = ClimateConfig(equator_y=terrain.world_height / 2)
        climate = Climate(terrain, config)

        equator_row = climate.temperature.shape[0] // 2
        assert climate.temperature[equator_row, 0] > climate.temperature[0, 0]
        assert climate.temperature[equator_row, 0] > climate.temperature[-1, 0]

    def test_generated_world_has_varied_temperature(self):
        terrain = make_terrain()
        climate = Climate(terrain, make_climate_config(terrain))
        assert climate.temperature.std() > 0

    def test_shape_matches_terrain(self):
        terrain = make_terrain(width=17, height=25)
        climate = Climate(terrain, make_climate_config(terrain))
        assert climate.temperature.shape == terrain.heights.shape


class TestTemperatureLookup:
    def test_matches_grid_points_exactly(self):
        heights = np.zeros((3, 3), dtype=np.float32)
        terrain = Terrain(heights, cell_size=2.0)
        config = ClimateConfig(equator_y=0.0, latitude_gradient=0.0)
        climate = Climate(terrain, config)

        assert climate.temperature_at(0.0, 0.0) == pytest.approx(climate.temperature[0, 0])

    def test_accepts_vectorized_positions(self):
        terrain = make_terrain()
        climate = Climate(terrain, make_climate_config(terrain))
        xs = np.array([0.0, terrain.world_width])
        ys = np.array([0.0, terrain.world_height])
        result = climate.temperature_at(xs, ys)
        assert result.shape == (2,)

    def test_raises_outside_bounds(self):
        terrain = make_terrain()
        climate = Climate(terrain, make_climate_config(terrain))
        with pytest.raises(ValueError):
            climate.temperature_at(-1.0, 0.0)


class TestZones:
    def test_labels_shape_matches_temperature(self):
        terrain = make_terrain()
        climate = Climate(terrain, make_climate_config(terrain))
        assert climate.zone_labels().shape == climate.temperature.shape

    def test_distinct_zones_identifiable_in_generated_world(self):
        # A wide latitude span plus varied elevation should produce more than one named zone.
        terrain = make_terrain(height=65, max_elevation=4000.0)
        config = make_climate_config(terrain, equator_temperature=35.0, latitude_gradient=1.0)
        climate = Climate(terrain, config)

        assert len(set(climate.zone_labels().ravel().tolist())) > 1

    def test_zone_boundaries_are_ordered_by_temperature(self):
        temperature = np.array([-20.0, -5.0, 10.0, 25.0], dtype=np.float32)
        labels = zone_at(temperature)
        assert list(labels) == ["polar", "tundra", "temperate", "tropical"]
