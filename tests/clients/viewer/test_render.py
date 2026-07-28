import numpy as np
import pytest

from clients.viewer.render import (
    apply_water_overlay,
    elevation_shading,
    interpolate_positions,
    species_colors,
    world_to_screen,
)
from core.world.terrain import Terrain
from core.world.water import Water


def flat_terrain(value=50.0, rows=6, cols=6, cell_size=1.0):
    return Terrain(np.full((rows, cols), value, dtype=np.float32), cell_size=cell_size)


def ramp_terrain(rows=6, cols=6, cell_size=1.0, low=0.0, high=100.0):
    """A constant-gradient ramp: slope and aspect are uniform everywhere, so hillshade brightness
    does not vary across the grid and elevation color alone drives the result."""
    heights = np.linspace(low, high, cols, dtype=np.float32)[None, :].repeat(rows, axis=0)
    return Terrain(heights, cell_size=cell_size)


class TestElevationShading:
    def test_shape_and_dtype(self):
        terrain = flat_terrain(rows=5, cols=7)
        shaded = elevation_shading(terrain)
        assert shaded.shape == (5, 7, 3)
        assert shaded.dtype == np.uint8

    def test_flat_terrain_is_uniform_color(self):
        shaded = elevation_shading(flat_terrain())
        assert np.all(shaded == shaded[0, 0])

    def test_lowest_point_matches_low_elevation_color(self):
        terrain = ramp_terrain()
        shaded = elevation_shading(terrain)
        # Uniform slope -> uniform brightness, so the lowest column matches the low-elevation
        # stop's hue ratios (green-dominant) rather than the high stop's (near-white).
        low_pixel = shaded[0, 0].astype(np.int16)
        assert low_pixel[1] > low_pixel[0]  # green channel exceeds red at the low stop

    def test_higher_elevation_is_brighter_toward_white(self):
        terrain = ramp_terrain()
        shaded = elevation_shading(terrain).astype(np.int16)
        low_brightness = shaded[0, 0].sum()
        high_brightness = shaded[0, -1].sum()
        assert high_brightness > low_brightness


class TestWaterOverlay:
    def test_dry_terrain_leaves_base_unchanged(self):
        terrain = flat_terrain()
        water = Water.generate(terrain)
        base = elevation_shading(terrain)
        overlaid = apply_water_overlay(base, water)
        assert np.array_equal(overlaid, base)

    def test_deep_water_tends_toward_water_color(self):
        terrain = flat_terrain()
        depth = np.full(terrain.heights.shape, 10.0, dtype=np.float32)
        flow_direction = np.full(terrain.heights.shape, -1, dtype=np.int8)
        flow_accumulation = np.ones(terrain.heights.shape, dtype=np.float32)
        water = Water(depth, flow_direction, flow_accumulation, terrain.cell_size)
        base = np.zeros(terrain.heights.shape + (3,), dtype=np.uint8)
        overlaid = apply_water_overlay(base, water)
        # Deep water saturates toward blue: the blue channel should dominate red and green.
        assert overlaid[0, 0, 2] > overlaid[0, 0, 0]
        assert overlaid[0, 0, 2] > overlaid[0, 0, 1]

    def test_shape_and_dtype_preserved(self):
        terrain = flat_terrain(rows=4, cols=4)
        water = Water.generate(terrain)
        base = elevation_shading(terrain)
        overlaid = apply_water_overlay(base, water)
        assert overlaid.shape == base.shape
        assert overlaid.dtype == np.uint8


class TestSpeciesColors:
    def test_unset_species_is_gray(self):
        colors = species_colors(np.array([-1, -1]))
        assert np.array_equal(colors[0], [128, 128, 128])
        assert np.array_equal(colors[1], [128, 128, 128])

    def test_same_id_is_deterministic(self):
        first = species_colors(np.array([7]))
        second = species_colors(np.array([7]))
        assert np.array_equal(first, second)

    def test_different_ids_get_different_colors(self):
        colors = species_colors(np.array([0, 1, 2]))
        assert not np.array_equal(colors[0], colors[1])
        assert not np.array_equal(colors[1], colors[2])

    def test_shape_and_dtype(self):
        colors = species_colors(np.array([0, 1, -1]))
        assert colors.shape == (3, 3)
        assert colors.dtype == np.uint8


class TestInterpolatePositions:
    def _positions(self, values):
        arr = np.array(values, dtype=np.float32)
        return (arr, arr.copy(), arr.copy())

    def test_alpha_zero_returns_previous(self):
        previous = self._positions([0.0, 0.0])
        current = self._positions([10.0, 10.0])
        result = interpolate_positions(previous, current, 0.0)
        for axis in result:
            assert np.allclose(axis, [0.0, 0.0])

    def test_alpha_one_returns_current(self):
        previous = self._positions([0.0, 0.0])
        current = self._positions([10.0, 10.0])
        result = interpolate_positions(previous, current, 1.0)
        for axis in result:
            assert np.allclose(axis, [10.0, 10.0])

    def test_alpha_half_returns_midpoint(self):
        previous = self._positions([0.0])
        current = self._positions([10.0])
        result = interpolate_positions(previous, current, 0.5)
        for axis in result:
            assert np.allclose(axis, [5.0])

    def test_out_of_range_alpha_raises(self):
        previous = self._positions([0.0])
        current = self._positions([1.0])
        with pytest.raises(ValueError):
            interpolate_positions(previous, current, 1.5)
        with pytest.raises(ValueError):
            interpolate_positions(previous, current, -0.1)


class TestWorldToScreen:
    def test_origin_maps_to_origin(self):
        px, py = world_to_screen(
            np.array([0.0]), np.array([0.0]), world_width=100.0, world_height=50.0,
            screen_width=800, screen_height=400,
        )
        assert px[0] == 0
        assert py[0] == 0

    def test_far_corner_maps_to_screen_extent(self):
        px, py = world_to_screen(
            np.array([100.0]), np.array([50.0]), world_width=100.0, world_height=50.0,
            screen_width=800, screen_height=400,
        )
        assert px[0] == 800
        assert py[0] == 400

    def test_midpoint_maps_to_screen_midpoint(self):
        px, py = world_to_screen(
            np.array([50.0]), np.array([25.0]), world_width=100.0, world_height=50.0,
            screen_width=800, screen_height=400,
        )
        assert px[0] == 400
        assert py[0] == 200
