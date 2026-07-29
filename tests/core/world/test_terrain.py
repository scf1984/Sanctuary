import numpy as np
import pytest

from core.world.terrain import Terrain, TerrainConfig


def make_config(**overrides):
    params = dict(
        width=33,
        height=33,
        cell_size=10.0,
        seed=42,
        octaves=5,
        persistence=0.5,
        min_elevation=0.0,
        max_elevation=32.0,  # a tenth of the 320-unit extent (#112)
    )
    params.update(overrides)
    return TerrainConfig(**params)


class TestTerrainConfig:
    def test_rejects_too_small_grid(self):
        with pytest.raises(ValueError):
            make_config(width=1)

    def test_rejects_non_positive_cell_size(self):
        with pytest.raises(ValueError):
            make_config(cell_size=0)

    def test_rejects_zero_octaves(self):
        with pytest.raises(ValueError):
            make_config(octaves=0)

    def test_rejects_persistence_out_of_range(self):
        with pytest.raises(ValueError):
            make_config(persistence=1.0)

    def test_rejects_inverted_elevation_range(self):
        with pytest.raises(ValueError):
            make_config(min_elevation=32.0, max_elevation=0.0)


class TestGeneration:
    def test_repeatable_from_same_seed(self):
        a = Terrain.generate(make_config(seed=7))
        b = Terrain.generate(make_config(seed=7))
        assert np.array_equal(a.heights, b.heights)

    def test_differs_across_seeds(self):
        a = Terrain.generate(make_config(seed=1))
        b = Terrain.generate(make_config(seed=2))
        assert not np.array_equal(a.heights, b.heights)

    def test_relief_spans_configured_range(self):
        terrain = Terrain.generate(make_config(min_elevation=10.0, max_elevation=90.0))
        assert terrain.heights.min() == pytest.approx(10.0, abs=1e-3)
        assert terrain.heights.max() == pytest.approx(90.0, abs=1e-3)

    def test_relief_is_varied_not_flat(self):
        terrain = Terrain.generate(make_config())
        assert terrain.heights.std() > 0

    def test_shape_matches_config(self):
        terrain = Terrain.generate(make_config(width=17, height=25))
        assert terrain.heights.shape == (25, 17)


class TestElevationLookup:
    def test_matches_grid_points_exactly(self):
        heights = np.array(
            [
                [0.0, 10.0, 20.0],
                [5.0, 15.0, 25.0],
            ],
            dtype=np.float32,
        )
        terrain = Terrain(heights, cell_size=2.0)
        for row in range(2):
            for col in range(3):
                x, y = col * 2.0, row * 2.0
                assert terrain.elevation_at(x, y) == pytest.approx(heights[row, col])

    def test_interpolates_midpoint(self):
        heights = np.array(
            [
                [0.0, 10.0],
                [0.0, 10.0],
            ],
            dtype=np.float32,
        )
        terrain = Terrain(heights, cell_size=1.0)
        # Halfway between column 0 (height 0) and column 1 (height 10).
        assert terrain.elevation_at(0.5, 0.0) == pytest.approx(5.0)

    def test_accepts_vectorized_positions(self):
        heights = np.array(
            [
                [0.0, 10.0],
                [20.0, 30.0],
            ],
            dtype=np.float32,
        )
        terrain = Terrain(heights, cell_size=1.0)
        xs = np.array([0.0, 1.0, 0.0])
        ys = np.array([0.0, 0.0, 1.0])
        result = terrain.elevation_at(xs, ys)
        assert np.allclose(result, [0.0, 10.0, 20.0])

    def test_raises_outside_bounds(self):
        heights = np.zeros((3, 3), dtype=np.float32)
        terrain = Terrain(heights, cell_size=1.0)
        with pytest.raises(ValueError):
            terrain.elevation_at(-0.1, 0.0)
        with pytest.raises(ValueError):
            terrain.elevation_at(0.0, 10.0)


class TestSlopeAndAspect:
    def test_flat_terrain_has_zero_slope(self):
        heights = np.full((5, 5), 42.0, dtype=np.float32)
        terrain = Terrain(heights, cell_size=1.0)
        assert np.allclose(terrain.slope, 0.0)

    def test_slope_and_aspect_shapes_match_heights(self):
        terrain = Terrain.generate(make_config())
        assert terrain.slope.shape == terrain.heights.shape
        assert terrain.aspect.shape == terrain.heights.shape

    def test_steeper_ramp_has_larger_slope(self):
        gentle = Terrain(
            np.array([[0.0, 1.0, 2.0]] * 3, dtype=np.float32), cell_size=1.0
        )
        steep = Terrain(
            np.array([[0.0, 10.0, 20.0]] * 3, dtype=np.float32), cell_size=1.0
        )
        assert steep.slope[1, 1] > gentle.slope[1, 1]
