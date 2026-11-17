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
        max_elevation=32.0,  # about a tenth of the 330-unit extent (#112)
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
                # A node sits at its cell's centre (#162).
                x, y = (col + 0.5) * 2.0, (row + 0.5) * 2.0
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
        # Halfway between column 0's centre (height 0) and column 1's (height 10).
        assert terrain.elevation_at(1.0, 0.5) == pytest.approx(5.0)

    def test_accepts_vectorized_positions(self):
        heights = np.array(
            [
                [0.0, 10.0],
                [20.0, 30.0],
            ],
            dtype=np.float32,
        )
        terrain = Terrain(heights, cell_size=1.0)
        xs = np.array([0.5, 1.5, 0.5])
        ys = np.array([0.5, 0.5, 1.5])
        result = terrain.elevation_at(xs, ys)
        assert np.allclose(result, [0.0, 10.0, 20.0])

    def test_raises_outside_bounds(self):
        heights = np.zeros((3, 3), dtype=np.float32)
        terrain = Terrain(heights, cell_size=1.0)
        with pytest.raises(ValueError):
            terrain.elevation_at(-0.1, 0.0)
        with pytest.raises(ValueError):
            terrain.elevation_at(0.0, 10.0)


class TestTheWorldSpansCellEdgeToCellEdge:
    """Every cell is a whole cell (#162). With nodes on the world's edge a rim cell held half a
    cell of world and a corner a quarter, so whatever a cell holds per cell (growth, soil, a scent
    deposit) was twice as dense per unit area along the rim."""

    def test_the_extent_is_one_cell_per_column_and_row(self):
        terrain = Terrain(np.zeros((4, 6), dtype=np.float32), cell_size=2.0)

        assert terrain.world_width == 12.0
        assert terrain.world_height == 8.0

    def test_a_uniform_population_falls_evenly_into_every_cell(self):
        """The defect itself: each cell, rim and corner included, catches its area's share."""
        terrain = Terrain(np.zeros((8, 8), dtype=np.float32), cell_size=1.0)
        # A regular lattice of positions covering the world evenly, 10 per unit on each axis, so
        # the count each cell catches is exact rather than a sample.
        along = (np.arange(80) + 0.5) / 10.0
        x, y = (grid.ravel() for grid in np.meshgrid(along, along))

        rows, cols = terrain.cell_indices(x, y)
        counts = np.bincount(rows * 8 + cols, minlength=64)

        assert (counts == 100).all()

    def test_a_cell_owns_the_half_open_span_from_its_west_edge(self):
        terrain = Terrain(np.zeros((4, 4), dtype=np.float32), cell_size=1.0)

        rows, cols = terrain.cell_indices(np.array([0.0, 0.999, 1.0, 3.5]), np.zeros(4))

        assert cols.tolist() == [0, 0, 1, 3]

    def test_the_far_rim_belongs_to_the_last_cell(self):
        """The world is closed, and `Movement` puts animals exactly on its far edge."""
        terrain = Terrain(np.zeros((4, 4), dtype=np.float32), cell_size=1.0)

        rows, cols = terrain.cell_indices(np.array([4.0]), np.array([4.0]))

        assert (rows[0], cols[0]) == (3, 3)

    def test_the_outer_half_cell_reads_the_rim_cell_flat(self):
        """There is no centre beyond the rim to interpolate toward, so the reading holds."""
        heights = np.array([[0.0, 10.0, 20.0]] * 3, dtype=np.float32)
        terrain = Terrain(heights, cell_size=1.0)

        readings = terrain.elevation_at(np.array([0.0, 0.25, 0.5, 2.5, 2.75, 3.0]), np.full(6, 1.5))

        assert readings.tolist() == [0.0, 0.0, 0.0, 20.0, 20.0, 20.0]

    def test_cell_centres_are_where_the_nodes_sit(self):
        terrain = Terrain(np.zeros((2, 3), dtype=np.float32), cell_size=2.0)

        assert terrain.column_x.tolist() == [1.0, 3.0, 5.0]
        assert terrain.row_y.tolist() == [1.0, 3.0]


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
