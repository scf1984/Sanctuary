import numpy as np
import pytest

from core.world.terrain import Terrain, TerrainConfig
from core.world.water import Water


def bowl(rows=7, cols=7, rim=10.0, cell_size=1.0):
    """A basin with a unique minimum (no flat bottom), so flow routing within it is unambiguous."""
    r, c = np.indices((rows, cols))
    center_r, center_c = (rows - 1) / 2, (cols - 1) / 2
    # Squared distance from center gives a smooth bowl; a tiny asymmetric term breaks any
    # possible tie between cells equidistant from the center so the minimum is unique.
    heights = (r - center_r) ** 2 + (c - center_c) ** 2 + 0.01 * c
    heights = rim - (heights.max() - heights) * (rim / heights.max())
    return Terrain(heights.astype(np.float32), cell_size=cell_size)


class TestPooling:
    def test_global_maximum_never_pools_water(self):
        terrain = Terrain.generate(
            TerrainConfig(width=25, height=25, seed=3, min_elevation=0.0, max_elevation=800.0)
        )
        water = Water.generate(terrain)
        peak_row, peak_col = np.unravel_index(np.argmax(terrain.heights), terrain.heights.shape)
        assert water.depth[peak_row, peak_col] == 0.0

    def test_basin_interior_pools_water(self):
        terrain = bowl()
        water = Water.generate(terrain)
        center = (terrain.heights.shape[0] // 2, terrain.heights.shape[1] // 2)
        assert water.depth[center] > 0.0

    def test_basin_rim_stays_dry(self):
        terrain = bowl()
        water = Water.generate(terrain)
        assert water.depth[0, 0] == 0.0
        assert water.depth[-1, -1] == 0.0

    def test_flat_terrain_has_no_water(self):
        heights = np.full((6, 6), 12.0, dtype=np.float32)
        terrain = Terrain(heights, cell_size=1.0)
        water = Water.generate(terrain)
        assert np.all(water.depth == 0.0)

    def test_depth_never_negative(self):
        terrain = Terrain.generate(TerrainConfig(width=20, height=20, seed=11))
        water = Water.generate(terrain)
        assert np.all(water.depth >= 0.0)

    def test_deeper_basin_pools_more_than_shallow_basin(self):
        shallow = Water.generate(bowl(rim=5.0))
        deep = Water.generate(bowl(rim=50.0))
        shallow_center = shallow.depth.shape[0] // 2, shallow.depth.shape[1] // 2
        deep_center = deep.depth.shape[0] // 2, deep.depth.shape[1] // 2
        assert deep.depth[deep_center] > shallow.depth[shallow_center]


class TestFlowDirection:
    def test_ramp_flows_toward_low_end(self):
        # Height decreases as column increases (col 7 is lowest); every interior cell should
        # drain toward a neighbour with a strictly larger column index.
        cols = np.arange(8, dtype=np.float32)
        heights = np.tile(20.0 - cols, (8, 1))
        terrain = Terrain(heights, cell_size=1.0)
        water = Water.generate(terrain)
        for r in range(1, 7):
            for c in range(1, 7):
                idx = water.flow_direction[r, c]
                assert idx != -1
                dr, dc = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))[idx]
                assert dc >= 0  # never flows uphill, back toward higher ground

    def test_low_map_edge_is_an_outlet(self):
        # The lowest column has no lower in-grid neighbour and no filling occurs (the ramp is
        # already monotonic), so it should be flagged as where water leaves the world.
        cols = np.arange(8, dtype=np.float32)
        heights = np.tile(20.0 - cols, (8, 1))
        terrain = Terrain(heights, cell_size=1.0)
        water = Water.generate(terrain)
        assert np.all(water.flow_direction[1:7, 7] == -1)

    def test_every_cell_flows_to_equal_or_lower_ground(self):
        terrain = Terrain.generate(TerrainConfig(width=15, height=15, seed=5))
        water = Water.generate(terrain)
        offsets = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
        filled = terrain.heights + water.depth
        rows, cols = filled.shape
        for r in range(rows):
            for c in range(cols):
                idx = water.flow_direction[r, c]
                if idx == -1:
                    continue
                dr, dc = offsets[idx]
                assert filled[r + dr, c + dc] <= filled[r, c] + 1e-4


class TestFlowAccumulation:
    def test_every_cell_accumulates_at_least_itself(self):
        terrain = Terrain.generate(TerrainConfig(width=12, height=12, seed=9))
        water = Water.generate(terrain)
        assert np.all(water.flow_accumulation >= 1.0)

    def test_converging_ramps_accumulate_more_downstream(self):
        # Two slopes meeting in a shared central channel: water from both sides should
        # accumulate into the channel, so its accumulation exceeds either slope's cells alone.
        rows, cols = 9, 9
        r, c = np.indices((rows, cols))
        heights = np.abs(c - cols // 2).astype(np.float32)
        heights += (rows - 1 - r).astype(np.float32) * 0.01
        terrain = Terrain(heights, cell_size=1.0)
        water = Water.generate(terrain)
        channel_accum = water.flow_accumulation[rows - 1, cols // 2]
        off_channel_accum = water.flow_accumulation[rows - 1, 0]
        assert channel_accum > off_channel_accum

    def test_total_accumulation_bounded_by_grid_size(self):
        terrain = Terrain.generate(TerrainConfig(width=10, height=10, seed=2))
        water = Water.generate(terrain)
        assert water.flow_accumulation.max() <= 100.0


class TestQueries:
    def test_depth_at_matches_grid_points(self):
        terrain = bowl()
        water = Water.generate(terrain)
        for row in range(terrain.heights.shape[0]):
            for col in range(terrain.heights.shape[1]):
                x, y = col * terrain.cell_size, row * terrain.cell_size
                assert water.depth_at(x, y) == pytest.approx(water.depth[row, col])

    def test_is_drinkable_true_in_basin_false_on_rim(self):
        terrain = bowl()
        water = Water.generate(terrain)
        center_x = (terrain.heights.shape[1] // 2) * terrain.cell_size
        center_y = (terrain.heights.shape[0] // 2) * terrain.cell_size
        assert water.is_drinkable_at(center_x, center_y)
        assert not water.is_drinkable_at(0.0, 0.0)

    def test_raises_outside_bounds(self):
        terrain = bowl()
        water = Water.generate(terrain)
        with pytest.raises(ValueError):
            water.depth_at(-0.1, 0.0)
        with pytest.raises(ValueError):
            water.depth_at(0.0, 1000.0)

    def test_accepts_vectorized_positions(self):
        terrain = bowl()
        water = Water.generate(terrain)
        xs = np.array([0.0, terrain.cell_size])
        ys = np.array([0.0, 0.0])
        result = water.depth_at(xs, ys)
        assert result.shape == (2,)


class TestConstructionAndRoundTrip:
    def test_rejects_mismatched_shapes(self):
        depth = np.zeros((3, 3), dtype=np.float32)
        bad_direction = np.zeros((3, 4), dtype=np.int8)
        accumulation = np.ones((3, 3), dtype=np.float32)
        with pytest.raises(ValueError):
            Water(depth, bad_direction, accumulation, cell_size=1.0)

    def test_round_trips_through_raw_arrays(self):
        terrain = bowl()
        original = Water.generate(terrain)
        restored = Water(
            original.depth, original.flow_direction, original.flow_accumulation, original.cell_size
        )
        assert np.array_equal(restored.depth, original.depth)
        assert np.array_equal(restored.flow_direction, original.flow_direction)
        assert np.array_equal(restored.flow_accumulation, original.flow_accumulation)
