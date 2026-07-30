"""Cost-aware diffusion: how a signal spreads over ground that is expensive to cross (#93).

The contract is checkable in advance, so these were written before the implementation (§8.1). What
is not asserted is any particular falloff figure — that is tuning, and the operator's job is only
that the ordering it produces is the one the terrain implies.
"""

import numpy as np
import pytest

from core.world.diffusion import CostAwareDiffusion, DiffusionConfig
from core.world.terrain import Terrain


GRID = 41
CELL_SIZE = 1.0
CENTRE = GRID // 2


def flat(elevation=0.0):
    return Terrain(np.full((GRID, GRID), elevation, dtype=np.float32), cell_size=CELL_SIZE)


def wall(height, column, thickness=1):
    """Flat ground split by a north-south wall `thickness` columns wide."""
    heights = np.zeros((GRID, GRID), dtype=np.float32)
    heights[:, column : column + thickness] = height
    return Terrain(heights, cell_size=CELL_SIZE)


def point_source(row=CENTRE, column=CENTRE, strength=1.0):
    source = np.zeros((GRID, GRID), dtype=np.float32)
    source[row, column] = strength
    return source


def config(**overrides):
    params = dict(range=4.0, climb_penalty=2.0)
    params.update(overrides)
    return DiffusionConfig(**params)


class TestConfigValidation:
    @pytest.mark.parametrize("bad_range", [0.0, -1.0])
    def test_rejects_a_non_positive_range(self, bad_range):
        with pytest.raises(ValueError, match="range"):
            config(range=bad_range)

    def test_rejects_a_negative_climb_penalty(self):
        """Negative would make a signal carry *further* uphill, which is the barrier inverted."""
        with pytest.raises(ValueError, match="climb_penalty"):
            config(climb_penalty=-0.5)

    def test_a_zero_climb_penalty_is_allowed(self):
        """A world where relief does not impede perception is a legitimate world, and it is the
        control every test of the terrain-aware behaviour compares against."""
        assert config(climb_penalty=0.0).climb_penalty == 0.0


class TestOnFlatGround:
    def test_a_signal_falls_off_with_distance(self):
        field = CostAwareDiffusion(flat(), config()).spread(point_source())

        near = field[CENTRE, CENTRE + 1]
        far = field[CENTRE, CENTRE + 6]
        assert field[CENTRE, CENTRE] > near > far > 0.0

    def test_falloff_is_the_same_in_every_direction(self):
        field = CostAwareDiffusion(flat(), config()).spread(point_source())

        offsets = [(0, 3), (0, -3), (3, 0), (-3, 0)]
        readings = [field[CENTRE + dr, CENTRE + dc] for dr, dc in offsets]
        assert readings == pytest.approx([readings[0]] * 4, rel=1e-5)

    def test_elevation_alone_changes_nothing(self):
        """A plateau is not a barrier. Only *relief between cells* may matter, or a world's
        absolute height above its own datum would silently rescale every perception in it."""
        low = CostAwareDiffusion(flat(0.0), config()).spread(point_source())
        high = CostAwareDiffusion(flat(500.0), config()).spread(point_source())

        np.testing.assert_allclose(low, high, rtol=1e-6)

    def test_a_wider_range_carries_further(self):
        near_sighted = CostAwareDiffusion(flat(), config(range=2.0)).spread(point_source())
        far_sighted = CostAwareDiffusion(flat(), config(range=8.0)).spread(point_source())

        probe = (CENTRE, CENTRE + 7)
        assert far_sighted[probe] > near_sighted[probe]

    def test_sources_add(self):
        """Two meadows either side must read as more food than one, or a forager between them
        would be drawn to whichever it happened to sample rather than to the better ground."""
        operator = CostAwareDiffusion(flat(), config())
        one = operator.spread(point_source(column=CENTRE - 2))
        both = operator.spread(
            point_source(column=CENTRE - 2) + point_source(column=CENTRE + 2)
        )

        assert both[CENTRE, CENTRE] > one[CENTRE, CENTRE]

    def test_an_empty_source_stays_empty(self):
        field = CostAwareDiffusion(flat(), config()).spread(
            np.zeros((GRID, GRID), dtype=np.float32)
        )

        assert (field == 0.0).all()

    def test_nothing_is_created_anywhere(self):
        """A spreading operator may move a signal around and damp it; it may never amplify one,
        or a gradient would point at an artifact of the arithmetic."""
        field = CostAwareDiffusion(flat(), config()).spread(point_source(strength=1.0))

        assert field.max() == pytest.approx(field[CENTRE, CENTRE])
        assert (field >= 0.0).all()


class TestTerrainImpedes:
    def test_a_ridge_damps_what_is_behind_it(self):
        """The whole point (#93): food across a ridge reads fainter than food the same distance
        away over open ground, because reaching it costs more."""
        column = CENTRE + 3
        source = point_source(column=CENTRE + 6)

        open_ground = CostAwareDiffusion(flat(), config()).spread(source)
        blocked = CostAwareDiffusion(wall(height=8.0, column=column), config()).spread(source)

        assert blocked[CENTRE, CENTRE] < open_ground[CENTRE, CENTRE]

    def test_a_higher_wall_damps_more(self):
        source = point_source(column=CENTRE + 6)
        low = CostAwareDiffusion(wall(2.0, CENTRE + 3), config()).spread(source)
        high = CostAwareDiffusion(wall(20.0, CENTRE + 3), config()).spread(source)

        assert high[CENTRE, CENTRE] < low[CENTRE, CENTRE]

    def test_a_signal_routes_around_a_barrier_rather_than_through_it(self):
        """A wall that does not span the world is a detour, not a wall. What arrives behind it
        should be what came round the end — which is what makes a fence (#27) a real intervention
        rather than a multiplier."""
        heights = np.zeros((GRID, GRID), dtype=np.float32)
        heights[: GRID - 4, CENTRE + 3] = 30.0  # a gap at the southern end
        terrain = Terrain(heights, cell_size=CELL_SIZE)

        field = CostAwareDiffusion(terrain, config(range=12.0)).spread(
            point_source(row=CENTRE, column=CENTRE + 6)
        )

        # Behind the wall, the reading near the gap beats the reading level with the source.
        assert field[GRID - 2, CENTRE] > field[CENTRE, CENTRE]

    def test_downhill_carries_further_than_uphill(self):
        """Only *gain* is charged, exactly as §2.5 prices a step: an animal in a valley perceives
        the slope above it less readily than one on the rim perceives the valley floor."""
        x = np.arange(GRID, dtype=np.float32) * CELL_SIZE
        slope = np.broadcast_to(x * 2.0, (GRID, GRID)).astype(np.float32)
        operator = CostAwareDiffusion(Terrain(slope, CELL_SIZE), config())

        from_above = operator.spread(point_source(column=CENTRE + 5))
        from_below = operator.spread(point_source(column=CENTRE - 5))

        # The source uphill has to be reached by climbing; the one downhill does not.
        assert from_below[CENTRE, CENTRE] > from_above[CENTRE, CENTRE]

    def test_with_no_climb_penalty_terrain_stops_mattering(self):
        source = point_source(column=CENTRE + 6)
        flat_field = CostAwareDiffusion(flat(), config(climb_penalty=0.0)).spread(source)
        walled = CostAwareDiffusion(
            wall(20.0, CENTRE + 3), config(climb_penalty=0.0)
        ).spread(source)

        np.testing.assert_allclose(flat_field, walled, rtol=1e-6)


class TestTheGradientPointsAtFood:
    def test_the_gradient_points_toward_a_lone_source(self):
        operator = CostAwareDiffusion(flat(), config())
        field = operator.spread(point_source(column=CENTRE + 5))

        # Standing west of the source, looking east.
        gx, gy = operator.gradient_at(field, np.array([CENTRE * 1.0]), np.array([CENTRE * 1.0]))

        assert gx[0] > 0.0
        assert gy[0] == pytest.approx(0.0, abs=1e-6)

    def test_the_gradient_prefers_the_richer_of_two_sources(self):
        operator = CostAwareDiffusion(flat(), config())
        field = operator.spread(
            point_source(column=CENTRE - 4, strength=1.0)
            + point_source(column=CENTRE + 4, strength=8.0)
        )

        gx, _ = operator.gradient_at(field, np.array([CENTRE * 1.0]), np.array([CENTRE * 1.0]))

        assert gx[0] > 0.0

    def test_the_gradient_prefers_the_nearer_of_two_equal_sources(self):
        operator = CostAwareDiffusion(flat(), config())
        field = operator.spread(
            point_source(column=CENTRE - 2) + point_source(column=CENTRE + 8)
        )

        gx, _ = operator.gradient_at(field, np.array([CENTRE * 1.0]), np.array([CENTRE * 1.0]))

        assert gx[0] < 0.0

    def test_the_gradient_prefers_the_cheaper_of_two_equal_sources(self):
        """The distance discount and the climb discount are the same mechanism, which is the whole
        argument for folding them into one operator: equally distant meadows are not equally
        attractive if one is behind a ridge."""
        heights = np.zeros((GRID, GRID), dtype=np.float32)
        heights[:, CENTRE + 2] = 25.0
        operator = CostAwareDiffusion(Terrain(heights, CELL_SIZE), config())
        field = operator.spread(
            point_source(column=CENTRE - 5) + point_source(column=CENTRE + 5)
        )

        gx, _ = operator.gradient_at(field, np.array([CENTRE * 1.0]), np.array([CENTRE * 1.0]))

        assert gx[0] < 0.0, "the gradient pointed at the meadow behind the ridge"

    def test_an_empty_field_has_no_gradient(self):
        operator = CostAwareDiffusion(flat(), config())
        field = operator.spread(np.zeros((GRID, GRID), dtype=np.float32))

        gx, gy = operator.gradient_at(field, np.array([10.0]), np.array([10.0]))

        assert gx[0] == 0.0 and gy[0] == 0.0

    def test_the_gradient_is_read_for_a_whole_population_at_once(self):
        operator = CostAwareDiffusion(flat(), config())
        field = operator.spread(point_source(column=CENTRE + 5))
        x = np.full(32, float(CENTRE))
        y = np.linspace(CENTRE - 3.0, CENTRE + 3.0, 32)

        gx, gy = operator.gradient_at(field, x, y)

        assert gx.shape == (32,) and gy.shape == (32,)
        assert (gx > 0.0).all()

    def test_a_position_on_the_world_edge_is_readable(self):
        """Foragers reach the boundary — `Movement._landing` puts them exactly on it — so the
        gradient has to be defined there rather than raising or reading off-grid."""
        operator = CostAwareDiffusion(flat(), config())
        field = operator.spread(point_source(row=CENTRE, column=CENTRE))
        edge = (GRID - 1) * CELL_SIZE

        gx, gy = operator.gradient_at(field, np.array([0.0, edge]), np.array([0.0, edge]))

        assert np.isfinite(gx).all() and np.isfinite(gy).all()
