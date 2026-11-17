"""How the view draws what the boundary hands it (#35). Inputs are `api` values, never a world.

The few tests that need a real world open one through `api.create` and reach into
`session._world` only to stage a state, exactly as the api's own tests do.
"""

import numpy as np
import pytest

import api
from api import BarrierView, Column, FieldView, Frame, TerrainView
from clients.viewer import render
from clients.viewer.render import (
    CONDITION_MODES,
    CONDITION_RAMP,
    FIELD_LAYERS,
    apply_field_overlay,
    apply_water_overlay,
    barrier_segments,
    condition_colors,
    drag_rectangle,
    elevation_shading,
    field_overlay,
    live_positions,
    species_colors,
    world_to_screen,
)
from core.world.barriers import Barriers
from core.world.terrain import Terrain


def terrain_view(heights, cell_size=1.0):
    """A `TerrainView` of a hand-built height grid, dry everywhere."""
    terrain = Terrain(np.asarray(heights, dtype=np.float32), cell_size=cell_size)
    return TerrainView(
        elevation=Column.of(terrain.heights),
        slope=Column.of(terrain.slope),
        aspect=Column.of(terrain.aspect),
        water_depth=Column.of(np.zeros_like(terrain.heights)),
    )


def flat(value=50.0, rows=6, cols=6):
    return terrain_view(np.full((rows, cols), value))


def ramp(rows=6, cols=6, low=0.0, high=100.0):
    """A constant-gradient ramp: slope and aspect are uniform everywhere, so hillshade brightness
    does not vary across the grid and elevation color alone drives the result."""
    return terrain_view(np.linspace(low, high, cols)[None, :].repeat(rows, axis=0))


def frame(ids, x, tick=0):
    """A frame whose y and z equal x and whose species are all 0: positions are what is tested."""
    xs = np.asarray(x, dtype=np.float32)
    return Frame(
        tick=tick,
        ids=Column.of(np.asarray(ids, dtype=np.int64)),
        x=Column.of(xs),
        y=Column.of(xs),
        z=Column.of(xs),
        species=Column.of(np.zeros(len(xs), dtype=np.int32)),
    )


class TestElevationShading:
    def test_shape_and_dtype(self):
        shaded = elevation_shading(flat(rows=5, cols=7))
        assert shaded.shape == (5, 7, 3)
        assert shaded.dtype == np.uint8

    def test_flat_terrain_is_uniform_color(self):
        shaded = elevation_shading(flat())
        assert np.all(shaded == shaded[0, 0])

    def test_lowest_point_matches_low_elevation_color(self):
        # Uniform slope -> uniform brightness, so the lowest column matches the low-elevation
        # stop's hue ratios (green-dominant) rather than the high stop's (near-white).
        low_pixel = elevation_shading(ramp())[0, 0].astype(np.int16)
        assert low_pixel[1] > low_pixel[0]

    def test_higher_elevation_is_brighter_toward_white(self):
        shaded = elevation_shading(ramp()).astype(np.int16)
        assert shaded[0, -1].sum() > shaded[0, 0].sum()


class TestWaterOverlay:
    def test_dry_terrain_leaves_base_unchanged(self):
        base = elevation_shading(flat())
        assert np.array_equal(apply_water_overlay(base, np.zeros((6, 6))), base)

    def test_deep_water_tends_toward_water_color(self):
        base = np.zeros((6, 6, 3), dtype=np.uint8)
        overlaid = apply_water_overlay(base, np.full((6, 6), 10.0))
        # Deep water saturates toward blue: the blue channel should dominate red and green.
        assert overlaid[0, 0, 2] > overlaid[0, 0, 0]
        assert overlaid[0, 0, 2] > overlaid[0, 0, 1]

    def test_shape_and_dtype_preserved(self):
        base = elevation_shading(flat(rows=4, cols=4))
        overlaid = apply_water_overlay(base, np.zeros((4, 4)))
        assert overlaid.shape == base.shape
        assert overlaid.dtype == np.uint8


class TestApplyFieldOverlay:
    GREEN = np.array([0.0, 255.0, 0.0])

    def test_empty_field_leaves_base_unchanged(self):
        base = elevation_shading(flat())
        field = np.zeros(base.shape[:2])
        assert np.array_equal(apply_field_overlay(base, field, 10.0, self.GREEN), base)

    def test_field_at_the_reference_saturates_toward_the_color(self):
        base = np.zeros((2, 2, 3), dtype=np.uint8)
        overlaid = apply_field_overlay(base, np.full((2, 2), 10.0), 10.0, self.GREEN)
        assert overlaid[0, 0, 1] > overlaid[0, 0, 0]
        assert overlaid[0, 0, 1] > overlaid[0, 0, 2]

    def test_more_of_the_field_means_more_tint(self):
        base = np.zeros((1, 3, 3), dtype=np.uint8)
        overlaid = apply_field_overlay(base, np.array([[0.0, 5.0, 10.0]]), 10.0, self.GREEN)
        greens = overlaid[0, :, 1].astype(np.int16)
        assert greens[0] < greens[1] < greens[2]

    def test_above_the_reference_clips_rather_than_tinting_further(self):
        """Nutrients are conserved, so a cell can concentrate well past an even share (#18). That
        must read as 'full' rather than overflowing the blend into a different colour."""
        base = np.zeros((1, 2, 3), dtype=np.uint8)
        overlaid = apply_field_overlay(base, np.array([[10.0, 40.0]]), 10.0, self.GREEN)
        assert np.array_equal(overlaid[0, 0], overlaid[0, 1])

    def test_reference_is_fixed_rather_than_read_off_the_field(self):
        """The regression test this overlay exists for.

        An implementation that normalised against the field's *current* range would render a
        starving world and a lush one identically, because the ramp would rescale under the
        viewer — which is precisely the slow decline §3.3 says the instrument must reveal.
        """
        base = np.zeros((2, 2, 3), dtype=np.uint8)
        lush = apply_field_overlay(base, np.full((2, 2), 10.0), 10.0, self.GREEN)
        starving = apply_field_overlay(base, np.full((2, 2), 1.0), 10.0, self.GREEN)
        assert not np.array_equal(lush, starving)
        assert starving[0, 0, 1] < lush[0, 0, 1]

    def test_shape_and_dtype_preserved(self):
        base = elevation_shading(flat(rows=4, cols=5))
        overlaid = apply_field_overlay(base, np.ones((4, 5)), 2.0, self.GREEN)
        assert overlaid.shape == (4, 5, 3)
        assert overlaid.dtype == np.uint8

    @pytest.mark.parametrize("reference", [0.0, -1.0])
    def test_non_positive_reference_raises(self, reference):
        base = np.zeros((2, 2, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="reference must be positive"):
            apply_field_overlay(base, np.ones((2, 2)), reference, self.GREEN)

    def test_field_shape_must_match_the_base(self):
        base = np.zeros((2, 2, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="field and base_rgb must share a grid shape"):
            apply_field_overlay(base, np.ones((3, 3)), 1.0, self.GREEN)


@pytest.fixture(scope="module")
def session():
    return api.create("demo", seed=1, founders=12)


class TestFieldOverlay:
    def test_every_layer_the_world_offers_has_a_tint(self):
        """The boundary decides which fields exist and this module only colours them. A layer
        offered without a tint would raise mid-frame, on the keypress that selects it."""
        assert set(render._LAYER_COLORS) == set(api.FIELD_LAYERS)
        assert FIELD_LAYERS == api.FIELD_LAYERS

    def test_every_offered_layer_renders(self, session):
        base = elevation_shading(session.terrain())
        for layer in FIELD_LAYERS:
            overlaid = field_overlay(base, session.field(layer))
            assert overlaid.shape == base.shape
            assert overlaid.dtype == np.uint8

    def test_grazed_ground_reads_darker_than_ungrazed(self):
        """What the overlay is for: a stripped patch must stand out from the crop around it."""
        session = api.create("demo", seed=1, founders=12)
        for _ in range(60):
            session._world.plants.grow()
        base = np.zeros(session.describe().grid_shape + (3,), dtype=np.uint8)
        before = field_overlay(base, session.field("biomass"))

        session._world.plants.biomass[2, 2] = 0.0
        after = field_overlay(base, session.field("biomass"))

        assert after[2, 2].sum() < before[2, 2].sum()
        assert np.array_equal(after[0, 0], before[0, 0])

    def test_a_kill_shows_on_the_carrion_layer(self):
        """The whole point of adding the layer: predation was running and invisible."""
        session = api.create("demo", seed=1, founders=12)
        base = np.zeros(session.describe().grid_shape + (3,), dtype=np.uint8)
        before = field_overlay(base, session.field("carrion"))

        session._world.carrion.mass[3, 3] = session._world.config.predation.strike_power
        after = field_overlay(base, session.field("carrion"))

        assert after[3, 3].sum() > before[3, 3].sum()
        assert np.array_equal(after[0, 0], before[0, 0])

    def test_the_tint_follows_the_layer_named_in_the_view(self):
        base = np.zeros((1, 1, 3), dtype=np.uint8)
        values = Column.of(np.ones((1, 1)))

        biomass = field_overlay(base, FieldView("biomass", values, 1.0))
        carrion = field_overlay(base, FieldView("carrion", values, 1.0))

        assert not np.array_equal(biomass, carrion)


class TestConditionColors:
    """What an animal's colour means. With one species, colouring by species is a field of
    identical dots, and every mechanic this view exists to show is a *condition* (#39)."""

    def test_the_modes_are_species_then_whatever_the_world_measures(self):
        assert CONDITION_MODES == ("species", *api.CONDITIONS)

    def test_one_colour_per_animal(self):
        colors = condition_colors(np.array([0.0, 0.5, 1.0]))
        assert colors.shape == (3, 3)
        assert colors.dtype == np.uint8

    def test_a_larger_deficit_is_darker(self):
        """Deficits darken toward death. Coloured the other way up — by the reserve — the animal
        in trouble would be the faintest mark on screen, which is backwards for an instrument
        whose job is spotting trouble."""
        colors = condition_colors(np.array([0.0, 1.0]))
        assert colors[0].sum() > colors[1].sum()

    def test_every_colour_drawn_is_a_validated_ramp_step(self):
        """Quantised rather than interpolated, so nothing reaches the screen that the ordinal
        checks never saw — and so a thousand dots read as bands rather than as mush."""
        colors = condition_colors(np.linspace(0.0, 1.0, 50))
        for colour in np.unique(colors, axis=0):
            assert any(np.array_equal(colour, step) for step in CONDITION_RAMP)

    def test_a_deficit_beyond_the_ends_still_lands_on_the_ramp(self):
        """Whatever arrives, indexing off the end of the ramp is this module's to prevent."""
        colors = condition_colors(np.array([-3.0, 7.0]))
        assert np.array_equal(colors[0], CONDITION_RAMP[0])
        assert np.array_equal(colors[1], CONDITION_RAMP[-1])

    def test_a_real_hungry_animal_draws_darker_than_a_fed_one(self):
        """The seam: the world's deficit, through this module's ramp."""
        session = api.create("demo", seed=1, founders=12)
        world = session._world
        first, second = (int(i) for i in session.frame().ids.to_numpy()[:2])
        row_ids = world.store.row_ids()
        world.store.energy[int(np.flatnonzero(row_ids == first)[0])] = (
            world.config.hunger.satiation_energy
        )
        world.store.energy[int(np.flatnonzero(row_ids == second)[0])] = 0.0

        colors = condition_colors(session.condition("hunger").values.to_numpy())

        assert colors[0].sum() > colors[1].sum()


class TestSpeciesColors:
    def test_unset_species_is_gray(self):
        colors = species_colors(np.array([-1, -1]))
        assert np.array_equal(colors[0], [128, 128, 128])
        assert np.array_equal(colors[1], [128, 128, 128])

    def test_same_id_is_deterministic(self):
        assert np.array_equal(species_colors(np.array([7])), species_colors(np.array([7])))

    def test_different_ids_get_different_colors(self):
        colors = species_colors(np.array([0, 1, 2]))
        assert not np.array_equal(colors[0], colors[1])
        assert not np.array_equal(colors[1], colors[2])

    def test_shape_and_dtype(self):
        colors = species_colors(np.array([0, 1, -1]))
        assert colors.shape == (3, 3)
        assert colors.dtype == np.uint8


class TestLivePositions:
    def test_alpha_zero_returns_previous(self):
        x, y, z = live_positions(frame([0, 1], [0.0, 0.0]), frame([0, 1], [10.0, 10.0]), 0.0)
        for axis in (x, y, z):
            assert np.allclose(axis, [0.0, 0.0])

    def test_alpha_one_returns_current(self):
        x, y, z = live_positions(frame([0, 1], [0.0, 0.0]), frame([0, 1], [10.0, 10.0]), 1.0)
        for axis in (x, y, z):
            assert np.allclose(axis, [10.0, 10.0])

    def test_alpha_half_returns_midpoint(self):
        x, y, z = live_positions(frame([7], [0.0]), frame([7], [10.0]), 0.5)
        for axis in (x, y, z):
            assert np.allclose(axis, [5.0])

    def test_out_of_range_alpha_raises(self):
        previous, current = frame([0], [0.0]), frame([0], [1.0])
        with pytest.raises(ValueError):
            live_positions(previous, current, 1.5)
        with pytest.raises(ValueError):
            live_positions(previous, current, -0.1)

    def test_an_animal_that_died_is_not_drawn(self):
        """The ghost of #119. A frame holds only the living, so the dead one is simply absent."""
        x, _y, _z = live_positions(
            frame([10, 11, 12], [1.0, 2.0, 3.0]), frame([10, 12], [1.0, 3.0]), 1.0
        )
        assert x.tolist() == [1.0, 3.0]

    def test_a_newborn_is_drawn_at_its_current_position_not_blended(self):
        """It has no previous position. Blended at 0.5 from anywhere, it would render halfway
        from a place it never was."""
        x, _y, _z = live_positions(frame([], []), frame([4], [100.0]), 0.5)
        assert x.tolist() == [100.0]

    def test_a_newborn_does_not_streak_from_where_another_animal_died(self):
        """The case a row comparison existed for: a newborn taking a dead animal's row. By id the
        two are unrelated, so nothing blends between them."""
        x, _y, _z = live_positions(frame([10], [900.0]), frame([11], [10.0]), 0.5)
        assert x.tolist() == [10.0]

    def test_a_survivor_blends_while_others_do_not(self):
        previous = frame([10, 20, 40], [0.0, 900.0, 50.0])
        current = frame([10, 21, 30], [100.0, 10.0, 20.0])

        x, _y, _z = live_positions(previous, current, 0.5)

        # The survivor blends to the midpoint; the two newcomers snap to where they are.
        assert x.tolist() == [50.0, 10.0, 20.0]

    def test_a_survivor_blends_from_itself_whatever_order_the_frames_list_it_in(self):
        """Frames come in whatever order the world holds animals in, and that order changes as
        rows are freed and reused. Matching has to be by id, never by position in the list."""
        previous = frame([3, 1, 2], [30.0, 10.0, 20.0])
        current = frame([1, 2, 3], [11.0, 22.0, 33.0])

        x, _y, _z = live_positions(previous, current, 0.5)

        assert x.tolist() == [10.5, 21.0, 31.5]

    def test_the_result_lines_up_with_the_current_frames_other_columns(self):
        """What lets species and condition colour be read straight off the frame."""
        current = frame([1, 3], [1.0, 3.0])
        x, _y, _z = live_positions(frame([1, 2, 3], [0.0, 0.0, 0.0]), current, 1.0)
        assert len(x) == current.ids.shape[0]


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


class TestDragRectangle:
    """A drag is normalised, so the player never has to know which corner the code sees first."""

    def test_a_downward_right_drag_is_the_box_it_looks_like(self):
        assert drag_rectangle((10, 20), (110, 220)) == (10, 20, 100, 200)

    def test_dragging_the_other_way_gives_the_same_box(self):
        assert drag_rectangle((110, 220), (10, 20)) == drag_rectangle((10, 20), (110, 220))

    def test_a_click_without_a_drag_has_no_area(self):
        assert drag_rectangle((40, 40), (40, 40)) == (40, 40, 0, 0)


class TestBarrierSegments:
    """A barrier lives on a cell *edge* (#27), so it draws as a line between two cells. Drawn as a
    filled cell instead it would sit half a cell off and a pen would look one cell smaller than the
    one the animals are actually held by."""

    def barriers(self):
        return Barriers(Terrain(np.full((6, 6), 50.0, dtype=np.float32), cell_size=1.0))

    def view(self, barriers):
        return BarrierView(
            blocked_north=Column.of(barriers.blocked_north),
            blocked_west=Column.of(barriers.blocked_west),
        )

    def segments(self, barriers):
        # A 10-world-unit extent drawn at 100 px, one world unit per cell.
        return barrier_segments(self.view(barriers), 1.0, 10.0, 10.0, 100, 100)

    def test_an_unfenced_world_draws_nothing(self):
        assert self.segments(self.barriers()) == []

    def test_every_blocked_edge_becomes_one_segment(self):
        barriers = self.barriers()
        blocked = barriers.enclose(1.0, 1.0, 4.0, 4.0)

        assert len(self.segments(barriers)) == blocked

    def test_a_north_edge_draws_horizontally_and_a_west_edge_vertically(self):
        barriers = self.barriers()
        barriers.blocked_north[3, 4] = True
        barriers.blocked_west[4, 2] = True

        horizontal, vertical = self.segments(barriers)

        assert horizontal[1] == horizontal[3] and horizontal[0] != horizontal[2]
        assert vertical[0] == vertical[2] and vertical[1] != vertical[3]

    def test_an_edge_draws_on_the_cell_boundary_not_through_its_middle(self):
        """The edge above row 3 is at world y = 3, which is 30 pixels into a 10-unit world drawn
        at 100 pixels — not 35, which is where the cell's centre is (#162)."""
        barriers = self.barriers()
        barriers.blocked_north[3, 4] = True

        (segment,) = self.segments(barriers)

        assert segment[1] == 30
