import numpy as np
import pytest

from clients.viewer.render import (
    PLANT_LAYERS,
    apply_field_overlay,
    apply_water_overlay,
    elevation_shading,
    live_positions,
    plant_overlay,
    plant_overlay_references,
    species_colors,
    world_to_screen,
)
from core.ecology.plants import Plants, PlantsConfig
from core.world.climate import Climate, ClimateConfig
from core.world.diffusion import DiffusionConfig
from core.world.terrain import Terrain
from core.world.water import Water


def flat_terrain(value=50.0, rows=6, cols=6, cell_size=1.0):
    return Terrain(np.full((rows, cols), value, dtype=np.float32), cell_size=cell_size)


PLANTS_CONFIG = PlantsConfig(
    solar_constant=10.0,
    latitude_tilt=0.02,
    min_growth_temperature=0.0,
    optimal_growth_temperature=25.0,
    max_growth_temperature=45.0,
    nutrient_per_biomass=0.1,
    initial_soil_nutrients=100.0,
    senescence_rate=0.05,
    saturation_accumulation=50.0,
    max_rooting_depth=0.5,
    forage_diffusion=DiffusionConfig(range=4.0, climb_penalty=0.5),
)


def plants(terrain=None, config=PLANTS_CONFIG):
    terrain = flat_terrain() if terrain is None else terrain
    # Equator through the middle of the grid, so every cell sits in the growth band and the field
    # is productive everywhere — the overlay is what is under test, not the climate.
    climate = Climate(terrain, ClimateConfig(equator_y=terrain.world_height / 2.0))
    return Plants(terrain, climate, Water.generate(terrain), config)


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


class TestApplyFieldOverlay:
    GREEN = np.array([0.0, 255.0, 0.0])

    def test_empty_field_leaves_base_unchanged(self):
        base = elevation_shading(flat_terrain())
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
        base = elevation_shading(flat_terrain(rows=4, cols=5))
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


class TestPlantOverlayReferences:
    def test_biomass_reference_is_the_light_limited_steady_state(self):
        field = plants()
        expected = field.potential_growth.max() / PLANTS_CONFIG.senescence_rate
        assert plant_overlay_references(field)["biomass"] == pytest.approx(expected)

    def test_light_limited_steady_state_is_where_unlimited_biomass_settles(self):
        """Justifies the number above rather than taking it on trust: with soil so deep that
        growth is never nutrient-limited, biomass converges on exactly this reference, so a cell
        reading 1.0 is one as green as its light and water allow."""
        deep_soil = PlantsConfig(
            solar_constant=10.0,
            latitude_tilt=0.02,
            min_growth_temperature=0.0,
            optimal_growth_temperature=25.0,
            max_growth_temperature=45.0,
            nutrient_per_biomass=0.1,
            initial_soil_nutrients=1.0e9,
            senescence_rate=0.05,
            saturation_accumulation=50.0,
            max_rooting_depth=0.5,
            forage_diffusion=DiffusionConfig(range=4.0, climb_penalty=0.5),
        )
        field = plants(config=deep_soil)
        reference = plant_overlay_references(field)["biomass"]
        for _ in range(400):
            field.grow()
        assert field.biomass.max() == pytest.approx(reference, rel=1e-3)

    def test_soil_reference_is_the_starting_per_cell_pool(self):
        references = plant_overlay_references(plants())
        assert references["soil_nutrients"] == pytest.approx(PLANTS_CONFIG.initial_soil_nutrients)

    def test_moisture_reference_is_one_because_moisture_is_already_a_fraction(self):
        assert plant_overlay_references(plants())["moisture"] == pytest.approx(1.0)

    def test_references_do_not_move_as_the_field_changes(self):
        """The anti-trap property at the level the viewer actually uses it: references are computed
        once and a hundred ticks of growth and senescence must not shift the ramp underneath."""
        field = plants()
        before = plant_overlay_references(field)
        for _ in range(100):
            field.grow()
        assert plant_overlay_references(field) == before

    def test_every_declared_layer_has_a_reference(self):
        references = plant_overlay_references(plants())
        assert set(references) == set(PLANT_LAYERS)

    def test_a_world_with_no_light_anywhere_raises_rather_than_scaling_against_zero(self):
        frozen = PlantsConfig(
            solar_constant=10.0,
            latitude_tilt=0.02,
            # Every cell sits far below the minimum growth temperature, so potential growth is
            # zero everywhere and there is no scale to draw a biomass ramp against.
            min_growth_temperature=500.0,
            optimal_growth_temperature=600.0,
            max_growth_temperature=700.0,
            nutrient_per_biomass=0.1,
            initial_soil_nutrients=100.0,
            senescence_rate=0.05,
            saturation_accumulation=50.0,
            max_rooting_depth=0.5,
            forage_diffusion=DiffusionConfig(range=4.0, climb_penalty=0.5),
        )
        with pytest.raises(ValueError, match="no cell can grow anything"):
            plant_overlay_references(plants(config=frozen))

    def test_a_world_with_no_soil_nutrients_raises(self):
        """`PlantsConfig` permits zero, so this is reachable rather than defensive. It is caught
        here, where the message can name the config field, instead of surfacing later as a
        complaint about a reference the caller never chose."""
        barren = PlantsConfig(
            solar_constant=10.0,
            latitude_tilt=0.02,
            min_growth_temperature=0.0,
            optimal_growth_temperature=25.0,
            max_growth_temperature=45.0,
            nutrient_per_biomass=0.1,
            initial_soil_nutrients=0.0,
            senescence_rate=0.05,
            saturation_accumulation=50.0,
            max_rooting_depth=0.5,
            forage_diffusion=DiffusionConfig(range=4.0, climb_penalty=0.5),
        )
        with pytest.raises(ValueError, match="initial_soil_nutrients is zero"):
            plant_overlay_references(plants(config=barren))


class TestPlantOverlay:
    def test_every_declared_layer_renders(self):
        field = plants()
        references = plant_overlay_references(field)
        base = elevation_shading(flat_terrain())
        for layer in PLANT_LAYERS:
            overlaid = plant_overlay(base, field, layer, references)
            assert overlaid.shape == base.shape
            assert overlaid.dtype == np.uint8

    def test_grazed_ground_reads_darker_than_ungrazed(self):
        """What the issue is for: a stripped patch must be visibly distinguishable from the
        standing crop around it."""
        field = plants()
        for _ in range(60):
            field.grow()
        base = np.zeros(field.biomass.shape + (3,), dtype=np.uint8)
        references = plant_overlay_references(field)
        before = plant_overlay(base, field, "biomass", references)

        field.biomass[2, 2] = 0.0
        after = plant_overlay(base, field, "biomass", references)
        assert after[2, 2].sum() < before[2, 2].sum()
        assert np.array_equal(after[0, 0], before[0, 0])

    def test_unknown_layer_raises(self):
        field = plants()
        references = plant_overlay_references(field)
        base = np.zeros(field.biomass.shape + (3,), dtype=np.uint8)
        with pytest.raises(KeyError, match="carrion"):
            plant_overlay(base, field, "carrion", references)


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


class TestLivePositions:
    def _positions(self, values):
        arr = np.array(values, dtype=np.float32)
        return (arr, arr.copy(), arr.copy())

    def _ids(self, values):
        return np.array(values, dtype=np.int64)

    def test_alpha_zero_returns_previous(self):
        previous = self._positions([0.0, 0.0])
        current = self._positions([10.0, 10.0])
        ids = self._ids([0, 1])
        x, y, z, _drawn = live_positions(previous, ids, current, ids, 0.0)
        for axis in (x, y, z):
            assert np.allclose(axis, [0.0, 0.0])

    def test_alpha_one_returns_current(self):
        previous = self._positions([0.0, 0.0])
        current = self._positions([10.0, 10.0])
        ids = self._ids([0, 1])
        x, y, z, _drawn = live_positions(previous, ids, current, ids, 1.0)
        for axis in (x, y, z):
            assert np.allclose(axis, [10.0, 10.0])

    def test_alpha_half_returns_midpoint(self):
        previous = self._positions([0.0])
        current = self._positions([10.0])
        ids = self._ids([7])
        x, y, z, _drawn = live_positions(previous, ids, current, ids, 0.5)
        for axis in (x, y, z):
            assert np.allclose(axis, [5.0])

    def test_out_of_range_alpha_raises(self):
        previous = self._positions([0.0])
        current = self._positions([1.0])
        ids = self._ids([0])
        with pytest.raises(ValueError):
            live_positions(previous, ids, current, ids, 1.5)
        with pytest.raises(ValueError):
            live_positions(previous, ids, current, ids, -0.1)

    def test_free_rows_are_not_drawn_at_all(self):
        """The ghost: a released row keeps its coordinates, so drawing capacity draws corpses."""
        previous = self._positions([1.0, 2.0, 3.0])
        current = self._positions([1.0, 2.0, 3.0])
        # Middle row released between the snapshots; its position column still reads 2.0.
        x, y, z, drawn = live_positions(
            previous, self._ids([10, 11, 12]), current, self._ids([10, -1, 12]), 1.0
        )

        assert drawn.tolist() == [True, False, True]
        for axis in (x, y, z):
            assert axis.tolist() == [1.0, 3.0]

    def test_a_row_never_occupied_is_not_drawn(self):
        previous = self._positions([0.0, 0.0])
        current = self._positions([5.0, 0.0])
        x, _y, _z, drawn = live_positions(
            previous, self._ids([-1, -1]), current, self._ids([3, -1]), 1.0
        )

        assert drawn.tolist() == [True, False]
        assert x.tolist() == [5.0]

    def test_a_newborn_is_drawn_at_its_current_position_not_blended(self):
        """A fresh row's previous coordinate is whatever the array was initialised to — here 0.

        Blended at alpha 0.5 it would render at 50, halfway from a place it never was.
        """
        previous = self._positions([0.0])
        current = self._positions([100.0])
        x, _y, _z, _drawn = live_positions(
            previous, self._ids([-1]), current, self._ids([4]), 0.5
        )

        assert x.tolist() == [100.0]

    def test_a_recycled_row_does_not_streak_from_its_predecessors_death_site(self):
        """Both ends occupied, different entities: the case an `alive` flag cannot detect.

        Row 0 held entity 10 at the far edge and now holds newborn 11 near the origin. Blending
        would draw a line between two unrelated animals.
        """
        previous = self._positions([900.0])
        current = self._positions([10.0])
        x, _y, _z, drawn = live_positions(
            previous, self._ids([10]), current, self._ids([11]), 0.5
        )

        assert drawn.tolist() == [True]
        assert x.tolist() == [10.0]

    def test_a_surviving_neighbour_still_blends_while_others_do_not(self):
        """One array, three different rules, so the mask arithmetic has to be per-row."""
        previous = self._positions([0.0, 900.0, 0.0, 50.0])
        current = self._positions([100.0, 10.0, 20.0, 50.0])
        x, _y, _z, drawn = live_positions(
            previous,
            self._ids([10, 20, -1, 40]),  # survivor, recycled, fresh, released
            current,
            self._ids([10, 21, 30, -1]),
            0.5,
        )

        assert drawn.tolist() == [True, True, True, False]
        # survivor blends to the midpoint; the other two snap to current; the dead one is gone.
        assert x.tolist() == [50.0, 10.0, 20.0]

    def test_drawn_mask_selects_other_columns_onto_the_same_rows(self):
        """What `drawn` is for: species colour must line up with the positions drawn."""
        previous = self._positions([0.0, 0.0, 0.0])
        current = self._positions([1.0, 2.0, 3.0])
        species_id = np.array([5, 6, 7], dtype=np.int32)

        x, _y, _z, drawn = live_positions(
            previous, self._ids([1, 2, 3]), current, self._ids([1, -1, 3]), 1.0
        )

        assert species_id[drawn].tolist() == [5, 7]
        assert len(x) == len(species_id[drawn])


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
