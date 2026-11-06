import numpy as np
import pytest

from core.world.climate import Climate, ClimateConfig, SeasonsConfig, zone_at
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
        max_elevation=32.0,  # a tenth of the 320-unit extent (#112)
    )
    params.update(overrides)
    return Terrain.generate(TerrainConfig(**params))


def make_climate_config(terrain, **overrides):
    params = dict(equator_y=terrain.world_height / 2, equator_temperature=30.0)
    params.update(overrides)
    return ClimateConfig(**params)


class TestClimateConfig:
    def test_altitude_and_latitude_gradients_default_as_a_tuned_pair(self):
        """The two default gradients are one tuning decision, not two constants (#112).

        `lapse_rate` was Earth's tropospheric value, denominated per physical length unit; read
        against elevation in world units it cooled a peak by hundredths of a degree, so altitude
        stopped driving climate and §2.6's "climate zones are consequences of terrain" became
        decoration. It is re-derived from the world's own geometry: a peak should be about as cold
        as a pole, so ``lapse_rate * relief ~= latitude_gradient * half_extent``, and with relief
        at a tenth of extent (`TerrainConfig.max_elevation`) that is a factor of five.

        Pinned because the factor is what makes terrain matter. Moving either default alone
        silently halves or doubles how much relief contributes to climate relative to latitude,
        which is precisely §2.1's warning about constants that must be tuned as a table drifting
        apart instead.
        """
        defaults = ClimateConfig(equator_y=0.0, equator_temperature=30.0)

        assert defaults.lapse_rate == pytest.approx(5.0 * defaults.latitude_gradient)

    def test_the_equator_temperature_has_no_default(self):
        """An absolute temperature means something only against the others it is read beside —
        the neutral temperatures and the growth band, all required — so a default on this one side
        could drift from them unnoticed. Same rule #112 applied to absolute lengths (#238)."""
        with pytest.raises(TypeError, match="equator_temperature"):
            ClimateConfig(equator_y=0.0)  # type: ignore[call-arg]

    def test_rejects_negative_latitude_gradient(self):
        with pytest.raises(ValueError):
            ClimateConfig(equator_y=0.0, equator_temperature=30.0, latitude_gradient=-0.1)

    def test_rejects_negative_lapse_rate(self):
        with pytest.raises(ValueError):
            ClimateConfig(equator_y=0.0, equator_temperature=30.0, lapse_rate=-0.1)


class TestTemperatureField:
    def test_temperature_falls_with_altitude(self):
        # Flat latitude (all rows at the equator) isolates the altitude effect.
        heights = np.array([[0.0, 1.0, 2.0]] * 3, dtype=np.float32)
        terrain = Terrain(heights, cell_size=1.0)
        config = ClimateConfig(
            equator_y=terrain.world_height / 2, equator_temperature=30.0, latitude_gradient=0.0
        )
        climate = Climate(terrain, config)

        row = climate.temperature[1]
        assert row[0] > row[1] > row[2]

    def test_temperature_varies_with_latitude(self):
        # Flat terrain isolates the latitude effect.
        heights = np.zeros((5, 3), dtype=np.float32)
        terrain = Terrain(heights, cell_size=10.0)
        config = ClimateConfig(equator_y=terrain.world_height / 2, equator_temperature=30.0)
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
        config = ClimateConfig(equator_y=0.0, equator_temperature=30.0, latitude_gradient=0.0)
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
        terrain = make_terrain(height=65, max_elevation=64.0)
        config = make_climate_config(terrain, equator_temperature=35.0, latitude_gradient=1.0)
        climate = Climate(terrain, config)

        assert len(set(climate.zone_labels().ravel().tolist())) > 1

    def test_zone_boundaries_are_ordered_by_temperature(self):
        temperature = np.array([-20.0, -5.0, 10.0, 25.0], dtype=np.float32)
        labels = zone_at(temperature)
        assert list(labels) == ["polar", "tundra", "temperate", "tropical"]


def flat_terrain(rows=21, cols=5, cell_size=1.0):
    return Terrain(np.zeros((rows, cols), dtype=np.float32), cell_size=cell_size)


def seasons(**overrides):
    params = dict(
        year_length=40,
        amplitude_fraction=0.5,
        weather_amplitude=0.0,
        weather_scale=4.0,
        weather_timescale=5,
    )
    params.update(overrides)
    return SeasonsConfig(**params)


def seasonal_climate(terrain, seed=3, latitude_gradient=0.5, **season_overrides):
    config = ClimateConfig(
        equator_y=terrain.world_height / 2,
        equator_temperature=30.0,
        latitude_gradient=latitude_gradient,
        seasons=seasons(**season_overrides),
    )
    return Climate(terrain, config, seed=seed)


def static_part(terrain, latitude_gradient=0.5):
    """The same climate with no seasons and no weather: what the time-varying terms are added to."""
    config = ClimateConfig(
        equator_y=terrain.world_height / 2,
        equator_temperature=30.0,
        latitude_gradient=latitude_gradient,
    )
    return Climate(terrain, config).temperature


class TestSeasonsConfig:
    @pytest.mark.parametrize(
        "changes",
        [
            dict(year_length=1),
            dict(amplitude_fraction=-0.1),
            dict(weather_amplitude=-1.0),
            dict(weather_scale=0.0),
            dict(weather_timescale=0),
        ],
    )
    def test_rejects_out_of_range_values(self, changes):
        with pytest.raises(ValueError):
            seasons(**changes)

    def test_an_amplitude_that_would_invert_the_latitude_gradient_is_rejected(self):
        """#159's measured trap: a seasonal swing that grows with latitude, at or past the
        equator-to-pole gradient itself, makes summer poles as hot as the tropics or hotter.
        Amplitude is a fraction *of the latitude gradient*, so 1.0 is exactly that crossover."""
        with pytest.raises(ValueError, match="invert"):
            seasons(amplitude_fraction=1.0)

    def test_seasons_need_the_world_seed(self):
        terrain = flat_terrain()
        config = ClimateConfig(
            equator_y=terrain.world_height / 2, equator_temperature=30.0, seasons=seasons()
        )
        with pytest.raises(ValueError, match="seed"):
            Climate(terrain, config)


class TestSeasonalCycle:
    def test_summer_never_inverts_the_latitude_gradient(self):
        """The pinned set (§2.1): amplitude moves with `latitude_gradient`, so no tuning of either
        can make a pole the warmest ground on the map. Checked just below the crossover, over a
        whole year, with no weather noise to blur it."""
        terrain = flat_terrain()
        climate = seasonal_climate(terrain, amplitude_fraction=0.99)
        equator_row = terrain.heights.shape[0] // 2
        for tick in range(climate.config.seasons.year_length):
            column = climate.field_at(tick)[:, 0]
            assert np.all(column[equator_row] >= column)
            assert column[equator_row] > column[0]
            assert column[equator_row] > column[-1]

    def test_the_swing_is_a_fraction_of_the_latitude_gradient(self):
        """Doubling the gradient doubles the season, because amplitude is denominated in it."""
        terrain = flat_terrain()
        quarter_year = 10  # the cycle peaks a quarter of the way through a 40-tick year
        swings = []
        for gradient in (0.5, 1.0):
            climate = seasonal_climate(terrain, latitude_gradient=gradient)
            swings.append(climate.field_at(quarter_year)[0, 0] - climate.field_at(0)[0, 0])
        assert swings[1] == pytest.approx(2.0 * swings[0], rel=1e-5)
        assert swings[0] > 0.0

    def test_the_equator_has_no_season(self):
        terrain = flat_terrain()
        climate = seasonal_climate(terrain)
        equator_row = terrain.heights.shape[0] // 2
        year = [climate.field_at(t)[equator_row, 0] for t in range(40)]
        assert np.ptp(year) == pytest.approx(0.0, abs=1e-5)

    def test_winter_is_colder_than_summer_away_from_the_equator(self):
        terrain = flat_terrain()
        climate = seasonal_climate(terrain)
        summer, winter = climate.field_at(10), climate.field_at(30)
        assert summer[0, 0] > winter[0, 0]
        assert summer[-1, 0] > winter[-1, 0]

    def test_a_year_later_the_season_is_the_same(self):
        terrain = flat_terrain()
        climate = seasonal_climate(terrain)
        np.testing.assert_allclose(climate.field_at(7), climate.field_at(47), atol=1e-5)


class TestWeather:
    def test_weather_varies_in_space_and_time(self):
        terrain = flat_terrain(rows=33, cols=33)
        climate = seasonal_climate(terrain, amplitude_fraction=0.0, weather_amplitude=2.0)
        anomaly = climate.field_at(100) - static_part(terrain)
        assert np.std(anomaly) > 0.1
        assert not np.allclose(climate.field_at(100), climate.field_at(160))

    def test_weather_changes_smoothly_from_tick_to_tick(self):
        """Weather persists for about `weather_timescale` ticks rather than flickering per tick."""
        terrain = flat_terrain(rows=33, cols=33)
        climate = seasonal_climate(
            terrain, amplitude_fraction=0.0, weather_amplitude=2.0, weather_timescale=20
        )
        one_tick = np.abs(climate.field_at(101) - climate.field_at(100)).mean()
        many_ticks = np.abs(climate.field_at(160) - climate.field_at(100)).mean()
        assert one_tick < 0.25 * many_ticks

    def test_weather_has_its_stated_size(self):
        terrain = flat_terrain(rows=33, cols=33)
        climate = seasonal_climate(
            terrain, amplitude_fraction=0.0, weather_amplitude=2.0, weather_timescale=3
        )
        base = static_part(terrain)
        anomaly = np.stack([climate.field_at(t) - base for t in range(0, 3000, 3)])
        # Exactly 2.0 at the lattice knots; interpolation between knots only shrinks the spread.
        assert 1.2 < anomaly.std() < 2.2


class TestTheFieldIsAFunctionNotAState:
    """§2.4 and §3.2: weather is evaluated at whatever tick the world reaches, never integrated,
    so a world that skipped ticks, batched them or was reloaded from a snapshot has the same sky."""

    def test_the_same_tick_and_seed_give_the_same_field_in_any_order(self):
        terrain = flat_terrain(rows=33, cols=33)
        stepped = seasonal_climate(terrain, weather_amplitude=2.0)
        jumped = seasonal_climate(terrain, weather_amplitude=2.0)
        for tick in range(0, 501):
            stepped.advance_to(tick)
        jumped.advance_to(500)
        np.testing.assert_array_equal(stepped.temperature, jumped.temperature)
        fresh = seasonal_climate(terrain, weather_amplitude=2.0)
        np.testing.assert_array_equal(jumped.temperature, fresh.field_at(500))

    def test_a_different_seed_is_different_weather(self):
        terrain = flat_terrain(rows=33, cols=33)
        a = seasonal_climate(terrain, seed=1, weather_amplitude=2.0).field_at(77)
        b = seasonal_climate(terrain, seed=2, weather_amplitude=2.0).field_at(77)
        assert not np.allclose(a, b)

    def test_advancing_writes_in_place(self):
        """Readers hold `climate.temperature`; a replaced array would leave them on a stale sky."""
        terrain = flat_terrain()
        climate = seasonal_climate(terrain)
        held = climate.temperature
        climate.advance_to(10)
        assert held is climate.temperature
        np.testing.assert_array_equal(held, climate.field_at(10))

    def test_a_climate_without_seasons_never_changes(self):
        terrain = make_terrain()
        climate = Climate(terrain, make_climate_config(terrain))
        before = climate.temperature.copy()
        climate.advance_to(12345)
        np.testing.assert_array_equal(climate.temperature, before)
