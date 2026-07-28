"""Plants: sunlight as the only energy input, drawn against a finite soil-nutrient pool (#18).

Most of this contract is checkable in advance (CLAUDE.md §8.1): the growth field is a pure
function of terrain, climate and water, and the nutrient ledger is an arithmetic identity. What is
*not* test-first is the tuning — whether these coefficients make a legible ecology — so
`TestClimateZoneProductivity` locks in the shape of the tuned result (which zones out-produce
which) without asserting any particular number.
"""

import numpy as np
import pytest

from core.ecology.plants import Plants, PlantsConfig
from core.world.climate import Climate, ClimateConfig
from core.world.terrain import Terrain
from core.world.water import Water


CONFIG = PlantsConfig(
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
)

_CONFIG_FIELDS = (
    "solar_constant",
    "latitude_tilt",
    "min_growth_temperature",
    "optimal_growth_temperature",
    "max_growth_temperature",
    "nutrient_per_biomass",
    "initial_soil_nutrients",
    "senescence_rate",
    "saturation_accumulation",
    "max_rooting_depth",
)


def replace(config, **changes):
    """A copy of `config` with `changes` applied — PlantsConfig is frozen."""
    fields = {name: getattr(config, name) for name in _CONFIG_FIELDS}
    fields.update(changes)
    return PlantsConfig(**fields)


def make_plants(
    heights=None,
    config=CONFIG,
    equator_y=0.0,
    equator_temperature=25.0,
    latitude_gradient=0.0,
    cell_size=1.0,
):
    """A plant field over flat, uniformly warm terrain unless a test asks for otherwise."""
    if heights is None:
        heights = np.zeros((11, 11), dtype=np.float32)
    terrain = Terrain(heights, cell_size=cell_size)
    climate = Climate(
        terrain,
        ClimateConfig(
            equator_y=equator_y,
            equator_temperature=equator_temperature,
            latitude_gradient=latitude_gradient,
        ),
    )
    water = Water.generate(terrain)
    return Plants(terrain, climate, water, config)


def settle(plants, ticks=600):
    """Advance far enough that standing biomass has reached its equilibrium."""
    for _ in range(ticks):
        plants.grow()


class TestConfigValidation:
    """Bad tuning fails at construction, not as a silently dead field (CLAUDE.md §8.7)."""

    @pytest.mark.parametrize(
        "changes, message",
        [
            ({"solar_constant": -1.0}, "solar_constant"),
            ({"latitude_tilt": -0.1}, "latitude_tilt"),
            ({"nutrient_per_biomass": 0.0}, "nutrient_per_biomass"),
            ({"initial_soil_nutrients": -1.0}, "initial_soil_nutrients"),
            ({"senescence_rate": 0.0}, "senescence_rate"),
            ({"senescence_rate": 1.5}, "senescence_rate"),
            ({"saturation_accumulation": 0.0}, "saturation_accumulation"),
            ({"max_rooting_depth": -1.0}, "max_rooting_depth"),
        ],
    )
    def test_rejects_out_of_range_values(self, changes, message):
        with pytest.raises(ValueError, match=message):
            replace(CONFIG, **changes)

    def test_rejects_unordered_growth_temperatures(self):
        with pytest.raises(ValueError, match="temperature"):
            replace(CONFIG, min_growth_temperature=30.0, optimal_growth_temperature=25.0)
        with pytest.raises(ValueError, match="temperature"):
            replace(CONFIG, optimal_growth_temperature=50.0, max_growth_temperature=45.0)


class TestSunlight:
    """Sunlight varies with latitude and terrain aspect — the only energy input (§2.5)."""

    def test_flat_ground_is_most_productive_at_the_equator(self):
        plants = make_plants(equator_y=0.0)
        # Row index is world y, so ascending rows are increasing distance from the equator line.
        assert np.all(np.diff(plants.potential_growth[:, 5]) < 0)

    def test_slope_facing_the_equator_outproduces_the_slope_facing_away(self):
        # An east-west ridge: heights rise with y, so every cell's surface normal faces -y —
        # toward an equator placed below the grid. Mirroring it in y turns it away from the sun
        # while leaving its elevation distribution, and therefore its temperatures, unchanged.
        rows, cols = 11, 11
        toward = np.tile(np.arange(rows, dtype=np.float32)[:, None] * 20.0, (1, cols))
        away = toward[::-1].copy()
        facing_equator = make_plants(heights=toward, equator_y=-50.0)
        facing_away = make_plants(heights=away, equator_y=-50.0)
        interior = (slice(1, -1), slice(1, -1))
        assert (
            facing_equator.potential_growth[interior].mean()
            > facing_away.potential_growth[interior].mean()
        )

    def test_ground_the_sun_cannot_reach_receives_nothing_rather_than_negative_light(self):
        # A latitude tilt steep enough to drive the sun below the horizon partway up the grid.
        # Beyond that point the incidence angle is clamped, so growth is a rounding artifact of
        # cos(pi/2) rather than exactly zero — hence the ratio rather than an equality.
        plants = make_plants(config=replace(CONFIG, latitude_tilt=0.5), equator_y=0.0)
        assert np.all(plants.potential_growth >= 0.0)
        assert plants.potential_growth[-1, 5] < plants.potential_growth[0, 5] * 1e-9


class TestTemperatureResponse:
    """Growth responds to the temperature field, so productivity varies by climate zone."""

    def test_peaks_at_the_optimal_temperature(self):
        at_optimum = make_plants(equator_temperature=CONFIG.optimal_growth_temperature)
        off_optimum = make_plants(equator_temperature=CONFIG.optimal_growth_temperature - 10.0)
        assert at_optimum.potential_growth.max() > off_optimum.potential_growth.max()

    @pytest.mark.parametrize("temperature", [-5.0, 50.0])
    def test_nothing_grows_outside_the_tolerated_range(self, temperature):
        plants = make_plants(equator_temperature=temperature)
        assert np.all(plants.potential_growth == 0.0)


class TestWaterResponse:
    def test_plants_drown_where_standing_water_exceeds_rooting_depth(self):
        # A single deep pit in otherwise flat ground pools well past max_rooting_depth.
        heights = np.zeros((11, 11), dtype=np.float32)
        heights[5, 5] = -10.0
        plants = make_plants(heights=heights)
        assert plants.potential_growth[5, 5] == 0.0
        assert plants.potential_growth[5, 3] > 0.0

    def test_wetter_ground_is_more_productive(self):
        # A valley running east-west: the floor collects the drainage of both slopes and drains
        # out of the world at the map edge, so it is the wettest ground that is not flooded.
        rows, cols = 11, 11
        distance_from_floor = np.abs(np.arange(rows, dtype=np.float32)[:, None] - 5.0)
        heights = np.tile(distance_from_floor * 5.0, (1, cols))
        plants = make_plants(heights=heights)
        assert plants.moisture[5, 5] > plants.moisture[0, 5]


class TestNutrientLimitedGrowth:
    def test_growth_consumes_soil_nutrients(self):
        plants = make_plants()
        before = plants.soil_nutrients.copy()
        plants.grow()
        assert np.all(plants.biomass > 0.0)
        assert np.all(plants.soil_nutrients < before)

    def test_exhausted_soil_grows_nothing(self):
        plants = make_plants()
        plants.soil_nutrients[:] = 0.0
        plants.grow()
        assert np.all(plants.biomass == 0.0)

    def test_growth_never_overdraws_the_soil_pool(self):
        # Potential growth far exceeds what one tick's nutrients could build, so the nutrient
        # limit — not the light limit — is what this tick is clamped to.
        plants = make_plants(config=replace(CONFIG, solar_constant=1e6))
        plants.grow()
        assert np.all(plants.soil_nutrients >= 0.0)

    def test_standing_biomass_settles_rather_than_growing_without_bound(self):
        plants = make_plants(config=replace(CONFIG, initial_soil_nutrients=1e9))
        settle(plants)
        settled = plants.biomass.copy()
        settle(plants, ticks=200)
        # Senescence returning a fixed fraction per tick is what makes an equilibrium exist at
        # all: standing biomass stops rising once that loss matches the light-limited gain.
        assert np.allclose(plants.biomass, settled, rtol=1e-6)

    def test_a_poorer_soil_supports_less_standing_biomass(self):
        rich = make_plants(config=replace(CONFIG, initial_soil_nutrients=5.0))
        poor = make_plants(config=replace(CONFIG, initial_soil_nutrients=2.0))
        settle(rich)
        settle(poor)
        assert poor.biomass.mean() < rich.biomass.mean()


class TestNutrientConservation:
    """CLAUDE.md §2.5's closed loop: nutrients cycle, they are never created or destroyed."""

    def test_conserved_across_growth_and_senescence(self):
        plants = make_plants()
        opening = plants.total_nutrients()
        settle(plants, ticks=200)
        assert plants.total_nutrients() == pytest.approx(opening, rel=1e-9)

    def test_grazed_nutrients_leave_the_field_but_stay_on_the_ledger(self):
        plants = make_plants()
        settle(plants, ticks=50)
        opening = plants.total_nutrients()
        harvested = plants.graze(np.full(2, 5.0), np.full(2, 5.0), np.ones(2))
        assert harvested.sum() > 0.0
        # The nutrients that left with the harvest are held against the ledger rather than
        # vanishing; #21's decomposition is what will eventually return them to the soil.
        assert plants.exported_nutrients > 0.0
        assert plants.total_nutrients() == pytest.approx(opening, rel=1e-9)


class TestGrazing:
    def test_grazing_depletes_only_the_grazed_cell(self):
        plants = make_plants()
        settle(plants, ticks=50)
        before = plants.biomass[5, 5]
        untouched = plants.biomass[5, 8]

        plants.graze(np.array([5.0]), np.array([5.0]), np.array([before]))

        assert plants.biomass[5, 5] == pytest.approx(0.0, abs=1e-9)
        assert plants.biomass[5, 8] == untouched

    def test_a_grazed_cell_regrows(self):
        plants = make_plants()
        settle(plants, ticks=50)
        plants.graze(np.array([5.0]), np.array([5.0]), np.array([plants.biomass[5, 5]]))
        settle(plants, ticks=50)
        assert plants.biomass[5, 5] > 0.0

    def test_sustained_grazing_lowers_the_equilibrium_it_grazes(self):
        # Poor soil, so standing crop is nutrient-limited rather than light-limited: what the
        # grazer carries away is what the cell then cannot rebuild. This is the closed loop
        # showing up as real spatial competition — the same test on rich soil would pass for the
        # wrong reason, because light would still cap regrowth at the ungrazed equilibrium.
        plants = make_plants(config=replace(CONFIG, initial_soil_nutrients=5.0))
        settle(plants, ticks=200)
        grazed_position = (np.array([5.0]), np.array([5.0]))
        for _ in range(200):
            plants.grow()
            plants.graze(*grazed_position, np.array([plants.biomass[5, 5] * 0.2]))
        assert plants.biomass[5, 5] < plants.biomass[5, 8] * 0.5

    def test_grazers_never_harvest_more_than_the_cell_holds(self):
        plants = make_plants()
        settle(plants, ticks=50)
        standing = plants.biomass[5, 5]
        # Three grazers on one cell, each demanding the whole standing crop.
        harvested = plants.graze(np.full(3, 5.0), np.full(3, 5.0), np.full(3, standing * 2.0))
        assert harvested.sum() == pytest.approx(standing, rel=1e-9)
        assert plants.biomass[5, 5] == pytest.approx(0.0, abs=1e-9)

    def test_contending_grazers_split_the_cell_in_proportion_to_demand(self):
        plants = make_plants()
        settle(plants, ticks=50)
        standing = plants.biomass[5, 5]
        harvested = plants.graze(
            np.full(2, 5.0), np.full(2, 5.0), np.array([standing, standing * 3.0])
        )
        assert harvested[1] == pytest.approx(3.0 * harvested[0], rel=1e-9)

    def test_grazers_on_different_cells_do_not_compete(self):
        plants = make_plants()
        settle(plants, ticks=50)
        demand = np.array([0.01, 0.01])
        harvested = plants.graze(np.array([2.0, 8.0]), np.array([5.0, 5.0]), demand)
        assert harvested == pytest.approx(demand)

    def test_modest_demand_is_met_in_full(self):
        plants = make_plants()
        settle(plants, ticks=50)
        available = plants.biomass_at(np.array([5.0]), np.array([5.0]))[0]
        harvested = plants.graze(np.array([5.0]), np.array([5.0]), np.array([available / 4.0]))
        assert harvested[0] == pytest.approx(available / 4.0, rel=1e-9)

    def test_negative_demand_is_rejected(self):
        plants = make_plants()
        with pytest.raises(ValueError, match="demand"):
            plants.graze(np.array([5.0]), np.array([5.0]), np.array([-1.0]))

    def test_grazing_outside_the_world_is_rejected(self):
        plants = make_plants()
        with pytest.raises(ValueError, match="bounds"):
            plants.graze(np.array([500.0]), np.array([5.0]), np.array([1.0]))


class TestBiomassAt:
    def test_reports_the_cell_a_grazer_would_actually_eat(self):
        plants = make_plants()
        settle(plants, ticks=50)
        # Positions resolve to the containing cell, not an interpolation of its neighbours: a
        # grazer must never be shown biomass that `graze` will not then hand it.
        assert plants.biomass_at(np.array([5.4]), np.array([5.4]))[0] == plants.biomass[5, 5]

    def test_reflects_depletion_immediately(self):
        plants = make_plants()
        settle(plants, ticks=50)
        x, y = np.array([5.0]), np.array([5.0])
        before = plants.biomass_at(x, y)[0]
        plants.graze(x, y, np.array([before / 2.0]))
        assert plants.biomass_at(x, y)[0] == pytest.approx(before / 2.0, rel=1e-9)

    def test_reading_outside_the_world_is_rejected(self):
        plants = make_plants()
        with pytest.raises(ValueError, match="bounds"):
            plants.biomass_at(np.array([-1.0]), np.array([5.0]))


class TestPerceive:
    """How a forager finds food it is not already standing on (#93).

    `perceive` reports patches, never a choice between them: which patch is worth walking to is
    the hunger drive's decision (#22), and the field has no business making it. So these tests
    assert what is *visible* and what is not, and never that some particular patch wins.

    Biomass is set directly rather than grown, so each case controls exactly which cells hold
    food and the assertions are about perception alone.
    """

    def test_reports_the_biomass_of_every_cell_within_reach(self):
        plants = make_plants()
        plants.biomass[5, 5] = 7.0
        plants.biomass[5, 7] = 3.0

        _, _, biomass = plants.perceive(np.array([5.0]), np.array([5.0]), np.array([2.0]))

        assert sorted(biomass[0][biomass[0] > 0.0]) == [3.0, 7.0]

    def test_reports_positions_that_graze_will_honour(self):
        # The coherence contract that makes the whole query usable: every patch is a position
        # `graze` accepts and `biomass_at` agrees with, so what a forager walks toward is what it
        # can then actually eat. Without this, perception and feeding could disagree and a
        # herbivore would starve standing on the meadow it aimed for.
        plants = make_plants()
        settle(plants, ticks=50)
        patch_x, patch_y, biomass = plants.perceive(
            np.array([5.0]), np.array([5.0]), np.array([2.0])
        )
        visible = biomass[0] > 0.0

        assert np.allclose(
            plants.biomass_at(patch_x[0][visible], patch_y[0][visible]), biomass[0][visible]
        )

    def test_sight_range_gates_what_a_forager_can_find(self):
        # CLAUDE.md §2.5: if perception were unlimited, sight range would be priced by the energy
        # budget while buying nothing for foraging, and only predator avoidance would select on it.
        plants = make_plants()
        plants.biomass[5, 8] = 4.0
        x, y = np.array([5.0]), np.array([5.0])

        _, _, short = plants.perceive(x, y, np.array([2.0]))
        _, _, long = plants.perceive(x, y, np.array([3.5]))

        assert short.sum() == 0.0
        assert long.sum() == 4.0

    def test_each_forager_gets_its_own_radius(self):
        plants = make_plants()
        plants.biomass[5, 8] = 4.0
        positions = np.full(2, 5.0)

        _, _, biomass = plants.perceive(positions, positions, np.array([2.0, 3.5]))

        assert biomass[0].sum() == 0.0
        assert biomass[1].sum() == 4.0

    def test_the_cell_underfoot_is_always_perceived(self):
        # A forager knows what it is standing on regardless of how far it can see: `graze` already
        # works at its own position, so a perception that hid it would contradict feeding.
        plants = make_plants()
        plants.biomass[5, 5] = 7.0

        _, _, biomass = plants.perceive(np.array([5.0]), np.array([5.0]), np.array([0.0]))

        assert biomass.sum() == 7.0

    def test_reflects_grazing_immediately(self):
        plants = make_plants()
        plants.biomass[5, 5] = 8.0
        x, y = np.array([5.0]), np.array([5.0])
        plants.graze(x, y, np.array([6.0]))

        _, _, biomass = plants.perceive(x, y, np.array([1.0]))

        assert biomass.sum() == pytest.approx(2.0, rel=1e-9)

    def test_a_forager_at_the_edge_perceives_no_food_outside_the_world(self):
        # Every cell holds food, so any phantom patch beyond the border would show up as biomass
        # a grazer could never reach — and would pull it into the map edge forever.
        plants = make_plants()
        plants.biomass[:] = 1.0

        patch_x, patch_y, biomass = plants.perceive(
            np.array([0.0]), np.array([0.0]), np.array([1.0])
        )

        # Reachable in-world cells: the corner itself and its two orthogonal neighbours. The
        # diagonal one sits at sqrt(2) and is out of range; the other six window cells are
        # outside the world.
        assert (biomass[0] > 0.0).sum() == 3
        assert np.all((patch_x >= 0.0) & (patch_x <= plants.terrain.world_width))
        assert np.all((patch_y >= 0.0) & (patch_y <= plants.terrain.world_height))

    def test_no_foragers_yields_no_patches(self):
        plants = make_plants()
        empty = np.zeros(0)

        patch_x, _, biomass = plants.perceive(empty, empty, empty)

        assert patch_x.shape[0] == 0
        assert biomass.shape[0] == 0

    def test_perceiving_outside_the_world_is_rejected(self):
        plants = make_plants()
        with pytest.raises(ValueError, match="bounds"):
            plants.perceive(np.array([500.0]), np.array([5.0]), np.array([1.0]))

    def test_negative_radius_is_rejected(self):
        plants = make_plants()
        with pytest.raises(ValueError, match="radius"):
            plants.perceive(np.array([5.0]), np.array([5.0]), np.array([-1.0]))

    def test_mismatched_input_lengths_are_rejected(self):
        plants = make_plants()
        with pytest.raises(ValueError, match="same length"):
            plants.perceive(np.array([5.0, 5.0]), np.array([5.0, 5.0]), np.array([1.0]))


class TestClimateZoneProductivity:
    """Issue #18's "done when": productivity visibly differs across climate zones.

    Directional, never exact (CLAUDE.md §6): the assertion is which zone out-produces which, so
    retuning the coefficients is free but silently flattening the climate response is not.
    """

    def test_warm_zones_outproduce_cold_ones_on_identical_flat_ground(self):
        # Flat terrain holds moisture and aspect constant across the whole grid, so the only
        # thing varying between the bands compared below is climate.
        plants = make_plants(
            heights=np.zeros((41, 41), dtype=np.float32),
            equator_y=0.0,
            equator_temperature=30.0,
            latitude_gradient=0.8,
        )
        settle(plants)

        zones = plants.climate.zone_labels()
        biomass_by_zone = {
            zone: plants.biomass[zones == zone].mean() for zone in np.unique(zones)
        }
        assert set(biomass_by_zone) == {"tropical", "temperate", "tundra"}
        assert biomass_by_zone["tropical"] > biomass_by_zone["temperate"]
        assert biomass_by_zone["temperate"] > biomass_by_zone["tundra"]
        # "Visibly" differs: the coldest band is not merely poorer, it is barren.
        assert biomass_by_zone["tundra"] == pytest.approx(0.0, abs=1e-9)


class TestTickIntegration:
    def test_grow_is_usable_as_a_tick_system(self):
        # TickLoop systems take no arguments (core.world.tick.System); `grow` must fit that
        # signature or the plant field cannot be advanced by the loop at all.
        plants = make_plants()
        systems = [plants.grow]
        for system in systems:
            system()
        assert np.all(plants.biomass > 0.0)
