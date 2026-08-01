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
from core.world.diffusion import DiffusionConfig
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
    forage_diffusion=DiffusionConfig(range=4.0, climb_penalty=0.5),
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
    "forage_diffusion",
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


class TestReturningNutrients:
    """The way back onto the ledger. Feeding assimilates only part of a mouthful and the rest is
    faeces (#19); #21's decomposition will use the same door for a whole carcass."""

    def grazed(self, ticks=50):
        plants = make_plants()
        settle(plants, ticks=ticks)
        plants.graze(np.array([5.0]), np.array([5.0]), np.array([plants.biomass[5, 5]]))
        return plants

    def test_returned_nutrients_land_in_the_cell_they_are_dropped_in(self):
        plants = self.grazed()
        before = plants.soil_nutrients[5, 5]

        plants.return_nutrients(np.array([5.0]), np.array([5.0]), np.array([2.0]))

        assert plants.soil_nutrients[5, 5] > before
        assert plants.soil_nutrients[5, 5] - before == pytest.approx(
            2.0 * CONFIG.nutrient_per_biomass
        )

    def test_it_comes_off_the_ledger_it_went_onto(self):
        plants = self.grazed()
        outstanding = plants.exported_nutrients

        plants.return_nutrients(np.array([5.0]), np.array([5.0]), np.array([2.0]))

        assert plants.exported_nutrients == pytest.approx(
            outstanding - 2.0 * CONFIG.nutrient_per_biomass
        )

    def test_the_total_is_untouched_because_this_only_moves_nutrients(self):
        plants = self.grazed()
        opening = plants.total_nutrients()

        plants.return_nutrients(np.array([5.0]), np.array([5.0]), np.array([2.0]))

        assert plants.total_nutrients() == pytest.approx(opening, rel=1e-9)

    def test_several_depositors_in_one_cell_accumulate(self):
        plants = self.grazed()
        before = plants.soil_nutrients[5, 5]

        plants.return_nutrients(np.full(3, 5.0), np.full(3, 5.0), np.full(3, 1.0))

        assert plants.soil_nutrients[5, 5] - before == pytest.approx(
            3.0 * CONFIG.nutrient_per_biomass
        )

    def test_returning_nothing_is_free_rather_than_an_error(self):
        """The ordinary case: a perfectly efficient gut, and every animal standing on ground it has
        already stripped."""
        plants = self.grazed()
        opening = plants.total_nutrients()

        plants.return_nutrients(np.array([5.0]), np.array([5.0]), np.array([0.0]))

        assert plants.total_nutrients() == pytest.approx(opening, rel=1e-9)

    def test_a_negative_return_is_rejected(self):
        plants = self.grazed()

        with pytest.raises(ValueError, match="non-negative"):
            plants.return_nutrients(np.array([5.0]), np.array([5.0]), np.array([-1.0]))

    def test_returning_more_than_ever_left_is_rejected(self):
        """Conservation would still *hold* — this only moves between two terms of the same total —
        which is exactly why it needs its own guard. The ledger going negative means nutrients were
        invented upstream, and nothing downstream could tell (§8.7)."""
        plants = self.grazed()
        outstanding = plants.exported_nutrients / CONFIG.nutrient_per_biomass

        with pytest.raises(ValueError, match="more nutrients than have left"):
            plants.return_nutrients(
                np.array([5.0]), np.array([5.0]), np.array([outstanding * 2.0])
            )


class TestFoundingStock:
    """Founders exist before anything has been grazed, so their bodies are nutrients that are out
    of the field without the field having supplied them (#21)."""

    def test_it_puts_the_bodies_on_the_ledger(self):
        plants = make_plants()

        plants.record_founding_stock(50.0)

        assert plants.exported_nutrients == pytest.approx(50.0 * CONFIG.nutrient_per_biomass)

    def test_the_world_total_grows_by_exactly_the_bodies(self):
        """Unlike every other movement of nutrients this is not a transfer — it is the rest of the
        world's budget arriving. Everything after it conserves."""
        plants = make_plants()
        opening = plants.total_nutrients()

        plants.record_founding_stock(50.0)

        assert plants.total_nutrients() == pytest.approx(
            opening + 50.0 * CONFIG.nutrient_per_biomass
        )

    def test_a_negative_founding_stock_is_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            make_plants().record_founding_stock(-1.0)


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


class TestTheForageFieldIsTickState:
    """One tick has one forage field, and every forager reads the same one (#170).

    `Hunger.appeal` used to build it per call, so the cost-aware diffusion was recomputed once per
    drive that read it. Exactly one drive reads it today, which is why nothing was wrong with the
    numbers and why nothing caught it either.
    """

    def test_a_fresh_field_matches_the_biomass_beside_it(self):
        """Never a held field that disagrees with the crop it describes — the attribute is built at
        construction rather than left empty."""
        plants = make_plants()

        np.testing.assert_allclose(plants.forage, plants.forage_field())

    def test_rebuilding_picks_up_growth(self):
        plants = make_plants()
        before = plants.forage.copy()
        settle(plants, ticks=50)

        plants.rebuild_forage()

        assert plants.forage.max() > before.max()

    def test_it_is_stale_until_rebuilt(self):
        """The property that makes this a tick step rather than a cache: nothing invalidates it, so
        the *order* is what guarantees freshness (§8.7 — a stamp that is wrong raises nothing)."""
        plants = make_plants()
        settle(plants, ticks=50)
        # `grow` does not touch the forage field — only the registered step does, which is the
        # whole point — so it has to be refreshed before it can be shown to go stale.
        plants.rebuild_forage()
        stale = plants.forage.copy()

        plants.graze(np.array([5.0]), np.array([5.0]), np.array([plants.biomass[5, 5]]))

        np.testing.assert_array_equal(plants.forage, stale)
        plants.rebuild_forage()
        assert plants.forage[5, 5] < stale[5, 5]


class TestForageField:
    """How a forager finds food it is not already standing on (#93).

    The field reports what is *reachable* from every cell and ranks nothing: whether a reading is
    strong enough to notice is a question about the animal's sight phenotype, and which way to walk
    is the gradient. Both belong to the drive that asks (`core.behaviour.drives.Hunger`), which is
    why nothing here takes a gene or returns a winner.

    This replaced a per-forager list of candidate patches. The list could only be ranked by
    distance, so a meadow across a gorge scored exactly as well as one on open ground; diffusing the
    crop makes the distance discount and the cost of the ground between into one mechanism.
    """

    def test_standing_crop_is_readable_from_a_distance(self):
        plants = make_plants()
        plants.biomass[:] = 0.0
        plants.biomass[5, 5] = 40.0

        field = plants.forage_field()

        assert field[5, 7] > 0.0
        assert field[5, 7] < field[5, 6]

    def test_an_empty_world_reads_empty_everywhere(self):
        plants = make_plants()
        plants.biomass[:] = 0.0

        assert (plants.forage_field() == 0.0).all()

    def test_the_field_never_reports_more_than_grows_anywhere(self):
        """A gradient may only ever point at food that is really there, so the spreading must not
        amplify: every reading is bounded by the largest standing crop in the world."""
        plants = make_plants()
        plants.biomass[:] = 0.0
        plants.biomass[5, 5] = 40.0

        assert plants.forage_field().max() <= 40.0 + 1e-6

    def test_grazing_changes_what_is_readable(self):
        """Rebuilt from the crop rather than cached: a stale field would send foragers at ground
        that was stripped bare while they walked toward it."""
        plants = make_plants()
        plants.biomass[:] = 0.0
        plants.biomass[5, 5] = 40.0
        before = plants.forage_field()[5, 6]

        plants.graze(np.array([5.0]), np.array([5.0]), np.array([39.0]))

        assert plants.forage_field()[5, 6] < before

    def test_a_ridge_damps_what_lies_behind_it(self):
        """The whole reason the field is cost-aware: the same crop, the same distance away, reads
        fainter when reaching it means a climb."""
        heights = np.zeros((11, 11), dtype=np.float32)
        heights[:, 7] = 30.0
        walled = make_plants(heights=heights)
        open_ground = make_plants()
        for plants in (walled, open_ground):
            plants.biomass[:] = 0.0
            plants.biomass[5, 9] = 40.0

        assert walled.forage_field()[5, 5] < open_ground.forage_field()[5, 5]

    def test_the_field_reads_higher_nearer_the_food(self):
        plants = make_plants()
        plants.biomass[:] = 0.0
        plants.biomass[5, 8] = 40.0
        field = plants.forage_field()

        toward, away = plants.forage_at(field, np.array([6.0, 4.0]), np.array([5.0, 5.0]))

        assert toward > away

    def test_a_whole_block_of_candidate_options_is_sampled_in_one_call(self):
        """#114 reads `(n_entities, n_options)` positions at once, so this takes any shape rather
        than a flat population vector — `_cell_indices` is elementwise and needs no reshaping.
        """
        plants = make_plants()
        plants.biomass[:] = 0.0
        plants.biomass[:, 9] = 40.0
        field = plants.forage_field()
        x = np.tile(np.array([6.0, 4.0, 5.0]), (7, 1))
        y = np.tile(np.linspace(3.0, 7.0, 7)[:, None], (1, 3))

        readings = plants.forage_at(field, x, y)

        assert readings.shape == (7, 3)
        assert (readings[:, 0] > readings[:, 1]).all()

    def test_sampling_outside_the_world_is_rejected(self):
        """A candidate off the map is a bug in whatever generated it, and defaulting to an edge
        cell would let foragers read a border strip forever (§8.7).
        """
        plants = make_plants()
        field = plants.forage_field()

        with pytest.raises(ValueError, match="outside terrain bounds"):
            plants.forage_at(field, np.array([-1.0]), np.array([1.0]))

    def test_a_source_that_does_not_cover_the_grid_is_rejected(self):
        plants = make_plants()

        with pytest.raises(ValueError, match="terrain grid"):
            plants.forage_diffusion.spread(np.zeros((3, 3), dtype=np.float32))


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
