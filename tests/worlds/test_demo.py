"""Headless coverage of the named worlds: that each config is one a world can be built from.

These worlds were the viewer's and moved to `worlds/` when `api/` became the only thing a client
may import (#35). The viewer's frame path over them is tested in
`tests/clients/viewer/test_frame_path.py`, through the boundary.

Since #115 this is config only — `core.world.assembly.build_world` does the wiring — so what these
tests cover is the *world config* being one a world can actually be built from, and holding the
properties each world was tuned for.
"""

import dataclasses

import numpy as np
import pytest

from worlds.demo import (
    DEFAULT_WORLD,
    build_demo_world,
    build_named_world,
    build_temperate_world,
    demo_world_config,
    temperate_world_config,
)
from core.invariants import default_registry
from core.selection import Selection
from core.world.assembly import build_world



class TestBuildDemoWorld:
    def test_allocates_every_entity_it_was_asked_for(self):
        world = build_demo_world(seed=0, n_entities=25)

        assert int(world.store.alive.sum()) == 25

    def test_entities_stand_on_the_terrain_surface(self):
        """§2.6 stores z from the start but movement is surface-locked, so a demo entity's z is
        the ground beneath it rather than an independent coordinate it could drift from.
        """
        world = build_demo_world(seed=1, n_entities=25)
        alive = world.store.alive

        surface = world.terrain.elevation_at(world.store.x[alive], world.store.y[alive])

        np.testing.assert_allclose(world.store.z[alive], surface, rtol=1e-6)

    def test_a_freshly_built_world_satisfies_every_invariant_checkable_today(self):
        world = build_demo_world(seed=2, n_entities=50)
        registry = default_registry(
            0.0, world.terrain.world_width, 0.0, world.terrain.world_height, plants=world.plants
        )

        registry.check_all(world.store, world.loop.tick_count)  # must not raise

    def test_the_same_seed_rebuilds_the_same_world(self):
        """The simulation is non-deterministic by design (§2.2), but *generation* is seeded so a
        crash can be replayed. A demo world that could not be rebuilt would make the viewer
        useless for exactly the diagnosis §3.3 exists for.
        """
        first = build_demo_world(seed=7, n_entities=30)
        second = build_demo_world(seed=7, n_entities=30)

        np.testing.assert_array_equal(first.store.x, second.store.x)
        np.testing.assert_array_equal(first.store.y, second.store.y)
        np.testing.assert_array_equal(first.store.genes, second.store.genes)

    def test_the_loop_advances_the_store_it_was_built_around(self):
        world = build_demo_world(seed=3, n_entities=10)
        registry = default_registry(
            0.0, world.terrain.world_width, 0.0, world.terrain.world_height, plants=world.plants
        )

        world.loop.advance(5)

        assert world.loop.tick_count == 5
        registry.check_all(world.store, world.loop.tick_count)  # must not raise
        # The loop now runs the settled order (#115), so five ticks are five ticks of a live world:
        # ages advance, pools drain, and the position snapshot the renderer interpolates from is
        # the one this store holds.
        np.testing.assert_array_equal(world.loop.current_positions[0], world.store.x)
        assert (world.store.age[world.store.alive] == 5).all()


def living(world):
    """Everyone born and alive: the breeders whose mean mutability is each species' step (#304)."""
    return Selection.from_mask(world.store.alive & (world.store.age >= 0))


class TestTheShippedTuningIsNotDegenerate:
    """Guards against the failure this world keeps having: a config constant whose name says one
    thing while it silently does another.

    The interbreeding threshold once shipped at 8.0 against a median pairwise distance of ~29, so
    `interbreeding_probability` returned zero for 94% of candidate couples and suppressed births
    roughly 24-fold. Every test passed throughout — nothing asserted that the gate was *open*.
    """

    def test_most_couples_can_actually_breed(self):
        """The gate exists to stop diverged populations interbreeding. With one population and no
        divergence it should be nearly transparent, so a majority failing is a misconfiguration
        rather than ecology. Since #16, ordinary couples are fully fertile by construction."""
        world = build_demo_world(seed=2, n_entities=120)
        world.loop.advance(200)
        rows = np.flatnonzero(world.store.alive & (world.store.age >= 0))
        rng = np.random.default_rng(2)
        a, b = rng.choice(rows, 500), rng.choice(rows, 500)

        chance = world.speciation.compatibility(a, b, living(world))

        assert float((chance > 0.0).mean()) > 0.9
        assert float(np.median(chance)) > 0.9

    def test_the_gate_still_closes_on_a_genuinely_diverged_pair(self):
        """The other half: a threshold set so high that nothing is ever incompatible would pass the
        test above while making speciation impossible (#16)."""
        world = build_demo_world(seed=2, n_entities=40)
        rows = np.flatnonzero(world.store.alive)[:2]
        genes = world.genetics.genes_at(rows)
        # Push one animal's smell far past anything drift produces: signature is what distance
        # reads (#16).
        genes[1, world.genes.index_of("signature_0")] += 10.0
        world.genetics.set_genes(
            Selection.from_indices(rows, world.store.capacity), genes
        )

        chance = world.speciation.compatibility(rows[:1], rows[1:], living(world))

        assert float(chance[0]) == 0.0

    def test_founders_start_as_one_interbreeding_population(self):
        """#16: founders sit at one point in cue space, so a world begins as one population and a
        later split is drift and isolation rather than the founding spread (speciation-signature.md).
        """
        world = build_demo_world(seed=2, n_entities=200)
        columns = world.speciation.columns
        signatures = world.genetics.heritable_columns_at(world.founders.to_indices(), columns)

        assert float(signatures.min()) >= 0.45 and float(signatures.max()) <= 0.55
        assert world.species.n_species == 1

    def test_scent_emission_charges_no_upkeep(self):
        """§2.5: low emission is already a survival benefit, so any upkeep makes silence both
        cheaper and safer and drives emission to zero in every lineage. That would spend the
        variation #20's mate-finding counterweight needs to act on before it ever arrives (#149)."""
        world = build_demo_world(seed=2, n_entities=10)

        assert world.genes.spec("scent_emission").cost == 0.0

    def test_a_year_is_about_six_founding_generations(self):
        """§2.1's table: a year is a lifespan, ~6 generations, ~1.5 per season (#159). The year is
        read off the founders' breeding ranges, so retuning those moves the calendar with them
        rather than leaving a constant that silently stopped meaning one lifespan."""
        world = build_demo_world(seed=2, n_entities=10)
        ranges = world.config.founder_gene_ranges
        generation = sum(ranges["maturity_age"]) / 2.0 + sum(ranges["gestation_length"]) / 2.0

        year = world.config.climate.seasons.year_length

        assert year / generation == pytest.approx(6.0, rel=0.01)
        assert (year / 4) / generation == pytest.approx(1.5, rel=0.01)


class TestTemperateWorld:
    """The second world (#261): the demo world with a climate whose range straddles neutral, so the
    comfortable ground moves with the seasons and migration has somewhere to go.

    The demo world sits wholly above `neutral_temperature`, so its coolest ground is the best ground
    in every season, and every measurement taken so far runs on it. The owner chose a second world
    over retuning that one; these tests hold the two to differing in the one thing being tested.
    """

    def test_it_differs_from_the_demo_world_only_in_climate(self):
        """§2.1: two worlds to keep tuned is the accepted cost, so nothing but the climate may drift
        between them — the year, the founders, every coefficient outside the sky stay shared."""
        demo = demo_world_config(n_entities=30, seed=4)
        temperate = temperate_world_config(n_entities=30, seed=4)

        assert temperate.climate != demo.climate
        assert dataclasses.replace(temperate, climate=demo.climate) == demo
        # The calendar and the weather are ratio-table entries and statements about the sky's
        # texture, not about where neutral falls, so they are shared too.
        assert temperate.climate.seasons == demo.climate.seasons
        assert temperate.climate.equator_y == demo.climate.equator_y

    def test_its_gradients_stay_the_tuned_pair(self):
        """`lapse_rate` and `latitude_gradient` are one tuning decision (#112): a peak about as cold
        as a pole. Steepening one without the other would make terrain matter less or more than
        latitude for a reason nobody chose."""
        demo = demo_world_config(n_entities=10, seed=0).climate
        temperate = temperate_world_config(n_entities=10, seed=0).climate

        assert temperate.lapse_rate / temperate.latitude_gradient == pytest.approx(
            demo.lapse_rate / demo.latitude_gradient
        )

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_its_temperature_range_straddles_neutral_across_the_year(self, seed):
        """What makes migration possible at all: poles below neutral in winter, the equator above it
        in summer, and the share of ground on the warm side swinging with the season — so the
        comfortable band moves, rather than staying pinned to the coolest edge as on the demo map.
        """
        config = temperate_world_config(n_entities=10, seed=seed)
        world = build_world(config, seed=seed)
        neutral = config.metabolism.neutral_temperature
        year = config.climate.seasons.year_length
        rows = np.arange(world.terrain.heights.shape[0]) * world.terrain.cell_size
        latitude = np.abs(rows - config.climate.equator_y)
        poles = latitude >= latitude.max() - 1.0
        equator = latitude <= 0.5

        summer = world.climate.field_at(year // 4)
        winter = world.climate.field_at(3 * year // 4)

        assert (winter[poles] < neutral).all()
        assert (summer[equator] > neutral).any()
        assert float((summer > neutral).mean()) - float((winter > neutral).mean()) > 0.3

    def test_it_charges_the_demo_worlds_thermal_bill(self):
        """`thermoregulation_rate` is shared, and it was derived from how far the demo map's ground
        sits from neutral (#176). It stays valid only if the temperate climate, averaged over its
        ground and its year, sits about as far from neutral in squared deviation — the climate
        moves *where* the bill falls, not how large it typically is."""

        def mean_squared_deviation(config, seed):
            world = build_world(config, seed=seed)
            year = config.climate.seasons.year_length
            fields = np.stack(
                [world.climate.field_at(tick) for tick in range(0, year, year // 12)]
            )
            return float(np.mean((fields - config.metabolism.neutral_temperature) ** 2))

        seeds = range(1, 5)
        demo = np.mean([mean_squared_deviation(demo_world_config(5, s), s) for s in seeds])
        temperate = np.mean(
            [mean_squared_deviation(temperate_world_config(5, s), s) for s in seeds]
        )

        assert temperate == pytest.approx(demo, rel=0.15)



class TestChoosingAWorld:
    """The viewer opens either world by name (#268). The two builders must differ in config and
    nothing else, or watching the temperate world would also mean watching it unrecorded."""

    def test_the_temperate_builder_runs_the_temperate_climate(self):
        world = build_temperate_world(seed=4, n_entities=20)

        assert world.config == temperate_world_config(n_entities=20, seed=4)

    def test_the_temperate_world_is_composed_like_the_demo_world(self):
        demo = build_demo_world(seed=4, n_entities=20)
        temperate = build_temperate_world(seed=4, n_entities=20)

        assert temperate.loop.metrics is not None
        assert type(temperate.loop.metrics) is type(demo.loop.metrics)
        assert temperate.loop.interventions.balance == demo.loop.interventions.balance

    @pytest.mark.parametrize("name", ["demo", "temperate"])
    def test_a_world_is_built_by_its_name(self, name):
        world = build_named_world(name, seed=3, n_entities=10)

        expected = {"demo": demo_world_config, "temperate": temperate_world_config}[name]
        assert world.config == expected(n_entities=10, seed=3)

    def test_an_unknown_world_name_is_refused_naming_the_choices(self):
        with pytest.raises(ValueError, match="demo.*temperate"):
            build_named_world("tropical", seed=0, n_entities=10)

    def test_the_demo_world_is_the_default(self):
        """Every measurement so far was taken on the demo world, so opening the viewer without
        asking for anything must still show it."""
        world = build_named_world(DEFAULT_WORLD, seed=3, n_entities=10)

        assert world.config == demo_world_config(n_entities=10, seed=3)


class TestSpeciationRunsInTheWorld:
    """#16 end to end, through the tick order: the assembled world finds and records a split on
    its own. Drift takes hundreds of generations (speciation-signature.md), so the test moves one
    group's smell far away directly and then asks the world to notice."""

    def diverge_some(self, world, count):
        rows = np.flatnonzero(world.store.alive)[:count]
        genes = world.genetics.genes_at(rows)
        for column in world.speciation.columns:
            genes[:, column] += 1.0
        world.genetics.set_genes(Selection.from_indices(rows, world.store.capacity), genes)
        return world.store.row_ids()[rows]

    def test_a_diverged_group_becomes_a_species_at_the_next_check(self):
        world = build_demo_world(seed=3, n_entities=120)
        moved = self.diverge_some(world, 40)
        period = world.config.speciation.every_n_ticks

        world.loop.advance(1)  # tick 0 is a check
        assert world.species.n_species == 2
        (daughter,) = world.speciation.lineage.parents()
        living = world.store.row_ids()
        moved_rows = np.flatnonzero(np.isin(living, moved) & world.store.alive)
        assert (world.store.species_id[moved_rows] == daughter).all()

        world.loop.advance(period)
        assert world.species.n_species == 2  # a split happens once, not every check

    def test_nothing_splits_an_undisturbed_world(self):
        world = build_demo_world(seed=3, n_entities=120)
        world.loop.advance(3 * world.config.speciation.every_n_ticks + 1)

        assert world.species.n_species == 1

    def test_the_two_species_never_conceive_together(self):
        world = build_demo_world(seed=3, n_entities=120)
        self.diverge_some(world, 40)
        world.loop.advance(1)
        rows = np.flatnonzero(world.store.alive & (world.store.age >= 0))
        species = world.store.species_id[rows]
        a = rows[species == 0][:20]
        b = rows[species == 1][:20]

        assert (world.speciation.compatibility(a, b, living(world)) == 0.0).all()
