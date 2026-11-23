"""The api boundary's contract (#35), test-first where it could be.

Tests reach into `session._world` only to *set up* a world state or to read the truth a session
should report. Nothing here asserts on how the session computes it.
"""

import numpy as np
import pytest

import api
from api import CullRequest, FenceRequest, Refused
from core.ecology.plants import Plants, PlantsConfig
from core.world.climate import Climate, ClimateConfig
from core.world.diffusion import DiffusionConfig
from core.world.terrain import Terrain
from core.world.water import Water


def session_on(founders=40, seed=3, world="demo"):
    return api.create(world, seed=seed, founders=founders)


def row_of(session, entity_id):
    return int(np.flatnonzero(session._world.store.row_ids() == entity_id)[0])


class TestCreate:
    def test_the_named_worlds_are_offered(self):
        assert set(api.worlds()) == {"demo", "temperate"}

    def test_an_unknown_world_is_refused_rather_than_defaulted(self):
        with pytest.raises(ValueError, match="unknown world"):
            api.create("atlantis", seed=0, founders=10)

    def test_it_describes_the_world_it_opened(self):
        info = session_on(founders=12, seed=5).describe()

        assert info.name == "demo"
        assert info.seed == 5
        assert info.width > 0 and info.height > 0
        assert "speed" in info.gene_names
        assert info.field_layers == api.FIELD_LAYERS
        assert info.conditions == api.CONDITIONS


class TestControl:
    def test_advance_runs_the_ticks_and_says_where_the_world_stands(self):
        session = session_on()

        status = session.advance(7)

        assert status.tick == 7
        assert not status.collapsed
        assert status.living == len(session.frame().ids.to_numpy())

    def test_advancing_no_ticks_is_refused(self):
        with pytest.raises(ValueError, match="at least one tick"):
            session_on().advance(0)

    def test_a_save_reloads_into_the_same_world(self, tmp_path):
        session = session_on()
        session.advance(5)
        path = session.save(str(tmp_path / "slot.npz"))
        saved = session.frame()
        session.advance(10)

        status = session.load(path)

        assert status.tick == 5
        assert session.frame() == saved

    def test_loading_nothing_is_a_refusal_not_a_crash(self, tmp_path):
        with pytest.raises(Refused, match="no save"):
            session_on().load(str(tmp_path / "missing.npz"))

    def test_a_snapshot_from_other_rules_is_refused(self, tmp_path):
        """§2.8: a world runs under the rules it was made with, so a snapshot cannot be opened
        into a differently tuned one. Here the founder count changes the config."""
        path = session_on(founders=40).save(str(tmp_path / "slot.npz"))

        with pytest.raises(Refused, match="different world config"):
            session_on(founders=41).load(path)


class TestFrame:
    def test_it_holds_every_living_animal_once(self):
        session = session_on()
        store = session._world.store
        living = store.alive & (store.age >= 0)

        ids = session.frame().ids.to_numpy()

        assert sorted(ids.tolist()) == sorted(store.row_ids()[living].tolist())

    def test_a_gestating_young_is_not_drawn(self):
        """It is allocated but not yet a participant (#20), and it stands motionless where it
        was conceived."""
        session = session_on()
        some_id = int(session.frame().ids.to_numpy()[0])
        session._world.store.age[row_of(session, some_id)] = -10

        assert some_id not in session.frame().ids.to_numpy()

    def test_columns_are_parallel(self):
        frame = session_on().frame()
        n = frame.ids.shape[0]

        assert frame.x.shape == frame.y.shape == frame.z.shape == frame.species.shape == (n,)

    def test_a_survivor_keeps_its_id_across_ticks(self):
        """The property a client blends two frames by. A row would not have it (#119)."""
        session = session_on()
        before = session.frame()
        session.advance(3)
        after = session.frame()

        survivors = np.intersect1d(before.ids.to_numpy(), after.ids.to_numpy())

        assert survivors.size > 0
        for entity_id in survivors[:5]:
            row = row_of(session, int(entity_id))
            index = int(np.flatnonzero(after.ids.to_numpy() == entity_id)[0])
            assert after.x.to_numpy()[index] == session._world.store.x[row]


class TestCondition:
    def test_it_is_aligned_with_the_frame_of_the_same_tick(self):
        session = session_on()
        frame = session.frame()

        for mode in api.CONDITIONS:
            reading = session.condition(mode)
            assert reading.tick == frame.tick
            assert reading.values.shape == frame.ids.shape

    def test_a_hungrier_animal_has_the_larger_deficit(self):
        session = session_on()
        world = session._world
        first, second = (int(i) for i in session.frame().ids.to_numpy()[:2])
        world.store.energy[row_of(session, first)] = world.config.hunger.satiation_energy
        world.store.energy[row_of(session, second)] = 0.0

        values = session.condition("hunger").values.to_numpy()

        assert values[0] == pytest.approx(0.0)
        assert values[1] == pytest.approx(1.0)

    def test_a_thirstier_animal_has_the_larger_deficit(self):
        session = session_on()
        first, second = (int(i) for i in session.frame().ids.to_numpy()[:2])
        session._world.store.dehydration[row_of(session, first)] = 0.0
        session._world.store.dehydration[row_of(session, second)] = 1.0

        values = session.condition("thirst").values.to_numpy()

        assert values[0] < values[1]

    def test_deficits_beyond_either_end_are_clipped(self):
        """Energy can exceed satiation, and neither bound is a client's to guess."""
        session = session_on()
        world = session._world
        first, second = (int(i) for i in session.frame().ids.to_numpy()[:2])
        world.store.energy[row_of(session, first)] = world.config.hunger.satiation_energy * 10.0
        world.store.energy[row_of(session, second)] = -50.0

        values = session.condition("hunger").values.to_numpy()

        assert values[0] == 0.0
        assert values[1] == 1.0

    def test_an_unknown_condition_is_refused(self):
        with pytest.raises(ValueError, match="unknown condition"):
            session_on().condition("ennui")


class TestFieldReferences:
    """The value each layer reads as full at: fixed for the session and never measured off the
    field, because a scale that rescales itself renders a starving world and a lush one alike."""

    def test_biomass_reads_full_at_the_light_limited_steady_state(self):
        session = session_on(founders=12, seed=1)
        world = session._world
        expected = world.plants.potential_growth.max() / world.config.plants.senescence_rate

        assert session.field("biomass").reference == pytest.approx(expected)

    def test_the_light_limited_steady_state_is_where_unlimited_biomass_settles(self):
        """Justifies the number above rather than taking it on trust: with soil so deep that
        growth is never nutrient-limited, biomass converges on exactly this reference."""
        terrain = Terrain(np.full((6, 6), 50.0, dtype=np.float32), cell_size=1.0)
        climate = Climate(terrain, ClimateConfig(equator_y=3.0, equator_temperature=30.0))
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
        field = Plants(terrain, climate, Water.generate(terrain), deep_soil)
        reference = field.potential_growth.max() / deep_soil.senescence_rate
        for _ in range(400):
            field.grow()

        assert field.biomass.max() == pytest.approx(reference, rel=1e-3)

    def test_carrion_reads_full_at_what_one_strike_can_leave(self):
        """`strike_power`, not a whole body: scaled to one, the brightest cell of a settled world
        rendered at 13% and the layer looked empty for a mechanic that was running (#179)."""
        session = session_on(founders=12, seed=1)

        assert session.field("carrion").reference == pytest.approx(
            session._world.config.predation.strike_power
        )

    def test_soil_reads_full_at_the_starting_per_cell_pool(self):
        session = session_on(founders=12, seed=1)

        assert session.field("soil_nutrients").reference == pytest.approx(
            session._world.config.plants.initial_soil_nutrients
        )

    def test_the_reference_does_not_move_when_the_field_does(self):
        session = session_on(founders=12, seed=1)
        before = session.field("biomass").reference

        session._world.plants.biomass[:] = 0.0

        assert session.field("biomass").reference == before

    def test_every_offered_layer_has_a_reference(self):
        session = session_on(founders=12, seed=1)

        for layer in api.FIELD_LAYERS:
            assert session.field(layer).reference > 0.0


class TestField:
    def test_a_layer_reads_its_grid_from_the_service_that_owns_it(self):
        """Carrion does not live on `Plants`, which is why the source is a table."""
        session = session_on(founders=12, seed=1)
        world = session._world
        world.carrion.mass[3, 3] = 9.0

        assert np.array_equal(session.field("carrion").values.to_numpy(), world.carrion.mass)
        assert np.array_equal(session.field("biomass").values.to_numpy(), world.plants.biomass)

    def test_an_unknown_layer_is_refused(self):
        with pytest.raises(ValueError, match="unknown field"):
            session_on(founders=12, seed=1).field("nutrient_soup")


class TestInspect:
    def test_it_reports_the_animal_it_was_asked_about(self):
        session = session_on()
        session.advance(5)
        some_id = int(session.frame().ids.to_numpy()[0])
        row = row_of(session, some_id)

        report = session.inspect(some_id)

        assert report.id == some_id
        assert report.species == int(session._world.store.species_id[row])
        assert report.energy == pytest.approx(float(session._world.store.energy[row]))
        assert report.upkeep > 0.0
        assert report.top_speed > 0.0
        assert 0.0 < report.pace <= 1.0

    def test_every_drive_appears_largest_first(self):
        """"62% of that heading was hunger" says more than a winner's name (#114)."""
        session = session_on()
        session.advance(2)
        report = session.inspect(int(session.frame().ids.to_numpy()[0]))

        names = [share.name for share in report.drives]
        contributions = [share.contribution for share in report.drives]
        assert set(names) == set(session._world.behaviour.drive_names)
        assert contributions == sorted(contributions, reverse=True)

    def test_every_gene_is_read_both_ways(self):
        session = session_on()
        report = session.inspect(int(session.frame().ids.to_numpy()[0]))

        assert [gene.name for gene in report.genes] == list(session.describe().gene_names)

    def test_a_gestating_young_reports_its_gestation_clock(self):
        session = session_on()
        some_id = int(session.frame().ids.to_numpy()[0])
        session._world.store.age[row_of(session, some_id)] = -14

        assert session.inspect(some_id).age == -14

    def test_a_dead_animal_is_nobody_rather_than_a_corpse(self):
        """`release` leaves a row's columns untouched (#119), so a freed row still reads like an
        animal. By id, a dead one is simply gone."""
        session = session_on()
        some_id = int(session.frame().ids.to_numpy()[0])
        session._world.store.release(np.array([some_id], dtype=np.int64))

        assert session.inspect(some_id) is None

    def test_the_free_row_marker_is_not_an_animal(self):
        """A free row's id reads -1, so a lookup by id alone would describe a free row."""
        session = session_on()
        session._world.store.release(session.frame().ids.to_numpy()[:1])

        assert session.inspect(-1) is None


class TestInterventions:
    def test_a_quote_names_the_price_the_boundary_then_charges(self):
        session = session_on(founders=60, seed=1)
        cull = CullRequest(species_id=0, count=3, survivors=20)
        quote = session.quote(cull)

        session.request(cull)
        session.advance(1)

        record = session.ledger().records[-1]
        assert quote.refusal is None
        assert record.refusal is None
        assert record.cost == quote.cost

    def test_a_quote_gives_the_refusal_the_boundary_then_records(self):
        """§2.7: a cull may never take a species below its survivors."""
        session = session_on(founders=30, seed=1)
        greedy = CullRequest(species_id=0, count=25, survivors=20)
        quote = session.quote(greedy)

        session.request(greedy)
        session.advance(1)

        assert quote.refusal is not None
        assert session.ledger().records[-1].refusal == quote.refusal

    def test_a_quote_charges_nothing(self):
        session = session_on(founders=60, seed=1)
        before = session.ledger()

        session.quote(FenceRequest(1.0, 1.0, 20.0, 20.0))

        assert session.ledger() == before

    def test_a_requested_fence_stands_after_the_next_boundary(self):
        session = session_on(founders=20, seed=1)
        session.request(FenceRequest(10.0, 10.0, 30.0, 30.0))

        assert not session.barriers().blocked_north.to_numpy().any()
        session.advance(1)
        assert session.barriers().blocked_north.to_numpy().any()
        assert session.ledger().records[-1].name == "fence"

    def test_something_that_is_not_a_request_is_refused(self):
        with pytest.raises(TypeError, match="not a request"):
            session_on(founders=20).request("cull everything")


class TestSpecies:
    """#314: the lineage list a client draws, and the names the player gives it."""

    def split_off(self, session, count=12):
        from core.genetics.speciation import split
        from core.selection import Selection

        world = session._world
        rows = np.flatnonzero(world.store.alive & (world.store.age >= 0))[:count]
        return split(
            world.genetics,
            world.speciation.lineage,
            Selection.from_indices(rows, world.store.capacity),
            tick=world.loop.tick_count,
        )

    def test_a_founding_species_has_no_parent_and_holds_everyone(self):
        session = session_on(founders=30)

        (only,) = session.species()

        assert only == api.SpeciesReport(
            id=0, name=None, parent=None, split_tick=None, founders=None, living=30
        )

    def test_a_split_is_listed_after_its_parent_with_when_and_from_how_many(self):
        session = session_on(founders=30)
        session.advance(4)
        daughter = self.split_off(session, count=12)

        parent, child = session.species()

        assert child.id == daughter
        assert (child.parent, child.split_tick, child.founders) == (0, 4, 12)
        assert parent.living + child.living == session.status().living

    def test_the_status_counts_the_splits(self):
        session = session_on(founders=30)
        assert session.status().splits == 0

        self.split_off(session)

        assert session.status().splits == 1

    def test_a_name_is_applied_at_once_and_trimmed(self):
        session = session_on(founders=30)

        named = session.name(api.SpeciesNameRequest(species_id=0, name="  Grazers "))

        assert named.name == "Grazers"
        assert session.species()[0].name == "Grazers"

    @pytest.mark.parametrize(
        ("species_id", "name", "message"),
        [(7, "Ghosts", "no species 7"), (0, "   ", "blank"), (0, "x" * 41, "at most 40")],
    )
    def test_a_bad_name_is_refused(self, species_id, name, message):
        with pytest.raises(ValueError, match=message):
            session_on(founders=30).name(api.SpeciesNameRequest(species_id=species_id, name=name))

    def test_names_survive_a_save_and_a_load_rolls_them_back(self, tmp_path):
        """A restore brings back names from the snapshot's moment, not the latest ones."""
        session = session_on(founders=30)
        session.name(api.SpeciesNameRequest(species_id=0, name="Grazers"))
        path = session.save(str(tmp_path / "slot.npz"))
        session.name(api.SpeciesNameRequest(species_id=0, name="Renamed later"))

        session.load(path)

        assert session.species()[0].name == "Grazers"
