"""A world saved and reloaded is the same world (#31).

The contract is checkable in advance and was written against it (§8.1). The load-bearing test is
`test_a_reloaded_world_evolves_identically`: §3.2 makes the snapshot the only copy of a world in
existence, so "it round-trips" is not enough — what has to hold is that the world *continues*.
"""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from clients.viewer.demo_world import (
    DEMO_BUDGET,
    DEMO_METRICS,
    build_demo_world,
    demo_world_config,
)
from core.invariants import default_registry
from core.world.assembly import build_world
from core.world.cull import Cull
from metrics import MetricHistory
from persistence import SCHEMA_VERSION, SnapshotError, fingerprint, load, save

SEED = 1
FOUNDERS = 60


def world(seed=SEED, founders=FOUNDERS):
    return build_demo_world(seed=seed, n_entities=founders)


def run(ticks, seed=SEED, founders=FOUNDERS):
    built = world(seed, founders)
    built.loop.advance(ticks)
    return built


class TestARoundTripIsTheSameWorld:
    def test_every_column_comes_back_unchanged(self, tmp_path):
        saved = run(80)

        restored = world()
        load(restored, save(saved, tmp_path / "w.npz"))

        for column in ("x", "y", "z", "velocity_x", "energy", "age", "genes", "alive"):
            assert np.array_equal(
                getattr(restored.store, column), getattr(saved.store, column)
            ), column

    def test_the_tick_counter_and_the_id_counter_come_back(self, tmp_path):
        """Ids are never reused (§2.3), so resuming below the saved counter would hand a dead
        entity's id to the next birth — and nothing downstream could detect it."""
        saved = run(80)

        restored = world()
        load(restored, save(saved, tmp_path / "w.npz"))

        assert restored.loop.tick_count == saved.loop.tick_count == 80
        assert restored.store.ids_issued == saved.store.ids_issued

    def test_the_plant_field_and_its_ledger_come_back(self, tmp_path):
        """All three, because the nutrient loop is only closed across the set: their total is what
        `nutrients_are_conserved` asserts never moves (§6)."""
        saved = run(80)

        restored = world()
        load(restored, save(saved, tmp_path / "w.npz"))

        assert np.array_equal(restored.plants.biomass, saved.plants.biomass)
        assert np.array_equal(restored.plants.soil_nutrients, saved.plants.soil_nutrients)
        assert restored.plants.exported_nutrients == saved.plants.exported_nutrients
        assert restored.plants.total_nutrients() == pytest.approx(saved.plants.total_nutrients())

    def test_the_sky_comes_back_without_being_stored(self, tmp_path):
        """Weather is a function of the tick and the seed (#159), so a snapshot holds none of it
        (§3.2) and the reload evaluates it. A stored field would be a derivation that could disagree
        with the tick beside it."""
        saved = run(80)
        assert saved.config.climate.seasons is not None, "the demo world should have seasons"
        path = save(saved, tmp_path / "w.npz")

        restored = world()
        load(restored, path)

        assert np.array_equal(restored.climate.temperature, saved.climate.temperature)
        with np.load(path, allow_pickle=False) as archive:
            assert not [name for name in archive.files if "climate" in name or "temperature" in name]

    def test_a_restored_fence_reaches_the_water_field(self, tmp_path):
        """Derived and never stored (§3.2), so the wire has to reach it some other way (#275)."""
        saved = world()
        saved.barriers.enclose(30.0, 30.0, 50.0, 50.0)

        restored = world()
        unfenced = restored.hydration.reachable.copy()
        load(restored, save(saved, tmp_path / "w.npz"))

        assert np.array_equal(restored.hydration.reachable, saved.hydration.reachable)
        assert not np.array_equal(restored.hydration.reachable, unfenced)

    def test_a_world_that_grew_its_store_reloads_at_that_capacity(self, tmp_path):
        """Capacity is taken from the snapshot's own columns rather than being told separately, so
        a world that doubled while running does not need the target world to have doubled too."""
        saved = run(300)
        assert saved.store.capacity > world().store.capacity, "the run should have grown the store"

        restored = world()
        load(restored, save(saved, tmp_path / "w.npz"))

        assert restored.store.capacity == saved.store.capacity
        assert int(restored.store.alive.sum()) == int(saved.store.alive.sum())

    def test_the_loaded_world_holds_every_invariant(self, tmp_path):
        """#31's done-when. A snapshot that restored a subtly broken world would be worse than one
        that failed to load at all, because the damage would surface ticks later somewhere else."""
        saved = run(120)

        restored = world()
        load(restored, save(saved, tmp_path / "w.npz"))

        default_registry(
            0.0,
            restored.terrain.world_width,
            0.0,
            restored.terrain.world_height,
            plants=restored.plants,
            movement=restored.movement,
        ).check_all(restored.store, tick=restored.loop.tick_count)


class TestAReloadedWorldContinues:
    def test_a_reloaded_world_evolves_identically(self, tmp_path):
        """The property the whole issue reduces to, and the one #3.2 needs: a snapshot is the only
        copy of a world in existence, so it has to be the world and not a likeness of it.

        Run long enough past the load for births to have happened — those are what allocate rows,
        and the free-list order is the one piece of state that cannot be re-derived. Without it the
        two worlds diverge at the very first birth while looking identical up to it.
        """
        saved = run(80)
        restored = world()
        load(restored, save(saved, tmp_path / "w.npz"))

        saved.loop.advance(40)
        restored.loop.advance(40)

        assert int(restored.store.alive.sum()) > int(saved.store.alive.sum()) - 1
        assert np.array_equal(restored.store.x, saved.store.x)
        assert np.array_equal(restored.store.genes, saved.store.genes)
        assert restored.plants.total_nutrients() == pytest.approx(saved.plants.total_nutrients())

    def test_the_generator_resumes_rather_than_replaying(self, tmp_path):
        """Without the generator's state a reloaded world restarts from the seed and repeats the
        draws it has already used — permitted by §2.2, which promises no determinism, but a
        repetition nobody asked for that compounds with every reload."""
        saved = run(80)
        restored = world()
        load(restored, save(saved, tmp_path / "w.npz"))

        assert restored.rng.bit_generator.state == saved.rng.bit_generator.state
        assert restored.rng.random() == saved.rng.random()

    def test_a_freshly_built_world_would_not_have_matched(self, tmp_path):
        """The control. Without it the test above could be passing because two demo worlds of the
        same seed agree anyway, which would make the whole suite vacuous."""
        saved = run(80)
        untouched = world()

        load(world(), save(saved, tmp_path / "w.npz"))

        assert not np.array_equal(untouched.store.x, saved.store.x)


class TestASnapshotBelongsToOneWorld:
    def test_a_config_with_different_rules_is_refused(self, tmp_path):
        """§2.8: a world's populations *are* an equilibrium reached under one rule set, so loading
        state into different rules collapses it for reasons invisible to the player. Refused rather
        than repaired, because there is no repair."""
        saved = run(20)
        path = save(saved, tmp_path / "w.npz")

        config = demo_world_config(FOUNDERS, SEED)
        cheaper = build_world(
            dataclasses.replace(
                config,
                metabolism=dataclasses.replace(config.metabolism, basal_rate=0.01),
            ),
            seed=SEED,
        )

        with pytest.raises(SnapshotError, match="different world config"):
            load(cheaper, path)

    def test_the_same_config_fingerprints_the_same_twice(self):
        """A fingerprint that varied between processes would refuse every load, which is a worse
        failure than the one it exists to prevent."""
        assert fingerprint(demo_world_config(FOUNDERS, SEED)) == fingerprint(
            demo_world_config(FOUNDERS, SEED)
        )

    def test_a_changed_rule_changes_the_fingerprint(self):
        config = demo_world_config(FOUNDERS, SEED)
        moved = dataclasses.replace(
            config, movement=dataclasses.replace(config.movement, climb_cost=99.0)
        )

        assert fingerprint(config) != fingerprint(moved)

    def test_a_gene_vocabulary_change_changes_the_fingerprint(self):
        """The vocabulary is the one part of a config a snapshot's arrays are *shaped* by, so a
        mismatch there is not merely a rules difference — the gene matrix would be the wrong
        width."""
        config = demo_world_config(FOUNDERS, SEED)
        recosted = dataclasses.replace(
            config, genes=(dataclasses.replace(config.genes[0], cost=0.5),) + config.genes[1:]
        )

        assert fingerprint(config) != fingerprint(recosted)

    def test_an_unknown_schema_version_is_refused(self, tmp_path):
        """There is one version and therefore no migration (§8.2), which makes the refusal the
        whole of the forward-compatibility story — and the first real migration a change here
        rather than an archaeology exercise."""
        saved = run(10)
        path = save(saved, tmp_path / "w.npz")
        _rewrite_meta(path, schema_version=SCHEMA_VERSION + 1)

        with pytest.raises(SnapshotError, match="no migration"):
            load(world(), path)

    def test_a_free_list_that_disagrees_with_the_row_ids_is_refused(self, tmp_path):
        """A corrupt snapshot whose two accounts of occupancy differ would otherwise surface as a
        newborn overwriting a live entity, ticks later and somewhere else (§8.7)."""
        saved = run(40)
        path = save(saved, tmp_path / "w.npz")
        contents = dict(np.load(path, allow_pickle=False))
        contents["free_rows"] = np.append(contents["free_rows"], 0)
        np.savez(path, **contents)

        with pytest.raises(ValueError, match="free list"):
            load(world(), path)


def _cull(built, count=10):
    """A cull of the first living animal's species, small enough never to be refused."""
    species = int(built.store.species_id[built.store.alive][0])
    return Cull(built.ecology, built.store, species_id=species, count=count, survivors=5)


class TestTheLedgerSurvivesAReload:
    """The intervention ledger is the one place the player is accountable (#26), so a world that
    continued bit for bit everywhere else and reset its budget would let a costly cull be repeated
    for free."""

    def test_a_spent_balance_stays_spent(self, tmp_path):
        saved = world()
        saved.loop.interventions.request(_cull(saved))
        saved.loop.advance(5)
        assert saved.loop.interventions.balance < DEMO_BUDGET, "the cull should have been charged"

        restored = world()
        load(restored, save(saved, tmp_path / "w.npz"))

        assert restored.loop.interventions.balance == saved.loop.interventions.balance

    def test_the_history_remembers_what_was_done(self, tmp_path):
        """Refusals too: a player back from an absence needs to know what did *not* happen (§2.7)."""
        saved = world()
        saved.loop.interventions.request(_cull(saved))
        saved.loop.interventions.request(_cull(saved, count=10_000))
        saved.loop.advance(5)
        assert [record.applied for record in saved.loop.interventions.history] == [True, False]

        restored = world()
        load(restored, save(saved, tmp_path / "w.npz"))

        assert restored.loop.interventions.history == saved.loop.interventions.history

    def test_a_request_that_had_not_reached_a_boundary_is_dropped(self, tmp_path):
        """Firing it on reload would act on a decision the player made about a world that then
        stopped — the failure #26 refuses retries for."""
        saved = world()
        saved.loop.interventions.request(_cull(saved))

        restored = world()
        restored.loop.interventions.request(_cull(restored))
        load(restored, save(saved, tmp_path / "w.npz"))
        restored.loop.advance(1)

        assert restored.loop.interventions.history == []
        assert restored.loop.interventions.balance == DEMO_BUDGET

    def test_a_ledger_cannot_be_loaded_into_a_world_without_one(self, tmp_path):
        """It would be silently dropped, and the next save would lose it for good (§8.7)."""
        saved = run(10)
        unledgered = world()
        unledgered.loop.interventions = None

        with pytest.raises(SnapshotError, match="intervention ledger"):
            load(unledgered, save(saved, tmp_path / "w.npz"))

    def test_a_world_with_a_ledger_cannot_load_a_snapshot_without_one(self, tmp_path):
        """It would keep whatever balance it was built with, which is a free budget."""
        saved = run(10)
        saved.loop.interventions = None

        with pytest.raises(SnapshotError, match="intervention ledger"):
            load(world(), save(saved, tmp_path / "w.npz"))


# Half a cadence past a sample, so a reload lands between two, and late enough that the founders
# have begun to breed and die: on this seed the first conception is recorded at tick 80.
SAVED_AT = 5 * DEMO_METRICS.every_n_ticks + DEMO_METRICS.every_n_ticks // 2


class TestTheMetricHistorySurvivesAReload:
    """Scenario goals and ecosystem-value income both read the recorded series (#292, #26), and it
    cannot be regenerated: the simulation is non-deterministic (§2.2) and a series is recorded,
    never recomputed (§4)."""

    def test_the_series_continues_exactly_as_an_unsaved_one(self, tmp_path):
        """Saved off-cadence, so the first sample after the reload has to land on the cadence and
        difference its conceptions and deaths against the reading before the save."""
        cadence = DEMO_METRICS.every_n_ticks
        saved = run(SAVED_AT)
        restored = world()
        load(restored, save(saved, tmp_path / "w.npz"))
        assert restored.loop.metrics.samples == saved.loop.metrics.samples

        saved.loop.advance(4 * cadence)
        restored.loop.advance(4 * cadence)

        assert restored.loop.metrics.samples == saved.loop.metrics.samples
        assert [sample.tick for sample in restored.loop.metrics.samples] == list(
            range(cadence, SAVED_AT + 4 * cadence + 1, cadence)
        )

    def test_the_first_sample_after_a_reload_reports_real_flows(self, tmp_path):
        """The control for the test above: a recorder that forgot its previous reading reports
        zero conceptions and deaths, which reads as a world in which nothing happened."""
        saved = run(SAVED_AT)
        restored = world()
        load(restored, save(saved, tmp_path / "w.npz"))

        to_next_sample = DEMO_METRICS.every_n_ticks - SAVED_AT % DEMO_METRICS.every_n_ticks
        saved.loop.advance(to_next_sample)
        restored.loop.advance(to_next_sample)

        first = restored.loop.metrics.samples[-1]
        assert first.conceptions + first.deaths > 0, "nothing happened to difference against"
        assert (first.conceptions, first.deaths) == (
            saved.loop.metrics.samples[-1].conceptions,
            saved.loop.metrics.samples[-1].deaths,
        )

    def test_a_history_cannot_be_loaded_into_a_world_that_records_none(self, tmp_path):
        saved = run(DEMO_METRICS.every_n_ticks)
        unrecorded = world()
        unrecorded.loop.metrics = None

        with pytest.raises(SnapshotError, match="metric history"):
            load(unrecorded, save(saved, tmp_path / "w.npz"))

    def test_a_recording_world_cannot_load_a_snapshot_without_a_history(self, tmp_path):
        """Its series would restart empty mid-world, and its first flows would read zero."""
        saved = run(DEMO_METRICS.every_n_ticks)
        saved.loop.metrics = None

        with pytest.raises(SnapshotError, match="metric history"):
            load(world(), save(saved, tmp_path / "w.npz"))

    def test_a_different_recording_policy_is_refused(self, tmp_path):
        """A series sampled every 20 ticks continued every 10 is two quantities on one axis — the
        reason `MetricsConfig` discards rather than downsamples."""
        saved = run(DEMO_METRICS.every_n_ticks)
        restored = world()
        restored.loop.metrics = MetricHistory(
            restored.store,
            restored.genetics,
            restored.plants,
            restored.genes.vocabulary,
            dataclasses.replace(DEMO_METRICS, every_n_ticks=DEMO_METRICS.every_n_ticks // 2),
        )

        with pytest.raises(SnapshotError, match="recording policy"):
            load(restored, save(saved, tmp_path / "w.npz"))


def _rewrite_meta(path: Path, **changes) -> None:
    """Rewrite a snapshot's metadata in place — the only way to stage a foreign one, since this
    build cannot write a version it does not have."""
    contents = dict(np.load(path, allow_pickle=False))
    meta = json.loads(str(contents["meta"]))
    meta.update(changes)
    contents["meta"] = np.array(json.dumps(meta))
    np.savez(path, **contents)
