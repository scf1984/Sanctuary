"""The first scenario's goal, read off a recorded history (#317).

Written against the decisions on #317 before the panel that shows them: a goal is a check against
what was recorded, never a forecast (§2.3), so every case here is a history arranged by hand and a
reading asserted over it.
"""

import numpy as np
import pytest

from core.genetics.speciation import split
from core.selection import Selection
from metrics import MetricHistory, MetricsConfig
from metrics.goals import FirstSplit
from persistence import load, save
from worlds.demo import build_demo_world

CADENCE = MetricsConfig(every_n_ticks=10, history_limit=1_000)
GENERATION = 30
GOAL = FirstSplit(generation_ticks=GENERATION)


def world(seed=1, n_entities=60):
    built = build_demo_world(seed=seed, n_entities=n_entities)
    built.loop.metrics = MetricHistory(
        built.store,
        built.genetics,
        built.plants,
        built.speciation,
        built.genes.vocabulary,
        CADENCE,
    )
    return built


def record(built, *ticks):
    for tick in ticks:
        assert built.loop.metrics.record_if_due(tick) is not None


def split_off(built, count, tick):
    rows = np.flatnonzero(built.store.alive & (built.store.age >= 0))[:count]
    return split(
        built.genetics,
        built.speciation.lineage,
        Selection.from_indices(rows, built.store.capacity),
        tick=tick,
    )


def kill(built, species):
    members = built.store.alive & (built.store.species_id == species)
    built.store.release(built.store.row_ids()[members])


def assess(built):
    return GOAL.assess(built.loop.metrics, built.speciation.lineage)


class TestBeforeTheSplit:
    def test_it_is_open_with_the_last_measured_divergence_as_progress(self):
        built = world()
        built.speciation.divergence = {0: 0.42}
        record(built, 10)

        reading = assess(built)

        assert reading.outcome == "open"
        assert reading.progress == pytest.approx(0.42)
        assert reading.parent is None and reading.daughter is None and reading.due_tick is None

    def test_progress_reads_what_was_recorded_not_what_the_check_holds_now(self):
        """A goal reads the recorded series, so it answers the same after a reload as live."""
        built = world()
        built.speciation.divergence = {0: 0.3}
        record(built, 10)
        built.speciation.divergence = {0: 0.9}

        assert assess(built).progress == pytest.approx(0.3)

    def test_an_empty_history_is_open_and_unmeasured(self):
        assert assess(world()).outcome == "open"
        assert assess(world()).progress is None

    def test_a_check_that_measured_nothing_leaves_the_last_measurement_standing(self):
        """The check skips a species whose groups are under `min_founders`, which reads 0 in the
        sample. Progress shows the last measurement rather than dropping to nothing."""
        built = world()
        built.speciation.divergence = {0: 0.4}
        record(built, 10)
        built.speciation.divergence = {}
        record(built, 20, 30)

        assert assess(built).progress == pytest.approx(0.4)

    def test_it_is_lost_when_the_founding_species_dies_out(self):
        built = world()
        record(built, 10)
        kill(built, 0)
        record(built, 20)

        reading = assess(built)

        assert reading.outcome == "lost"
        assert reading.resolved_tick == 20
        assert "founding species" in reading.reason


class TestAfterTheSplit:
    def test_it_waits_a_generation_from_the_split(self):
        built = world()
        daughter = split_off(built, 20, tick=40)
        record(built, 40, 50, 60)

        reading = assess(built)

        assert reading.outcome == "open"
        assert reading.progress == 1.0
        assert (reading.parent, reading.daughter) == (0, daughter)
        assert reading.split_tick == 40
        assert reading.due_tick == 70
        assert reading.resolved_tick is None

    def test_it_is_met_at_the_sample_a_generation_on_with_both_alive(self):
        built = world()
        split_off(built, 20, tick=40)
        record(built, 40, 50, 60, 70)

        reading = assess(built)

        assert reading.outcome == "met"
        assert reading.resolved_tick == 70

    def test_the_due_sample_is_the_first_one_a_generation_on(self):
        """The recorder only looks on its cadence, so a generation ending between two samples is
        judged at the next one rather than guessed at."""
        built = world()
        split_off(built, 20, tick=45)
        record(built, 50, 60, 70)

        assert assess(built).due_tick == 80
        assert assess(built).outcome == "open"

    @pytest.mark.parametrize("which", ["daughter", "parent"])
    def test_it_is_lost_when_either_dies_within_the_generation(self, which):
        built = world()
        daughter = split_off(built, 20, tick=40)
        record(built, 40, 50)
        kill(built, daughter if which == "daughter" else 0)
        record(built, 60, 70, 80)

        reading = assess(built)

        assert reading.outcome == "lost"
        assert reading.resolved_tick == 60
        assert which in reading.reason

    def test_dying_on_the_due_sample_itself_is_a_loss(self):
        """At the due sample the recorder found nobody, so both were not alive a generation on."""
        built = world()
        daughter = split_off(built, 20, tick=40)
        record(built, 40, 50, 60)
        kill(built, daughter)
        record(built, 70)

        assert assess(built).outcome == "lost"

    def test_a_met_goal_stays_met_when_a_species_dies_later(self):
        """Met is final (#317, decision 5): the world runs on as a sandbox and can do what it
        likes, but the player did what was asked."""
        built = world()
        daughter = split_off(built, 20, tick=40)
        record(built, 40, 50, 60, 70, 80)
        kill(built, daughter)
        record(built, 90)

        reading = assess(built)

        assert reading.outcome == "met"
        assert reading.resolved_tick == 70

    def test_a_loss_outlives_the_window_that_saw_it(self):
        """The window forgets (`history_limit`), and a goal must not: a daughter that died within
        a generation stays a loss a week of samples later."""
        built = build_demo_world(seed=1, n_entities=60)
        built.loop.metrics = MetricHistory(
            built.store,
            built.genetics,
            built.plants,
            built.speciation,
            built.genes.vocabulary,
            MetricsConfig(every_n_ticks=10, history_limit=3),
        )
        daughter = split_off(built, 20, tick=40)
        record(built, 40, 50)
        kill(built, daughter)
        record(built, *range(60, 200, 10))

        assert built.loop.metrics.samples[0].tick > 70
        assert assess(built).outcome == "lost"


class TestExtinctions:
    def test_a_gestating_young_keeps_its_species_from_extinction(self):
        """It is born whatever happens to its parents, so its species is not over (§2.7)."""
        built = world()
        store = built.store
        born = np.flatnonzero(store.alive & (store.age >= 0))
        store.release(store.row_ids()[born[1:]])
        store.age[born[0]] = -5

        record(built, 10)

        assert built.loop.metrics.extinct_at(0) is None
        assert built.loop.metrics.sample(tick=10).species_living == (0,)

    def test_the_first_sample_that_found_nobody_is_when(self):
        built = world()
        daughter = split_off(built, 20, tick=0)
        record(built, 10)
        kill(built, daughter)
        record(built, 20, 30)

        assert built.loop.metrics.extinct_at(daughter) == 20
        assert built.loop.metrics.extinct_at(0) is None


class TestTheGoalSurvivesAReload:
    def test_a_reloaded_world_reads_the_same_outcome_and_progress(self, tmp_path):
        """Nothing about the goal is saved, because it is a reading of what is (§3.2)."""
        built = build_demo_world(seed=1, n_entities=60)
        built.loop.advance(20)
        daughter = split_off(built, 20, tick=20)
        built.loop.advance(20)
        kill(built, daughter)
        built.loop.advance(20)
        before = GOAL.assess(built.loop.metrics, built.speciation.lineage)

        restored = build_demo_world(seed=1, n_entities=60)
        load(restored, save(built, tmp_path / "w.npz", names={}))

        assert GOAL.assess(restored.loop.metrics, restored.speciation.lineage) == before
        assert before.outcome == "lost"


def test_a_goal_needs_a_generation_to_wait():
    with pytest.raises(ValueError, match="generation_ticks"):
        FirstSplit(generation_ticks=0)
