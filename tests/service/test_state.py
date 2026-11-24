"""Founding, restarting and stopping a served world (#36): a stop and a restart lose nothing."""

import json

import pytest

import api
from scheduler import Anchor, RateChange, Timeline
from service import close_world, keep_timeline, open_timeline, open_world


def comparable(session):
    """Everything a client reads, except `potential_growth`. That grid is derived from the tick's
    weather, and straight after a load it still shows the weather the world was built at, until
    the next tick re-derives it (#300). It changes no outcome. The comparison after further ticks
    includes it, because by then it has been re-derived."""
    info = session.describe()
    return (
        session.status(),
        session.frame(),
        session.ledger(),
        session.history(),
        session.species(),
        tuple(session.field(layer) for layer in info.field_layers if layer != "potential_growth"),
    )


def everything(session):
    return comparable(session) + (session.field("potential_growth"),)


class TestFounding:
    def test_a_new_state_directory_founds_the_world_asked_for(self, tmp_path):
        session, founding = open_world(
            tmp_path / "w", now=500.0, ticks_per_second=1.0, world="demo", seed=4, founders=30
        )

        assert founding.world == "demo" and founding.seed == 4 and founding.founders == 30
        assert founding.anchor.wall_seconds == 500.0
        assert founding.anchor.tick == 0
        assert session.status().tick == 0

    def test_founding_needs_to_be_told_what_to_found(self, tmp_path):
        with pytest.raises(ValueError, match="needs a world, seed and founders"):
            open_world(tmp_path / "w", now=0.0, ticks_per_second=1.0)

    def test_the_founding_record_is_plain_json(self, tmp_path):
        open_world(tmp_path, now=1.5, ticks_per_second=2.0, world="demo", seed=1, founders=10)

        record = json.loads((tmp_path / "world.json").read_text(encoding="utf-8"))

        assert record["anchor"] == {"wall_seconds": 1.5, "tick": 0, "ticks_per_second": 2.0}


class TestScenarios:
    """#317: a world's goal is chosen when it is founded, and a restart plays the same one."""

    def test_the_founding_records_the_scenario_and_a_restart_plays_it(self, tmp_path):
        session, founding = open_world(
            tmp_path, now=0.0, ticks_per_second=1.0, world="demo", seed=1, founders=20,
            scenario="first-split",
        )
        close_world(session, tmp_path)

        reopened, again = open_world(tmp_path, now=1.0, ticks_per_second=1.0)

        assert founding.scenario == again.scenario == "first-split"
        assert reopened.scenario().name == "first-split"
        assert json.loads((tmp_path / "world.json").read_text(encoding="utf-8"))["scenario"] == (
            "first-split"
        )

    def test_a_sandbox_world_has_no_scenario(self, tmp_path):
        session, founding = open_world(
            tmp_path, now=0.0, ticks_per_second=1.0, world="demo", seed=1, founders=20
        )

        assert founding.scenario is None
        assert session.scenario() is None

    def test_a_founding_written_before_scenarios_reads_as_a_sandbox(self, tmp_path):
        open_world(tmp_path, now=0.0, ticks_per_second=1.0, world="demo", seed=1, founders=20)
        path = tmp_path / "world.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        del record["scenario"]
        path.write_text(json.dumps(record), encoding="utf-8")

        _session, founding = open_world(tmp_path, now=1.0, ticks_per_second=1.0)

        assert founding.scenario is None

    def test_a_sandbox_cannot_be_reopened_as_a_scenario(self, tmp_path):
        open_world(tmp_path, now=0.0, ticks_per_second=1.0, world="demo", seed=1, founders=20)

        with pytest.raises(ValueError, match="not what was asked for"):
            open_world(tmp_path, now=1.0, ticks_per_second=1.0, scenario="first-split")


class TestRestarting:
    def test_a_stop_and_restart_loses_no_state(self, tmp_path):
        """The done-when. A world stopped at tick 60 and reopened is the world it was."""
        state = tmp_path / "w"
        session, _ = open_world(state, now=0.0, ticks_per_second=1.0, world="demo", seed=4, founders=40)
        session.advance(60)
        session.request(api.CullRequest(species_id=0, count=2))
        session.advance(1)
        # #314: a species' name is the player's, and a restart must not lose it.
        session.name(api.SpeciesNameRequest(species_id=0, name="Grazers"))
        before = comparable(session)
        close_world(session, state)

        reopened, founding = open_world(state, now=61.0, ticks_per_second=1.0)

        assert comparable(reopened) == before
        assert founding.anchor.wall_seconds == 0.0

    def test_the_downtime_is_owed_after_a_restart(self, tmp_path):
        """§2.4: the world kept running while nobody served it."""
        from scheduler import CatchUp

        state = tmp_path / "w"
        session, founding = open_world(state, now=0.0, ticks_per_second=1.0, world="demo", seed=4, founders=20)
        session.advance(30)
        close_world(session, state)

        reopened, founding = open_world(state, now=100.0, ticks_per_second=1.0)

        assert CatchUp(reopened, founding.anchor, chunk=50).owed(100.0) == 70

    def test_the_restored_world_continues_as_if_it_had_never_stopped(self, tmp_path):
        """Restoring continues bit for bit (§3.2), so stopping midway changes nothing."""
        state = tmp_path / "w"
        session, _ = open_world(state, now=0.0, ticks_per_second=1.0, world="demo", seed=4, founders=40)
        session.advance(40)
        close_world(session, state)
        reopened, _ = open_world(state, now=40.0, ticks_per_second=1.0)
        reopened.advance(40)

        never_stopped = api.create("demo", seed=4, founders=40)
        never_stopped.advance(80)

        assert everything(reopened) == everything(never_stopped)

    def test_a_state_directory_refuses_to_become_a_different_world(self, tmp_path):
        open_world(tmp_path, now=0.0, ticks_per_second=1.0, world="demo", seed=4, founders=20)

        with pytest.raises(ValueError, match="not what was asked for"):
            open_world(tmp_path, now=1.0, ticks_per_second=1.0, seed=5)

    def test_a_world_founded_but_never_saved_restarts_from_its_founding(self, tmp_path):
        """A crash before the first shutdown leaves no snapshot. The founding still says what the
        world is and when it began, so it is rebuilt at tick 0 and owes everything since."""
        open_world(tmp_path, now=0.0, ticks_per_second=1.0, world="demo", seed=4, founders=20)

        reopened, founding = open_world(tmp_path, now=10.0, ticks_per_second=1.0)

        assert reopened.status().tick == 0
        assert founding.anchor.due(10.0) == 10


class TestSnapshotReplacement:
    def test_a_save_leaves_exactly_one_snapshot(self, tmp_path):
        session, _ = open_world(tmp_path, now=0.0, ticks_per_second=1.0, world="demo", seed=4, founders=20)
        close_world(session, tmp_path)
        session.advance(5)
        close_world(session, tmp_path)

        assert sorted(p.name for p in tmp_path.iterdir()) == ["snapshot.npz", "world.json"]


class TestTimeline:
    """#307: the state directory keeps the world's history of time, and a restart is unwatched."""

    ANCHOR = Anchor(wall_seconds=0.0, tick=0, ticks_per_second=1.0)

    def test_a_world_never_retimed_has_only_its_anchor(self, tmp_path):
        timeline = open_timeline(tmp_path, self.ANCHOR, now=50.0)

        assert timeline == Timeline(self.ANCHOR)
        assert not (tmp_path / "timeline.json").exists()

    def test_a_kept_timeline_reopens_with_its_history(self, tmp_path):
        """Its age at every moment it lived through is still there to reconstruct."""
        kept = Timeline(self.ANCHOR).at_rate(10.0, 0.0).at_own_rate(50.0)
        keep_timeline(tmp_path, kept)

        reopened = open_timeline(tmp_path, self.ANCHOR, now=100.0)

        assert reopened == kept
        assert reopened.due(100.0) == 60

    def test_a_restart_resumes_the_worlds_own_rate(self, tmp_path):
        """A hard kill mid-pause leaves the pause on record. Nobody watches a world that has just
        started, so it resumes, at the restart since when the player left is not known."""
        keep_timeline(tmp_path, Timeline(self.ANCHOR).at_rate(10.0, 0.0))

        reopened = open_timeline(tmp_path, self.ANCHOR, now=80.0)

        assert reopened.ticks_per_second == 1.0
        assert reopened.changes[-1] == RateChange(wall_seconds=80.0, tick=10, ticks_per_second=1.0)
        assert open_timeline(tmp_path, self.ANCHOR, now=90.0) == reopened

    def test_the_record_is_plain_json_without_the_anchor(self, tmp_path):
        """The anchor lives in `world.json` alone, so the two files cannot disagree about it."""
        keep_timeline(tmp_path, Timeline(self.ANCHOR).at_rate(10.0, 0.0))

        record = json.loads((tmp_path / "timeline.json").read_text(encoding="utf-8"))

        assert record == {"changes": [{"wall_seconds": 10.0, "tick": 10, "ticks_per_second": 0.0}]}
