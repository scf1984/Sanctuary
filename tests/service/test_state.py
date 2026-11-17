"""Founding, restarting and stopping a served world (#36): a stop and a restart lose nothing."""

import json

import pytest

import api
from service import close_world, open_world


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


class TestRestarting:
    def test_a_stop_and_restart_loses_no_state(self, tmp_path):
        """The done-when. A world stopped at tick 60 and reopened is the world it was."""
        state = tmp_path / "w"
        session, _ = open_world(state, now=0.0, ticks_per_second=1.0, world="demo", seed=4, founders=40)
        session.advance(60)
        session.request(api.CullRequest(species_id=0, count=2, survivors=10))
        session.advance(1)
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
