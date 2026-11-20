"""The thin service (#36): one world, served over HTTP, ticking with nobody attached.

The tests drive time themselves. `Service.wake(now)` is the ticker's single step, so most tests
call it with a chosen `now` instead of waiting on a real clock. One test does run the ticker
thread for real, since "advances with no client connected" is a claim about that thread.
"""

import math
import threading
import time

import numpy as np
import pytest

import api
from api import CullRequest, Refused
from scheduler import Anchor, Stage, Timeline, WakeSchedule
from service import Service
from tests.api.test_values import assert_plain

SCHEDULE = WakeSchedule(watched_interval=1.0, absent=(Stage(math.inf, 3_600.0),))
FASTEST = 6.0


def make_service(session, anchor, schedule, clock, chunk, kept=None):
    """A service whose recorded timelines land in `kept`, where a test can read them."""
    return Service(
        session,
        Timeline(anchor),
        schedule,
        clock,
        chunk,
        fastest_ticks_per_second=FASTEST,
        keep=(kept if kept is not None else []).append,
    )


class Clock:
    def __init__(self, now=1_000.0):
        self.now = now

    def __call__(self):
        return self.now


@pytest.fixture
def kept():
    """Every timeline the served world's service recorded, in order."""
    return []


@pytest.fixture
def served(kept):
    """A service on a small demo world, an HTTP server on a free port, and a remote session."""
    clock = Clock()
    session = api.create("demo", seed=3, founders=40)
    service = make_service(session, Anchor(clock.now, 0, 1.0), SCHEDULE, clock, 10, kept)
    server = service.http_server("127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    remote = api.connect(f"http://127.0.0.1:{server.server_port}")
    yield service, session, remote, clock
    service.stop()
    server.shutdown()
    server.server_close()


class TestTheWorldTicksOnItsOwn:
    def test_a_wake_pays_what_real_time_owes_with_no_client_attached(self, served):
        service, session, _remote, clock = served
        assert not service.watched

        clock.now += 37.0
        assert service.wake(clock.now) == 37

        assert session.status().tick == 37

    def test_the_ticker_thread_advances_the_world_with_nobody_connected(self):
        """The done-when, on a real thread and a real clock: no client, and the world moves."""
        session = api.create("demo", seed=3, founders=20)
        fast = WakeSchedule(watched_interval=0.05, absent=(Stage(math.inf, 0.05),))
        service = make_service(session, Anchor(time.time(), 0, 40.0), fast, time.time, 5)
        service.start()
        deadline = time.time() + 10.0
        while session.status().tick < 10 and time.time() < deadline:
            time.sleep(0.05)
        service.stop()

        assert session.status().tick >= 10

    def test_stopping_waits_for_a_chunk_boundary_not_for_the_whole_catch_up(self):
        session = api.create("demo", seed=3, founders=20)
        clock = Clock()
        service = make_service(session, Anchor(clock.now, 0, 1.0), SCHEDULE, clock, 5)
        service._stop.set()

        assert service.wake(clock.now + 10_000.0) == 5


class TestTheProtocol:
    def test_a_remote_frame_is_the_local_frame(self, served):
        service, session, remote, clock = served
        clock.now += 12.0
        service.wake(clock.now)

        assert remote.frame() == session.frame()
        assert remote.status() == session.status()
        assert remote.describe() == session.describe()

    def test_every_remote_read_is_a_plain_value(self, served):
        """#35's rule holds on the far side of the wire too."""
        _service, _session, remote, _clock = served
        info = remote.describe()
        some_id = int(remote.frame().ids.to_numpy()[0])

        for value in (
            remote.status(), info, remote.terrain(), remote.frame(), remote.barriers(),
            remote.history(), remote.inspect(some_id), remote.inspect(-1), remote.ledger(),
            remote.pace(),
            *(remote.condition(m) for m in info.conditions),
            *(remote.field(f) for f in info.field_layers),
        ):
            assert_plain(value)

    def test_an_intervention_requested_remotely_lands_at_the_next_boundary(self, served):
        service, _session, remote, clock = served
        cull = CullRequest(species_id=0, count=2, survivors=10)
        quote = remote.quote(cull)

        remote.request(cull)
        clock.now += 1.0
        service.wake(clock.now)

        assert remote.ledger().records[-1].cost == quote.cost

    def test_a_served_world_cannot_be_driven_by_a_client(self, served):
        """One execution path (#32): the schedule advances a served world, and nothing else."""
        _service, _session, remote, _clock = served

        with pytest.raises(Refused, match="own schedule"):
            remote.advance(10)
        with pytest.raises(Refused):
            remote.save("anywhere.npz")
        with pytest.raises(Refused):
            remote.load("anywhere.npz")

    def test_a_remote_mistake_raises_what_a_local_one_would(self, served):
        _service, _session, remote, _clock = served

        with pytest.raises(ValueError, match="unknown condition"):
            remote.condition("ennui")
        with pytest.raises(ValueError, match="unknown field"):
            remote.field("nutrient_soup")

    def test_a_dead_animal_is_nobody_over_the_wire_too(self, served):
        _service, session, remote, _clock = served
        some_id = int(session.frame().ids.to_numpy()[0])
        session._world.store.release(np.array([some_id], dtype=np.int64))

        assert remote.inspect(some_id) is None

    def test_any_request_counts_as_the_player_looking(self, served):
        """A phone widget polling once a day resets the absence clock without holding a stream."""
        service, _session, remote, clock = served
        clock.now += 500.0

        remote.status()

        assert service.last_seen == clock.now


class TestWatching:
    def test_an_open_stream_is_what_watching_means(self, served):
        service, _session, remote, clock = served
        assert service.next_interval() == 3_600.0
        events = remote.events()
        received = []

        reader = threading.Thread(target=lambda: received.append(next(events)), daemon=True)
        reader.start()
        deadline = time.time() + 5.0
        while not service.watched and time.time() < deadline:
            time.sleep(0.01)
        assert service.watched
        assert service.next_interval() == 1.0

        clock.now += 3.0
        service.wake(clock.now)
        reader.join(timeout=5.0)

        assert received and received[0].status.tick == 3
        assert received[0].owed == 3

    def test_the_stream_reports_every_chunk(self, served):
        service, _session, remote, clock = served
        events = remote.events()
        received = []

        def read_three():
            for _ in range(3):
                received.append(next(events))

        reader = threading.Thread(target=read_three, daemon=True)
        reader.start()
        deadline = time.time() + 5.0
        while not service.watched and time.time() < deadline:
            time.sleep(0.01)
        clock.now += 25.0
        service.wake(clock.now)
        reader.join(timeout=5.0)

        assert [p.done for p in received] == [10, 20, 25]

    def test_closing_the_last_stream_starts_the_absence_clock(self, served):
        service, _session, _remote, clock = served
        watcher = service.attach()
        clock.now += 40.0

        service.detach(watcher)

        assert not service.watched
        assert service.last_seen == clock.now


def watch(service, remote):
    """Open a real event stream on a thread, returning once the service counts it as watching. It
    stays open until the service stops."""
    events = remote.events()
    reader = threading.Thread(target=lambda: next(events, None), daemon=True)
    reader.start()
    deadline = time.time() + 5.0
    while not service.watched and time.time() < deadline:
        time.sleep(0.01)
    assert service.watched


class TestPace:
    """#307: a watching player pauses and fast-forwards, and leaving resumes the world's own rate."""

    def test_the_pace_says_the_rate_and_the_range_offered(self, served):
        _service, _session, remote, _clock = served

        assert remote.pace() == api.Pace(
            ticks_per_second=1.0, own_ticks_per_second=1.0, fastest_ticks_per_second=FASTEST
        )

    def test_nobody_can_change_the_pace_of_a_world_they_are_not_watching(self, served):
        """Decision 8's follow-up: a pause that outlived the player would let them switch off the
        world running while they are away."""
        service, _session, remote, _clock = served

        with pytest.raises(Refused, match="watching"):
            remote.set_pace(0.0)
        assert service.timeline.changes == ()

    def test_a_watching_player_pauses_the_world(self, served):
        service, session, remote, clock = served
        watch(service, remote)
        clock.now += 5.0
        service.wake(clock.now)

        assert remote.set_pace(0.0).ticks_per_second == 0.0
        clock.now += 100.0
        service.wake(clock.now)

        assert session.status().tick == 5

    def test_a_watching_player_fast_forwards_the_world(self, served):
        service, session, remote, clock = served
        watch(service, remote)

        remote.set_pace(FASTEST)
        clock.now += 10.0
        service.wake(clock.now)

        assert session.status().tick == 60

    def test_no_rate_past_the_fastest_is_offered(self, served):
        service, _session, remote, _clock = served
        watch(service, remote)

        with pytest.raises(Refused, match="fastest"):
            remote.set_pace(FASTEST + 0.5)

    @pytest.mark.parametrize("rate", [-1.0, math.inf])
    def test_a_rate_that_is_not_one_is_a_mistake(self, served, rate):
        service, _session, remote, _clock = served
        watch(service, remote)

        with pytest.raises(ValueError, match="rate"):
            service.set_pace(rate)

    def test_only_a_pace_request_sets_the_pace(self, served):
        service, _session, remote, _clock = served
        watch(service, remote)

        with pytest.raises(TypeError, match="PaceRequest"):
            remote._post("/pace", CullRequest(species_id=0, count=1, survivors=1))

    def test_leaving_resumes_the_worlds_own_rate_at_that_moment(self, served):
        """A world paused and then closed keeps running, and owes nothing for the pause itself."""
        service, session, remote, clock = served
        watcher = service.attach()
        clock.now += 5.0
        remote.set_pace(0.0)
        clock.now += 300.0

        service.detach(watcher)
        clock.now += 20.0
        service.wake(clock.now)

        assert not service.watched
        assert service.timeline.ticks_per_second == 1.0
        assert service.timeline.changes[-1].wall_seconds == clock.now - 20.0
        assert session.status().tick == 25

    def test_every_change_is_recorded_before_it_takes_effect(self, served, kept):
        """The state directory must never hold a timeline older than the one the world runs by,
        or a crash would restart it owing a different number of ticks."""
        service, _session, remote, clock = served
        watcher = service.attach()
        clock.now += 1.0
        remote.set_pace(0.0)
        clock.now += 1.0
        remote.set_pace(3.0)
        clock.now += 1.0
        service.detach(watcher)

        assert [timeline.ticks_per_second for timeline in kept] == [0.0, 3.0, 1.0]
        assert kept[-1] == service.timeline

    def test_stopping_the_service_is_everyone_leaving(self, served, kept):
        """So the snapshot saved after a stop sits beside a timeline back at the world's own rate,
        whenever the stream threads get round to detaching."""
        service, _session, remote, _clock = served
        service.attach()
        remote.set_pace(0.0)

        service.stop()

        assert kept[-1].ticks_per_second == 1.0


class TestTheViewerOverTheProtocol:
    def test_one_frame_of_the_viewer_runs_against_a_served_world(self, served):
        """#36's done-when, "the viewer works over the protocol", as far as CI can reach: every
        per-frame call `app.py` makes, against a remote session. `app.py` itself imports pygame
        and is not collectable here (#110)."""
        from clients.viewer.render import (
            CONDITION_MODES,
            apply_water_overlay,
            barrier_segments,
            condition_colors,
            describe_entity,
            elevation_shading,
            field_overlay,
            live_positions,
            species_colors,
            world_to_screen,
        )

        service, _session, remote, clock = served
        info = remote.describe()
        terrain = remote.terrain()
        background = apply_water_overlay(elevation_shading(terrain), terrain.water_depth.to_numpy())
        previous = remote.frame()
        clock.now += 2.0
        service.wake(clock.now)
        current = remote.frame()

        x, y, _z = live_positions(previous, current, 0.5)
        px, _py = world_to_screen(x, y, info.width, info.height, 640, 480)
        for mode in CONDITION_MODES:
            colors = (
                species_colors(current.species.to_numpy())
                if mode == "species"
                else condition_colors(remote.condition(mode).values.to_numpy())
            )
            assert colors.shape == (current.ids.shape[0], 3)
        for layer in info.field_layers:
            assert field_overlay(background, remote.field(layer)).shape == background.shape
        barrier_segments(remote.barriers(), info.cell_size, info.width, info.height, 640, 480)
        some_id = int(current.ids.to_numpy()[0])
        assert describe_entity(some_id, remote.inspect(some_id))[0].startswith("entity")
        assert current.tick == previous.tick + 2
        assert px.shape == (current.ids.shape[0],)
