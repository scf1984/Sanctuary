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
from scheduler import Anchor, Stage, WakeSchedule
from service import Service
from tests.api.test_values import assert_plain

SCHEDULE = WakeSchedule(watched_interval=1.0, absent=(Stage(math.inf, 3_600.0),))


class Clock:
    def __init__(self, now=1_000.0):
        self.now = now

    def __call__(self):
        return self.now


@pytest.fixture
def served():
    """A service on a small demo world, an HTTP server on a free port, and a remote session."""
    clock = Clock()
    session = api.create("demo", seed=3, founders=40)
    service = Service(session, Anchor(clock.now, 0, 1.0), SCHEDULE, clock, chunk=10)
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
        service = Service(session, Anchor(time.time(), 0, 40.0), fast, time.time, chunk=5)
        service.start()
        deadline = time.time() + 10.0
        while session.status().tick < 10 and time.time() < deadline:
            time.sleep(0.05)
        service.stop()

        assert session.status().tick >= 10

    def test_stopping_waits_for_a_chunk_boundary_not_for_the_whole_catch_up(self):
        session = api.create("demo", seed=3, founders=20)
        clock = Clock()
        service = Service(session, Anchor(clock.now, 0, 1.0), SCHEDULE, clock, chunk=5)
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


@pytest.fixture
def client_served(tmp_path):
    """A service serving a stand-in client build: an index, a module, and a file the build would
    never emit. A secret sits beside the build, where a traversal would reach it."""
    build = tmp_path / "dist"
    (build / "assets").mkdir(parents=True)
    (build / "index.html").write_text("<!doctype html><title>Sanctuary</title>", encoding="utf-8")
    (build / "assets" / "main.js").write_text("export const x = 1;", encoding="utf-8")
    (build / "notes.txt").write_text("not part of a build", encoding="utf-8")
    (tmp_path / "world.json").write_text("{}", encoding="utf-8")
    clock = Clock()
    session = api.create("demo", seed=3, founders=20)
    service = Service(session, Anchor(clock.now, 0, 1.0), SCHEDULE, clock, chunk=10)
    server = service.http_server("127.0.0.1", 0, build)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    service.stop()
    server.shutdown()
    server.server_close()


def fetch(url):
    """(status, content type, body) for a GET, whether or not it succeeded."""
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(url) as response:
            return response.status, response.headers["Content-Type"], response.read()
    except urllib.error.HTTPError as failed:
        return failed.code, failed.headers["Content-Type"], failed.read()


class TestTheGameClientIsServedFromHere:
    """#292 decision 5: one process and one origin serve the game, so no route needs CORS."""

    def test_the_root_is_the_clients_index(self, client_served):
        status, kind, body = fetch(client_served + "/")
        assert status == 200
        assert kind.startswith("text/html")
        assert b"Sanctuary" in body

    def test_a_module_is_served_as_javascript(self, client_served):
        """Not `mimetypes`' answer: on Windows that can be text/plain, which a browser will not run."""
        status, kind, _ = fetch(client_served + "/assets/main.js")
        assert status == 200
        assert kind.startswith("text/javascript")

    def test_api_routes_still_answer(self, client_served):
        status, kind, body = fetch(client_served + "/status")
        assert status == 200
        assert kind == "application/json"
        assert b'"$type":"Status"' in body

    @pytest.mark.parametrize(
        "path", ["/../world.json", "/%2e%2e/world.json", "/assets/../../world.json"]
    )
    def test_nothing_outside_the_build_is_served(self, client_served, path):
        status, _, body = fetch(client_served + path)
        assert status == 404
        assert b"{}" != body

    def test_a_file_the_build_does_not_emit_is_not_served(self, client_served):
        assert fetch(client_served + "/notes.txt")[0] == 404

    def test_a_missing_file_is_404(self, client_served):
        assert fetch(client_served + "/assets/gone.js")[0] == 404

    def test_without_a_build_the_root_says_the_client_is_not_served(self, served):
        service, *_ = served
        server = service.http_server("127.0.0.1", 0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            status, _, body = fetch(f"http://127.0.0.1:{server.server_port}/")
        finally:
            server.shutdown()
            server.server_close()
        assert status == 404
        assert b"not served" in body
