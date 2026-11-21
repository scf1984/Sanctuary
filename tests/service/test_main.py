"""`python -m service` as a lifecycle: found, serve, drain, snapshot, and restart (#36).

`run` is driven with an event standing in for the signal, since only the main thread may install
a signal handler. The real process, signals and all, was run by hand for the PR.
"""

import threading
import time

import pytest

import api
from service.__main__ import parse, run


def serve(state, *extra, clock=time.time):
    """Start `run` on a thread. Returns the stop event, the thread, the announced lines and a
    remote session."""
    lines = []
    stopping = threading.Event()
    args = parse(["--state", str(state), "--port", "0", "--founders", "30", "--chunk", "5", *extra])
    thread = threading.Thread(
        target=lambda: lines.append(("saved", run(args, stopping, clock, lines.append))), daemon=True
    )
    thread.start()
    deadline = time.time() + 30.0
    while not any(isinstance(line, str) and line.startswith("serving") for line in lines):
        assert time.time() < deadline, "the service never announced itself"
        time.sleep(0.02)
    url = lines[0].split(" at ")[1].split(",")[0]
    return stopping, thread, lines, api.connect(url)


def test_it_founds_serves_and_snapshots_on_stop(tmp_path):
    stopping, thread, lines, remote = serve(tmp_path / "w")
    assert remote.describe().name == "demo"
    assert remote.frame().ids.shape == (30,)

    stopping.set()
    thread.join(timeout=30.0)

    assert (tmp_path / "w" / "snapshot.npz").exists()
    assert lines[-1][0] == "saved"


def test_a_restart_keeps_the_founding_and_pays_the_downtime(tmp_path):
    clock = {"now": 1_000.0}
    tick = lambda: clock["now"]  # noqa: E731 - a clock is a callable, and this one is set by hand
    stopping, thread, _lines, remote = serve(tmp_path / "w", clock=tick)
    stopping.set()
    thread.join(timeout=30.0)

    clock["now"] += 25.0
    stopping, thread, _lines, remote = serve(tmp_path / "w", clock=tick)
    deadline = time.time() + 30.0
    while remote.status().tick < 25 and time.time() < deadline:
        time.sleep(0.05)

    assert remote.status().tick == 25
    assert remote.describe().seed == 0
    stopping.set()
    thread.join(timeout=30.0)


def test_asking_an_existing_world_to_be_another_is_refused(tmp_path):
    stopping, thread, _lines, _remote = serve(tmp_path / "w")
    stopping.set()
    thread.join(timeout=30.0)

    args = parse(["--state", str(tmp_path / "w"), "--port", "0", "--seed", "9"])
    with pytest.raises(ValueError, match="not what was asked for"):
        run(args, threading.Event())


def test_a_pause_ends_when_the_service_stops_and_the_restart_runs_at_the_worlds_own_rate(tmp_path):
    """#307 end to end: paused while watched, the world does not age; stopped, everyone has left,
    so the downtime is owed at the world's own rate from the moment of the stop."""
    clock = {"now": 1_000.0}
    tick = lambda: clock["now"]  # noqa: E731
    stopping, thread, _lines, remote = serve(tmp_path / "w", clock=tick)
    events = remote.events()
    threading.Thread(target=lambda: next(events, None), daemon=True).start()
    deadline = time.time() + 10.0
    while True:
        try:
            remote.set_pace(0.0)
            break
        except api.Refused:
            assert time.time() < deadline, "the stream was never counted as watching"
            time.sleep(0.02)
    clock["now"] += 100.0
    stopping.set()
    thread.join(timeout=30.0)

    clock["now"] += 25.0
    stopping, thread, _lines, remote = serve(tmp_path / "w", clock=tick)
    deadline = time.time() + 30.0
    while remote.status().tick < 25 and time.time() < deadline:
        time.sleep(0.05)

    assert remote.status().tick == 25
    assert remote.pace().ticks_per_second == remote.pace().own_ticks_per_second
    stopping.set()
    thread.join(timeout=30.0)
