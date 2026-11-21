"""One world, served over HTTP, ticking whether or not anybody is looking (#36, §3.1 stage 2).

**The world advances in exactly one place: the ticker thread, running #32's `CatchUp`.** There is
no advance endpoint and no second loop. Watching shortens the wake interval and changes nothing
else, which is gate 2 on #35 as the owner decided it.

**One lock, held per call and per catch-up chunk, never across a whole catch-up.** A request
arriving mid-catch-up waits at most one chunk and then reads a world sitting at a tick boundary,
which is the only place a world may be read (§2.3). Holding the lock across a long absence's
catch-up would freeze the protocol for minutes.

**What "watched" means here is an open event stream.** Every request also counts as the player
looking, so a phone widget polling once a day resets the absence clock without holding anything
open. That matters because the absence clock is what decays the schedule (§2.4).

**A watching player may pause and fast-forward, and leaving undoes it** (#307, #292's decision 8).
`POST /pace` records a rate change on the world's `Timeline`, refused unless a stream is open. When
the last stream closes, the rate returns to the world's own at that moment, so a world paused and
forgotten keeps running. Each change is handed to `keep` before it takes effect, so the state
directory never holds a timeline older than the one the world is running under.

Standard library only, by the owner's decision on #36: `ThreadingHTTPServer` is enough for one
user and one world, and replacing it later sits behind the protocol where no client can see it.

**The game client is served from here too** (#292, decision 5; #306): a path that is not an API
route is a file of the built client, so one process and one origin serve the whole game and no
route needs a CORS policy.
"""

from __future__ import annotations

import contextlib
import json
import math
import queue
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import unquote

import api
from api import wire
from scheduler import CatchUp, Progress, Timeline, WakeSchedule

# How often an idle stream sends a comment line. Writing is the only way a server learns that a
# client has gone, so without it a closed viewer would count as watching until the next wake.
_KEEPALIVE_SECONDS = 15.0

# What the built client is made of, and what each is served as. A table rather than `mimetypes`,
# because on Windows `mimetypes` reads the registry, which commonly maps `.js` to `text/plain`, and
# a browser refuses to run a module served as text. An extension not listed is not served: the
# build emits nothing else, and anything else in the directory was not put there by it.
_CLIENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".woff2": "font/woff2",
    ".json": "application/json",
    ".map": "application/json",
}


class Service:
    """A `Session`, the lock around it, and the ticker that keeps it caught up.

    timeline: the world's anchor and its rate changes so far (#307). It should be at the world's
        own rate, since nobody is watching a service that has not started.
    clock: seconds on the wall clock the anchor was taken on. This is the one place in the
        repository that reads one for the simulation's sake, and it only decides how many ticks
        are owed (§2.1). Passed in so tests can say what time it is.
    chunk: ticks per catch-up chunk. Batching never changes outcomes (#32), so this decides only
        how long a request may wait for the lock and how often the stream reports.
    fastest_ticks_per_second: the most a watching player may set the rate to (`Pacing`).
    keep: called with every new timeline before it takes effect, to record it (`service.state`).
    """

    def __init__(
        self,
        session: api.Session,
        timeline: Timeline,
        schedule: WakeSchedule,
        clock: Callable[[], float],
        chunk: int,
        *,
        fastest_ticks_per_second: float,
        keep: Callable[[Timeline], None],
    ) -> None:
        self.session = session
        self.schedule = schedule
        self.clock = clock
        self.fastest_ticks_per_second = fastest_ticks_per_second
        self.lock = threading.Lock()
        self._timeline = timeline
        self._keep = keep
        self._catchup = CatchUp(_Locked(session, self.lock), self, chunk)
        self._watchers: list[queue.Queue[Optional[Progress]]] = []
        # Guards the audience and the timeline together, because whether anyone is watching is
        # what decides whether a rate may be set: a request that saw a watcher must not land after
        # that watcher's leaving has already restored the world's own rate.
        self._watchers_lock = threading.Lock()
        self.last_seen = clock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._ticker: Optional[threading.Thread] = None

    # -- ticking ------------------------------------------------------------------------------

    @property
    def watched(self) -> bool:
        with self._watchers_lock:
            return bool(self._watchers)

    @property
    def timeline(self) -> Timeline:
        return self._timeline

    def due(self, now: float) -> int:
        """The tick due at `now`, from the timeline as it stands: what `CatchUp` reckons by."""
        return self._timeline.due(now)

    def next_interval(self) -> float:
        """Seconds until the next wake, as the schedule says for the world's current audience."""
        return self.schedule.interval(watched=self.watched, away=self.clock() - self.last_seen)

    def wake(self, now: float) -> int:
        """Pay what the world owes at `now`, announcing every chunk. Returns the ticks run.

        Stops at the next chunk boundary once `stop()` is called, so a shutdown during a long
        catch-up waits for one chunk rather than for the whole absence.
        """
        done = 0
        for progress in self._catchup.run(now):
            done = progress.done
            self._announce(progress)
            if self._stop.is_set():
                break
        return done

    def start(self) -> None:
        """Start the ticker. Its first wake is immediate, so a restarted service pays its
        downtime before it waits for anything."""
        self._ticker = threading.Thread(target=self._tick_forever, name="ticker", daemon=True)
        self._ticker.start()

    def stop(self) -> None:
        """Stop the ticker at the next chunk boundary and wait for it, then end every stream.

        Ending the streams is everyone leaving, so the world's own rate is restored here rather
        than whenever each stream's thread gets round to detaching: the snapshot saved after a stop
        must sit beside a timeline that is already back at the world's own rate.
        """
        self._stop.set()
        self._wake.set()
        if self._ticker is not None:
            self._ticker.join()
        with self._watchers_lock:
            self._retime(self._timeline.at_own_rate(self.clock()))
            for watcher in self._watchers:
                watcher.put(None)

    def _tick_forever(self) -> None:
        while not self._stop.is_set():
            self.wake(self.clock())
            self._wake.wait(timeout=self.next_interval())
            self._wake.clear()

    # -- watching -----------------------------------------------------------------------------

    def attach(self) -> queue.Queue[Optional[Progress]]:
        """Open a stream. Wakes the ticker, so the short interval applies from now rather than
        from whenever the long one would have ended."""
        watcher: queue.Queue[Optional[Progress]] = queue.Queue()
        with self._watchers_lock:
            self._watchers.append(watcher)
        self.seen()
        self._wake.set()
        return watcher

    def detach(self, watcher: queue.Queue[Optional[Progress]]) -> None:
        """Close a stream. The last one closing is when the absence clock starts, and when a pause
        or fast-forward ends: the world's own rate resumes at that moment (#307)."""
        with self._watchers_lock:
            if watcher in self._watchers:
                self._watchers.remove(watcher)
            if not self._watchers:
                self._retime(self._timeline.at_own_rate(self.clock()))
        self.seen()

    def seen(self) -> None:
        self.last_seen = self.clock()

    # -- pace ---------------------------------------------------------------------------------

    def pace(self) -> api.Pace:
        timeline = self._timeline
        return api.Pace(
            ticks_per_second=timeline.ticks_per_second,
            own_ticks_per_second=timeline.own_ticks_per_second,
            fastest_ticks_per_second=self.fastest_ticks_per_second,
        )

    def set_pace(self, ticks_per_second: float) -> api.Pace:
        """Run the world at `ticks_per_second` from now, for as long as somebody watches.

        Raises `ValueError` for a rate that is not one, and `api.Refused` for one the world will
        not run at: past the fastest it sustains, or asked with nobody watching, since a pause
        that outlived the player would let them switch off the world running while they are away.
        """
        if not (math.isfinite(ticks_per_second) and ticks_per_second >= 0):
            raise ValueError(f"a rate is zero or a positive number of ticks, got {ticks_per_second}")
        if ticks_per_second > self.fastest_ticks_per_second:
            raise api.Refused(
                f"{ticks_per_second} ticks a second is past the fastest this world sustains "
                f"({self.fastest_ticks_per_second})"
            )
        with self._watchers_lock:
            if not self._watchers:
                raise api.Refused(
                    "only a watching player can change the pace: open the event stream first"
                )
            self._retime(self._timeline.at_rate(self.clock(), ticks_per_second))
        # A resume or a fast-forward should not wait out the interval the ticker is already in.
        self._wake.set()
        return self.pace()

    def _retime(self, timeline: Timeline) -> None:
        """Adopt `timeline`, recording it first. The caller holds `_watchers_lock`."""
        if timeline is self._timeline:
            return
        self._keep(timeline)
        self._timeline = timeline

    def _announce(self, progress: Progress) -> None:
        with self._watchers_lock:
            for watcher in self._watchers:
                watcher.put(progress)

    # -- serving ------------------------------------------------------------------------------

    def http_server(
        self, host: str, port: int, client: Optional[Path] = None
    ) -> ThreadingHTTPServer:
        """An HTTP server for this service. `port=0` picks a free one (`server.server_port`).

        client: the built game client's directory (`clients/web/dist`), served at `/` for every
            path that is not an API route. None serves the API alone, and `/` says so.
        """
        service = self

        class Handler(_Handler):
            pass

        Handler.service = service
        Handler.client = client.resolve() if client is not None else None
        server = ThreadingHTTPServer((host, port), Handler)
        server.daemon_threads = True
        return server


class _Locked:
    """`Session`'s two catch-up calls, each under the service's lock: what `CatchUp` drives."""

    def __init__(self, session: api.Session, lock: threading.Lock) -> None:
        self._session = session
        self._lock = lock

    def advance(self, ticks: int) -> api.Status:
        with self._lock:
            return self._session.advance(ticks)

    def status(self) -> api.Status:
        with self._lock:
            return self._session.status()


# Read routes: a path, and the `Session` call that answers it. One table, consulted by the
# handler, so a route cannot exist without a call behind it or the other way round.
_READS: tuple[tuple[re.Pattern[str], Callable[..., object]], ...] = (
    (re.compile(r"^/world$"), lambda s: s.describe()),
    (re.compile(r"^/status$"), lambda s: s.status()),
    (re.compile(r"^/terrain$"), lambda s: s.terrain()),
    (re.compile(r"^/frame$"), lambda s: s.frame()),
    (re.compile(r"^/barriers$"), lambda s: s.barriers()),
    (re.compile(r"^/history$"), lambda s: s.history()),
    (re.compile(r"^/ledger$"), lambda s: s.ledger()),
    (re.compile(r"^/condition/([^/]+)$"), lambda s, mode: s.condition(mode)),
    (re.compile(r"^/field/([^/]+)$"), lambda s, layer: s.field(layer)),
    (re.compile(r"^/entity/(-?\d+)$"), lambda s, entity: s.inspect(int(entity))),
)

_WRITES: dict[str, Callable[[api.Session, Any], object]] = {
    "/quote": lambda s, request: s.quote(request),
    "/request": lambda s, request: s.request(request),
}


class _Handler(BaseHTTPRequestHandler):
    service: Service
    client: Optional[Path] = None
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802 - the name http.server dispatches on
        self.service.seen()
        if self.path == "/events":
            self._stream()
            return
        if self.path == "/pace":
            self._answer(self.service.pace, world=False)
            return
        for pattern, call in _READS:
            match = pattern.match(self.path)
            if match:
                args = [unquote(group) for group in match.groups()]
                self._answer(lambda: call(self.service.session, *args))
                return
        self._serve_client()

    def _serve_client(self) -> None:
        """A file of the built client, or 404. `/` is its `index.html`.

        The path is resolved and must stay inside the client's directory, so `/../world.json`
        cannot read the state directory or anything else on a machine serving a public
        repository's code.
        """
        if self.client is None:
            self._error(404, "LookupError", f"no such route: {self.path}; the game client is not served")
            return
        relative = unquote(self.path.split("?", 1)[0]).lstrip("/") or "index.html"
        target = (self.client / relative).resolve()
        kind = _CLIENT_TYPES.get(target.suffix)
        if not target.is_relative_to(self.client) or kind is None or not target.is_file():
            self._error(404, "LookupError", f"no such route: {self.path}")
            return
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        self.service.seen()
        call = _WRITES.get(self.path)
        if call is None and self.path != "/pace":
            self._error(404, "LookupError", f"no such route: {self.path}")
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            request = wire.loads(self.rfile.read(length))
        except (ValueError, KeyError, TypeError) as bad:
            self._error(400, "ValueError", f"not a request: {bad}")
            return
        if call is None:
            self._answer(lambda: self._set_pace(request), world=False)
            return
        self._answer(lambda: call(self.service.session, request))

    def _set_pace(self, request: object) -> api.Pace:
        if not isinstance(request, api.PaceRequest):
            raise TypeError(f"/pace takes a PaceRequest, not {request!r}")
        return self.service.set_pace(request.ticks_per_second)

    def _answer(self, call: Callable[[], object], world: bool = True) -> None:
        """Answer with `call`'s value, or the error it raised. `world` says whether the call
        reads or writes the world and so must hold its lock. The pace does neither: it is the
        service's own, and setting it should not wait out a catch-up chunk."""
        try:
            with self.service.lock if world else contextlib.nullcontext():
                value = call()
        except api.Refused as refused:
            self._error(409, "Refused", str(refused))
            return
        except (ValueError, TypeError) as mistake:
            self._error(400, type(mistake).__name__, str(mistake))
            return
        self._send(200, wire.dumps(value))

    def _stream(self) -> None:
        """Server-sent events: one `progress` event per catch-up chunk, until the client leaves
        or the service stops."""
        watcher = self.service.attach()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            while True:
                try:
                    progress = watcher.get(timeout=_KEEPALIVE_SECONDS)
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    continue
                if progress is None:
                    return
                self.wfile.write(b"event: progress\ndata: " + wire.dumps(progress) + b"\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return
        finally:
            self.service.detach(watcher)
            self.close_connection = True

    def _error(self, code: int, kind: str, message: str) -> None:
        self._send(code, json.dumps({"error": message, "kind": kind}).encode("utf-8"))

    def _send(self, code: int, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Quiet. A request log is #54's observability, and printing one per frame from a
        watching viewer would bury anything that matters."""
