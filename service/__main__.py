"""Serve one world until stopped: `python -m service --state DIR [--world demo --seed 0 ...]`.

The composition root of the service (#36): it reads the wall clock, opens or founds the world in
`--state`, starts the ticker and the HTTP server, and on SIGINT or SIGTERM stops the ticker at a
chunk boundary and snapshots before exiting (§3.2: drain, never kill).

`run` is everything but the signals, so a test can stop it with an event instead. Python only lets
the main thread install a signal handler, so `main`, which does install them, is the one part a test
cannot drive.
"""

from __future__ import annotations

import argparse
import signal
import threading
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

import api
from service.server import Service
from service.state import close_world, keep_timeline, open_timeline, open_world
from worlds import pacing_for

# The game client's build output (#306): `npm run build` in `clients/web` writes it.
_CLIENT = Path(__file__).resolve().parents[1] / "clients" / "web" / "dist"

# Used only when founding. A world that already exists keeps what it was founded with.
_FOUNDING_DEFAULTS = {"world": api.DEFAULT_WORLD, "seed": 0, "founders": 200}


def parse(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve one Sanctuary world over HTTP")
    parser.add_argument("--state", type=Path, required=True, help="directory holding the world")
    parser.add_argument("--world", choices=api.worlds(), help="founds a new world (default demo)")
    parser.add_argument("--seed", type=int, help="founds a new world (default 0)")
    parser.add_argument("--founders", type=int, help="founds a new world (default 200)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--client",
        type=Path,
        default=_CLIENT,
        help="the built game client to serve at / (default clients/web/dist)",
    )
    parser.add_argument(
        "--chunk", type=int, default=60, help="ticks per catch-up chunk: how often the stream reports"
    )
    return parser.parse_args(argv)


def run(
    args: argparse.Namespace,
    stopping: threading.Event,
    clock: Callable[[], float] = time.time,
    announce: Callable[[str], None] = lambda line: print(line, flush=True),
) -> Path:
    """Serve until `stopping` is set, then drain and snapshot. Returns where the snapshot went."""
    asked = {"world": args.world, "seed": args.seed, "founders": args.founders}
    if not (args.state / "world.json").exists():
        # Defaults apply only when founding. For a world that already exists, an argument left out
        # means "whatever it was founded with", and one given must match it (see `open_world`).
        asked = {key: _FOUNDING_DEFAULTS[key] if value is None else value for key, value in asked.items()}
    now = clock()
    session, founding = open_world(
        args.state,
        now=now,
        ticks_per_second=pacing_for(asked["world"] or api.DEFAULT_WORLD).ticks_per_second,
        **asked,
    )
    pacing = pacing_for(founding.world)
    service = Service(
        session,
        open_timeline(args.state, founding.anchor, now),
        pacing.schedule,
        clock,
        args.chunk,
        fastest_ticks_per_second=pacing.fastest_ticks_per_second,
        keep=lambda timeline: keep_timeline(args.state, timeline),
    )
    # A missing build is said out loud rather than served as a blank page: the API still works
    # (the viewer's --connect needs nothing else), and the line says what to run (§8.7).
    client = args.client if (args.client / "index.html").is_file() else None
    server = service.http_server(args.host, args.port, client)
    threading.Thread(target=server.serve_forever, name="http", daemon=True).start()
    service.start()
    announce(
        f"serving {founding.world!r} (seed {founding.seed}) at http://{args.host}:{server.server_port}, "
        f"tick {session.status().tick}; Ctrl+C to stop"
    )
    if client is None:
        announce(f"no game client at {args.client}: run `npm ci && npm run build` in clients/web")

    # A timed wait rather than a bare one, so a signal is handled promptly on every platform.
    while not stopping.wait(timeout=0.5):
        pass

    # Drain in this order: no new ticks, then no new requests, then the snapshot, so the world that
    # is saved is one nothing can still be changing (§3.2).
    service.stop()
    server.shutdown()
    server.server_close()
    with service.lock:
        saved = close_world(session, args.state)
    announce(f"saved {saved} at tick {session.status().tick}")
    return saved


def main() -> None:
    stopping = threading.Event()
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), lambda *_: stopping.set())
    run(parse(), stopping)


if __name__ == "__main__":
    main()
