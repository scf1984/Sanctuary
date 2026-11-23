"""A session on a world served by another process (#36): `Session`'s surface, over HTTP.

What a client holds is the same either way. `api.create` opens a world in this process, and
`api.connect` reaches one the service (`service/`) is running, and every read and write a client
makes is spelled identically. That is what #35 built the boundary transport-free for: picking the
protocol had to cost a client nothing.

**Three calls are refused rather than forwarded**: `advance`, `save` and `load`. A served world
advances on its own schedule and nowhere else, which is #32's one execution path enforced at the
protocol. It saves its own snapshot when it shuts down (§3.2). A client asking to drive it, or to
write files on the machine serving it, is asking for something the service does not offer.

Standard library only (`urllib`), by the owner's decision on #36: one runtime dependency is what
the repository has, and this is enough for one user and one world.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Iterator, Optional

from api import wire
from api.session import Refused
from api.values import (
    BarrierView,
    ConditionView,
    EntityReport,
    FieldView,
    Frame,
    Ledger,
    Pace,
    PaceRequest,
    Progress,
    Quote,
    Request,
    Sample,
    SpeciesNameRequest,
    SpeciesReport,
    Status,
    TerrainView,
    WorldInfo,
)


def connect(url: str, timeout: float = 30.0) -> RemoteSession:
    """A session on the world served at `url`, e.g. `http://127.0.0.1:8765`. Checked at once, so
    a wrong address fails here rather than at the first frame."""
    session = RemoteSession(url, timeout)
    session.describe()
    return session


class RemoteSession:
    """See the module docstring. Errors arrive as the exceptions a local session raises: `Refused`
    for a refusal and `ValueError` for a name the world does not have."""

    def __init__(self, url: str, timeout: float) -> None:
        self._url = url.rstrip("/")
        self._timeout = timeout

    # -- control ------------------------------------------------------------------------------

    def advance(self, ticks: int) -> Status:
        raise Refused("a served world advances on its own schedule (#32), not on a client's")

    def status(self) -> Status:
        return self._get("/status")

    def save(self, path: str) -> str:
        raise Refused("a served world saves its own snapshot when it shuts down (§3.2)")

    def load(self, path: str) -> Status:
        raise Refused("a served world is restored by restarting the service on its state")

    # -- read ---------------------------------------------------------------------------------

    def describe(self) -> WorldInfo:
        return self._get("/world")

    def terrain(self) -> TerrainView:
        return self._get("/terrain")

    def frame(self) -> Frame:
        return self._get("/frame")

    def condition(self, mode: str) -> ConditionView:
        return self._get(f"/condition/{urllib.parse.quote(mode, safe='')}")

    def field(self, layer: str) -> FieldView:
        return self._get(f"/field/{urllib.parse.quote(layer, safe='')}")

    def barriers(self) -> BarrierView:
        return self._get("/barriers")

    def history(self) -> tuple[Sample, ...]:
        return self._get("/history")

    def inspect(self, entity_id: int) -> Optional[EntityReport]:
        return self._get(f"/entity/{int(entity_id)}")

    def ledger(self) -> Ledger:
        return self._get("/ledger")

    def species(self) -> tuple[SpeciesReport, ...]:
        return self._get("/species")

    # -- write --------------------------------------------------------------------------------

    def quote(self, request: Request) -> Quote:
        return self._post("/quote", request)

    def request(self, request: Request) -> None:
        self._post("/request", request)

    def name(self, request: SpeciesNameRequest) -> SpeciesReport:
        return self._post("/name", request)

    # -- watching -----------------------------------------------------------------------------

    def pace(self) -> Pace:
        """How fast the world runs now, and what a watching player may set it to (#307)."""
        return self._get("/pace")

    def set_pace(self, ticks_per_second: float) -> Pace:
        """Pause (0), resume (the world's own rate) or fast-forward, up to the fastest offered.

        **Only while watching.** It raises `Refused` unless some client holds `events()` open,
        and the rate returns to the world's own the moment the last one closes, so "the world
        kept running while you were away" holds however the player left it (#292, decision 8).
        A rate past the fastest is refused too, because it would only build a debt of ticks the
        world pays after the player has gone. A local session has no counterpart: it advances
        when its client says, so it has no clock to set.
        """
        return self._post("/pace", PaceRequest(ticks_per_second=ticks_per_second))

    def events(self) -> Iterator[Progress]:
        """Every catch-up chunk as the service announces it, until the connection closes.

        **Holding this open is what makes the world watched** (gate 2 on #35): while any stream
        is open the service wakes on its short interval, so a client that wants to see the world
        move keeps one open. It blocks between events, so a single-threaded client runs it on a
        thread of its own.
        """
        with urllib.request.urlopen(self._url + "/events") as stream:
            data: list[str] = []
            for raw in stream:
                line = raw.decode("utf-8").rstrip("\r\n")
                if line.startswith("data:"):
                    data.append(line[5:].lstrip())
                elif not line and data:
                    yield wire.decode(json.loads("\n".join(data)))
                    data = []

    # -- plumbing -----------------------------------------------------------------------------

    def _get(self, path: str):
        return self._send(urllib.request.Request(self._url + path))

    def _post(self, path: str, value):
        return self._send(
            urllib.request.Request(
                self._url + path,
                data=wire.dumps(value),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
        )

    def _send(self, request: urllib.request.Request):
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                return wire.loads(response.read())
        except urllib.error.HTTPError as error:
            body = json.loads(error.read() or b"{}")
            message = body.get("error", error.reason)
            if error.code == 409:
                raise Refused(message) from None
            if error.code == 400:
                # The exception the local call would have raised, so a client handles a remote
                # world's mistakes exactly as it handles its own.
                raise (TypeError if body.get("kind") == "TypeError" else ValueError)(message) from None
            raise
