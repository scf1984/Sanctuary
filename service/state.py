"""Where a served world lives between runs: its snapshot, and the founding it was built from (#36).

A state directory holds three files:

- `world.json`, written **once**, when the world is founded: its name, seed and founder count,
  the scenario it plays if any (#317), and the anchor (#32), the moment at which tick 0 and the
  wall clock were both known. It never changes afterwards, so it can never disagree with a newer
  snapshot. That is why the anchor sits here and not in the snapshot, and the scenario too: a
  world's goal is chosen when it is founded, and a restart plays the same one.
- `timeline.json`, every rate change since the anchor (#307): the world's history of time, from
  which its age at any moment can be reconstructed. Rewritten at every change, before the change
  takes effect, and absent until the first one. The anchor is not repeated in it, so the two
  files cannot disagree about where time began.
- `snapshot.npz`, rewritten at every shutdown (§3.2: drain, never kill).

**A restart owes its downtime.** Ticks owed come from the timeline, so a service that was stopped
for an hour comes back an hour behind and pays it on its first wake. The world kept running while
nobody served it, exactly as §2.4 says it does while nobody watches.

**A restart resumes at the world's own rate.** Nobody is watching a world that has just started,
and a pause holds only while somebody watches (#292, decision 8). A graceful stop has already
restored it (`Service.stop`); a hard kill mid-pause has not, and then the own rate resumes at the
restart, since the moment the player left is not known. That is the downtime a hard kill loses
anyway, which is #52's to solve.

**A snapshot is replaced, never overwritten in place.** It is the only copy of the world there is
(§3.2), and a crash halfway through a write must leave the previous one intact.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import api
from scheduler import Anchor, RateChange, Timeline

_FOUNDING = "world.json"
_TIMELINE = "timeline.json"
_SNAPSHOT = "snapshot.npz"
_SAVING = "snapshot.saving.npz"


@dataclass(frozen=True)
class Founding:
    """What a served world was built from, and when."""

    world: str
    seed: int
    founders: int
    anchor: Anchor
    # None for a sandbox world. A founding written before scenarios existed (#317) has no entry,
    # and it was a sandbox, so it reads as one.
    scenario: Optional[str] = None


def open_world(
    state: Path,
    now: float,
    ticks_per_second: float,
    world: Optional[str] = None,
    seed: Optional[int] = None,
    founders: Optional[int] = None,
    scenario: Optional[str] = None,
) -> tuple[api.Session, Founding]:
    """The world in `state`: restored if it has been founded there, founded at `now` if not.

    `world`, `seed` and `founders` found a new world, and `scenario` gives it a goal (#317). Given
    for an existing one, they must match what it was founded with, because a state directory holds
    one world, and quietly ignoring an argument that asked for a different one is §8.7's founding
    example.
    """
    founding_path = state / _FOUNDING
    if founding_path.exists():
        founding = _read(founding_path)
        asked = {"world": world, "seed": seed, "founders": founders, "scenario": scenario}
        clash = {k: v for k, v in asked.items() if v is not None and v != getattr(founding, k)}
        if clash:
            raise ValueError(
                f"{state} holds {founding.world!r} (seed {founding.seed}, {founding.founders} "
                f"founders, scenario {founding.scenario!r}), not what was asked for: {clash}. "
                "Use another state directory."
            )
        session = api.create(
            founding.world,
            seed=founding.seed,
            founders=founding.founders,
            scenario=founding.scenario,
        )
        snapshot = state / _SNAPSHOT
        if snapshot.exists():
            session.load(str(snapshot))
        return session, founding

    if world is None or seed is None or founders is None:
        raise ValueError(f"{state} holds no world yet; founding one needs a world, seed and founders")
    session = api.create(world, seed=seed, founders=founders, scenario=scenario)
    founding = Founding(
        world=world,
        seed=seed,
        founders=founders,
        anchor=Anchor(wall_seconds=now, tick=session.status().tick, ticks_per_second=ticks_per_second),
        scenario=scenario,
    )
    state.mkdir(parents=True, exist_ok=True)
    _write(founding_path, founding)
    return session, founding


def close_world(session: api.Session, state: Path) -> Path:
    """Snapshot `session` into `state`, replacing the previous snapshot only once the new one is
    whole. Returns where it went."""
    saving = Path(session.save(str(state / _SAVING)))
    final = state / _SNAPSHOT
    os.replace(saving, final)
    return final


def open_timeline(state: Path, anchor: Anchor, now: float) -> Timeline:
    """The timeline of the world founded at `anchor` in `state`, at the world's own rate from
    `now` on (see the module docstring), recording the resumption if there was one."""
    path = state / _TIMELINE
    changes: tuple[RateChange, ...] = ()
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        changes = tuple(RateChange(**change) for change in data["changes"])
    timeline = Timeline(anchor, changes)
    resumed = timeline.at_own_rate(now)
    if resumed is not timeline:
        keep_timeline(state, resumed)
    return resumed


def keep_timeline(state: Path, timeline: Timeline) -> None:
    """Record `timeline`'s rate changes in `state`, replacing the previous record only once the new
    one is whole: it is the only account of how old the world should be."""
    _replace(state / _TIMELINE, {"changes": [dataclasses.asdict(c) for c in timeline.changes]})


def _read(path: Path) -> Founding:
    data = json.loads(path.read_text(encoding="utf-8"))
    return Founding(
        world=data["world"],
        seed=int(data["seed"]),
        founders=int(data["founders"]),
        anchor=Anchor(**data["anchor"]),
        scenario=data.get("scenario"),
    )


def _write(path: Path, founding: Founding) -> None:
    _replace(path, dataclasses.asdict(founding))


def _replace(path: Path, record: dict) -> None:
    draft = path.with_suffix(".writing")
    draft.write_text(json.dumps(record, indent=2), encoding="utf-8")
    os.replace(draft, path)
