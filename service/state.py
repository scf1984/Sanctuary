"""Where a served world lives between runs: its snapshot, and the founding it was built from (#36).

A state directory holds two files:

- `world.json`, written **once**, when the world is founded: its name, seed and founder count,
  and the anchor (#32), the moment at which tick 0 and the wall clock were both known. It never
  changes afterwards, so it can never disagree with a newer snapshot. That is why the anchor
  sits here and not in the snapshot.
- `snapshot.npz`, rewritten at every shutdown (§3.2: drain, never kill).

**A restart owes its downtime.** Ticks owed come from the founding anchor, so a service that was
stopped for an hour comes back an hour behind and pays it on its first wake. The world kept
running while nobody served it, exactly as §2.4 says it does while nobody watches.

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
from scheduler import Anchor

_FOUNDING = "world.json"
_SNAPSHOT = "snapshot.npz"
_SAVING = "snapshot.saving.npz"


@dataclass(frozen=True)
class Founding:
    """What a served world was built from, and when."""

    world: str
    seed: int
    founders: int
    anchor: Anchor


def open_world(
    state: Path,
    now: float,
    ticks_per_second: float,
    world: Optional[str] = None,
    seed: Optional[int] = None,
    founders: Optional[int] = None,
) -> tuple[api.Session, Founding]:
    """The world in `state`: restored if it has been founded there, founded at `now` if not.

    `world`, `seed` and `founders` found a new world. Given for an existing one, they must match
    what it was founded with, because a state directory holds one world, and quietly ignoring an
    argument that asked for a different one is §8.7's founding example.
    """
    founding_path = state / _FOUNDING
    if founding_path.exists():
        founding = _read(founding_path)
        asked = {"world": world, "seed": seed, "founders": founders}
        clash = {k: v for k, v in asked.items() if v is not None and v != getattr(founding, k)}
        if clash:
            raise ValueError(
                f"{state} holds {founding.world!r} (seed {founding.seed}, {founding.founders} "
                f"founders), not what was asked for: {clash}. Use another state directory."
            )
        session = api.create(founding.world, seed=founding.seed, founders=founding.founders)
        snapshot = state / _SNAPSHOT
        if snapshot.exists():
            session.load(str(snapshot))
        return session, founding

    if world is None or seed is None or founders is None:
        raise ValueError(f"{state} holds no world yet; founding one needs a world, seed and founders")
    session = api.create(world, seed=seed, founders=founders)
    founding = Founding(
        world=world,
        seed=seed,
        founders=founders,
        anchor=Anchor(wall_seconds=now, tick=session.status().tick, ticks_per_second=ticks_per_second),
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


def _read(path: Path) -> Founding:
    data = json.loads(path.read_text(encoding="utf-8"))
    return Founding(
        world=data["world"],
        seed=int(data["seed"]),
        founders=int(data["founders"]),
        anchor=Anchor(**data["anchor"]),
    )


def _write(path: Path, founding: Founding) -> None:
    draft = path.with_suffix(".writing")
    draft.write_text(json.dumps(dataclasses.asdict(founding), indent=2), encoding="utf-8")
    os.replace(draft, path)
