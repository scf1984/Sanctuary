"""Writing a world's state to one file, and reading it back into an assembled world (#31)."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np

# Bumped whenever what is written changes shape. There is exactly one version, so there is no
# migration here and none is written: §8.2 forbids machinery without a caller, and a migration
# from a schema that never shipped would be untestable by construction. What *is* here is the
# refusal — an unknown version raises rather than being read hopefully, which is what makes the
# first real migration a change to this module rather than an archaeology exercise.
SCHEMA_VERSION = 2

# Every array the world cannot recompute. Terrain, water, the cue field and the forage field are
# all pure functions of the config or of the state below, so storing them would be storing a
# derivation — and a stored derivation is one that can disagree with what it came from. Terrain
# becomes state the day terraforming (#152) can edit it, and that is a schema bump.
_STORE_COLUMNS = (
    "x",
    "y",
    "z",
    "velocity_x",
    "velocity_y",
    "energy",
    "age",
    "health",
    "exertion",
    "species_id",
    "drive_scores",
    "choice_heading",
    "choice_moving",
    "choice_urge",
    "genes",
    "alive",
)


class SnapshotError(Exception):
    """A snapshot cannot be read into this world, and reading it anyway would be worse.

    Raised for an unknown schema version and for a config that is not the one the world was saved
    under. Both are refusals rather than repairs: §2.8's whole argument is that a world's
    populations *are* an equilibrium reached under one rule set, so loading state into different
    rules produces a collapse the player cannot attribute to anything they did.
    """


def fingerprint(config) -> str:
    """A stable hash of everything about a world that is a *rule* rather than a state.

    Canonical JSON over the config tree, sorted by key, with anything JSON cannot express — enums,
    tuples of gene specs — rendered by `str`. That covers the cost table, the inheritance
    parameters, the drive coefficients, the gene vocabulary and its expression modes: exactly the
    set §2.8 says a world pins forever.

    It is a *fingerprint* and not a serialisation, and the distinction is the point. It cannot
    rebuild a world, so it does not have to be exhaustive to be useful — it only has to change when
    the rules change, which is what makes "this snapshot does not belong to this world" a question
    with an answer.
    """
    return hashlib.sha256(
        json.dumps(dataclasses.asdict(config), sort_keys=True, default=str).encode()
    ).hexdigest()


def save(world, path: Path) -> Path:
    """Write `world`'s state to `path`, and return it. Creates the parent directory.

    One file, arrays and metadata together, because a snapshot split across two is a snapshot that
    can be half-copied — and §3.2 treats losing one as unrecoverable, which makes atomicity of the
    *unit* worth more than the convenience of separate parts.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    store = world.store
    meta = {
        "schema_version": SCHEMA_VERSION,
        "fingerprint": fingerprint(world.config),
        "tick_count": world.loop.tick_count,
        "next_id": store.ids_issued,
        "exported_nutrients": world.plants.exported_nutrients,
        # The generator's own state, so a reloaded world draws onward rather than replaying the
        # sequence it already used. Without it a world saved at tick 1,000 would restart from the
        # seed and repeat its first thousand ticks' worth of draws — not a correctness bug under
        # §2.2, but a repetition nobody asked for and one that would compound with every reload.
        "rng": world.rng.bit_generator.state,
    }
    arrays = {f"store.{name}": getattr(store, name) for name in _STORE_COLUMNS}
    arrays["row_ids"] = store.row_ids()
    # Order matters and `row_ids` cannot carry it — see `EntityStore.free_rows`.
    arrays["free_rows"] = store.free_rows()
    arrays["species_masks"] = world.species.mask_table()
    arrays["plants.biomass"] = world.plants.biomass
    arrays["plants.soil_nutrients"] = world.plants.soil_nutrients
    # Carrion mass is state, not a derivation: a body on the ground is the record of a kill
    # that already happened and nothing can recompute where it fell (§3.2). Its `scent` is
    # derived from it every tick and is therefore deliberately absent.
    arrays["carrion.mass"] = world.carrion.mass
    np.savez(path, meta=np.array(json.dumps(meta)), **arrays)
    return path


def load(world, path: Path) -> None:
    """Restore `path` into `world`, in place, or raise `SnapshotError`.

    `world` must already be assembled from the config the snapshot was taken under. That is the
    shape rather than a `load_world(path)` factory for two reasons, and both are §2.8's: the config
    is what carries the rules, and building a world is `build_world`'s job alone (§7.2) — a second
    construction path is exactly the duplicate assembly that rule exists to prevent.

    Restores in place rather than returning a new world, so every reference a caller already holds
    — the viewer's services, a recorder attached to the loop — keeps pointing at the world that now
    holds the loaded state. Handing back a second object would leave the first one live and stale,
    and nothing would say which was which.
    """
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["meta"]))
        _refuse_if_foreign(meta, world)
        world.store.restore(
            {name: archive[f"store.{name}"] for name in _STORE_COLUMNS},
            archive["row_ids"],
            meta["next_id"],
            archive["free_rows"],
        )
        world.species.restore(archive["species_masks"])
        world.plants.biomass[...] = archive["plants.biomass"]
        world.plants.soil_nutrients[...] = archive["plants.soil_nutrients"]
        world.carrion.mass[...] = archive["carrion.mass"]

    world.plants.exported_nutrients = meta["exported_nutrients"]
    world.plants.rebuild_forage()
    world.rng.bit_generator.state = meta["rng"]
    world.loop.tick_count = meta["tick_count"]
    # The loop holds a position snapshot taken before the load, and the renderer interpolates
    # between two of them (§3.3, #119). Advancing zero ticks re-reads the world without running
    # anything, so the first frame after a load draws where the animals *are* rather than streaking
    # them in from wherever the pre-load world left them.
    world.loop.advance(0)


def _refuse_if_foreign(meta: dict, world) -> None:
    """Raise unless this snapshot belongs to this world's rules and this module's schema."""
    if meta["schema_version"] != SCHEMA_VERSION:
        raise SnapshotError(
            f"snapshot is schema v{meta['schema_version']} and this build reads "
            f"v{SCHEMA_VERSION}; there is no migration between them yet (#31)"
        )
    expected = fingerprint(world.config)
    if meta["fingerprint"] != expected:
        raise SnapshotError(
            "snapshot was taken under a different world config: it fingerprints "
            f"{meta['fingerprint'][:12]} against this world's {expected[:12]}. A world runs under "
            "the rules it was created with (§2.8) — loading it into a differently-tuned world "
            "would collapse an equilibrium for reasons invisible to the player"
        )
