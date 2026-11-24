"""Writing a world's state to one file, and reading it back into an assembled world (#31)."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Mapping, Optional

import numpy as np

from core.genetics.speciation import Split
from core.world.interventions import Record

# Bumped whenever what is written changes shape. Only the current version is read, so there is no
# migration here and none is written: §8.2 forbids machinery without a caller, and a migration
# from a schema that never shipped would be untestable by construction. What *is* here is the
# refusal — an unknown version raises rather than being read hopefully, which is what makes the
# first real migration a change to this module rather than an archaeology exercise.
#
# v6 added the intervention ledger and the metric history (#31). A v5 snapshot is refused rather
# than read with a fresh budget and an empty series, which is exactly the loss v6 exists to stop.
#
# v7 added the lineage (#16): which species split from which. A v6 world could never have split,
# but reading one would still be reading a file that does not say so.
#
# v8 added income to the ledger (#308): what has been earned and the rate it is being earned at.
# A v7 ledger cannot say how much of its balance was income, so it does not reconcile.
#
# v9 added each split's tick and founder count to the lineage, the player's names for species, and
# the living count per species to every metric sample (#314). A v8 lineage cannot say when its
# splits happened, and a v8 series has no per-species counts to fill in.
#
# v10 added what the speciation check last measured, per species, to the world and to every metric
# sample, and the tick each extinct species was found extinct (#317). A v9 series cannot show how
# far a fenced group had come toward a split, and a v9 recorder cannot say whether a daughter died
# within a generation once the window has forgotten.
SCHEMA_VERSION = 10

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
    "dehydration",
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


def save(world, path: Path, names: Mapping[int, str]) -> Path:
    """Write `world`'s state to `path`, and return it. Creates the parent directory.

    `names` is what the player has called each species, by id (#314). It is presentation, so it
    lives beside the world in `api` and never in `core/`, where something that decides an outcome
    could come to read it. It is saved here rather than in a file of its own so a restore brings
    back names and lineage from one moment: a name can never outlive, or predate, its species.
    Required rather than defaulted, so a caller that holds names cannot forget to pass them and
    lose them silently (§8.7).

    One file, arrays and metadata together, because a snapshot split across two is a snapshot that
    can be half-copied — and §3.2 treats losing one as unrecoverable, which makes atomicity of the
    *unit* worth more than the convenience of separate parts.

    **The ledger and the series are reached through the loop**, at `world.loop.interventions` and
    `world.loop.metrics`, which is where the composition root attaches them. They are not passed
    in as arguments. `core/` still knows nothing of `metrics/` (§4): this module reads whatever the
    loop holds through its structural types, and `build_world` still constructs neither. An
    argument would make a second place for "which recorder belongs to this world", and a caller
    could save with one and load without it, silently emptying the series. Read off the loop, the
    world is saved as it is running, and `load` refuses a world that does not match it.
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
        "interventions": _ledger(world.loop.interventions),
        "metrics": _recording(world.loop.metrics),
        # Which species split from which, when, and from how many (#16, #314). The masks say what
        # species exist and the ids say who belongs to each, but neither says where a species came
        # from, and nothing can recompute it. In the order the splits were made.
        "lineage": [
            [split.species_id, split.parent_species_id, split.tick, split.founders]
            for split in world.speciation.lineage.splits()
        ],
        # What the last speciation check measured (#317). It decides nothing, but without it a
        # reloaded world reads no divergence until the next check, a generation later, and the
        # series a player watches for progress drops to zero for no reason they could see.
        "divergence": sorted(
            [int(species), float(reading)]
            for species, reading in world.speciation.divergence.items()
        ),
        # JSON object keys are strings, so the ids are written as pairs.
        "names": sorted([int(species), str(name)] for species, name in names.items()),
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
    # Fences are state and emphatically not a derivation: a barrier exists because the player
    # built it, and nothing in the config or the terrain implies it (§3.2). Losing them on
    # reload would silently undo the one intervention the world was shaped by.
    arrays["barriers.north"] = world.barriers.blocked_north
    arrays["barriers.west"] = world.barriers.blocked_west
    if world.loop.metrics is not None:
        arrays.update(
            {f"metrics.{name}": column for name, column in world.loop.metrics.columns().items()}
        )
    np.savez(path, meta=np.array(json.dumps(meta)), **arrays)
    return path


def load(world, path: Path) -> dict[int, str]:
    """Restore `path` into `world`, in place, and return the species names saved with it, or
    raise `SnapshotError`. The names go back to whoever keeps them (see `save`).

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
        world.speciation.lineage.restore(
            tuple(
                Split(species_id=species, parent_species_id=parent, tick=tick, founders=founders)
                for species, parent, tick, founders in meta["lineage"]
            )
        )
        world.speciation.divergence = {
            int(species): float(reading) for species, reading in meta["divergence"]
        }
        world.plants.biomass[...] = archive["plants.biomass"]
        world.plants.soil_nutrients[...] = archive["plants.soil_nutrients"]
        world.carrion.mass[...] = archive["carrion.mass"]
        world.barriers.blocked_north[...] = archive["barriers.north"]
        world.barriers.blocked_west[...] = archive["barriers.west"]
        # Bumped so every cached derivation rebuilds against the restored fences rather than
        # against the empty ones the world was assembled with.
        world.barriers.revision += 1
        if meta["metrics"] is not None:
            world.loop.metrics.restore(
                {
                    name[len("metrics."):]: archive[name]
                    for name in archive.files
                    if name.startswith("metrics.")
                },
                meta["metrics"]["previous"],
                {int(species): int(tick) for species, tick in meta["metrics"]["extinctions"]},
            )

    if meta["interventions"] is not None:
        world.loop.interventions.restore(
            meta["interventions"]["balance"],
            [Record(**line) for line in meta["interventions"]["history"]],
            meta["interventions"]["earned"],
            meta["interventions"]["income_per_tick"],
        )

    world.plants.exported_nutrients = meta["exported_nutrients"]
    world.plants.rebuild_forage()
    world.rng.bit_generator.state = meta["rng"]
    world.loop.tick_count = meta["tick_count"]
    # Weather is a function of the tick (#159) and was never written, so it is evaluated rather than
    # read: the sky of the last tick the saved world ran, which is what its field held when saved.
    world.climate.advance_to(max(world.loop.tick_count - 1, 0))
    # The loop holds a position snapshot taken before the load, and the renderer interpolates
    # between two of them (§3.3, #119). Advancing zero ticks re-reads the world without running
    # anything, so the first frame after a load draws where the animals *are* rather than streaking
    # them in from wherever the pre-load world left them.
    world.loop.advance(0)
    return {int(species): str(name) for species, name in meta["names"]}


def _ledger(interventions) -> Optional[dict]:
    """The intervention ledger as plain values, or `None` for a world that has none (#31).

    Balance and history, and **not the pending queue**. A request that had not reached a tick
    boundary when the world was saved is dropped by the save: it was a decision about a world that
    then stopped, and firing it on reload is the failure #26 refuses retries for (see
    `Interventions.restore`). `Record` is all plain values, so it goes through JSON as it is.
    """
    if interventions is None:
        return None
    return {
        "balance": interventions.balance,
        "earned": interventions.earned,
        "income_per_tick": interventions.income_per_tick,
        "history": [dataclasses.asdict(record) for record in interventions.history],
    }


def _recording(metrics) -> Optional[dict]:
    """What the metric recorder holds beside its window, or `None` for a world that records none.

    The window itself goes in the arrays (`MetricHistory.columns`), because it is large. This is
    the part that is small: the previous reading the first sample after a reload differences its
    conceptions and deaths against, the tick each extinct species was found extinct (which the
    window forgets, #317), and the recording policy the window was taken under, which
    `_refuse_if_foreign` checks. A recorder's config is not a rule and is not in the fingerprint,
    since a recording never moves an outcome, so it needs its own check.
    """
    if metrics is None:
        return None
    return {
        "config": dataclasses.asdict(metrics.config),
        "previous": metrics.previous_reading,
        # JSON object keys are strings, so the ids are written as pairs.
        "extinctions": sorted([int(species), int(tick)] for species, tick in metrics.extinctions.items()),
    }


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
    # Both ways round, for both. A snapshot's ledger or series loaded into a world with nowhere to
    # put it would be silently dropped, and the next save would lose it for good. A world that has
    # one, loading a snapshot that does not, would keep a ledger or a series from before the load:
    # a budget nobody spent, and a series that restarts mid-world with its first flows at zero.
    _refuse_unless_both(meta["interventions"], world.loop.interventions, "an intervention ledger")
    _refuse_unless_both(meta["metrics"], world.loop.metrics, "a metric history")
    if meta["metrics"] is not None:
        recorded = meta["metrics"]["config"]
        recording = dataclasses.asdict(world.loop.metrics.config)
        if recorded != recording:
            raise SnapshotError(
                f"snapshot's metric history was taken under the recording policy {recorded}, and "
                f"this world records under {recording}. A series continued under another policy "
                "is two quantities on one axis, which `MetricsConfig` refuses even to downsample"
            )


def _refuse_unless_both(saved, attached, what: str) -> None:
    """Raise unless the snapshot and the world either both carry `what` or both lack it."""
    if (saved is None) != (attached is None):
        held, lacking = ("snapshot", "world") if attached is None else ("world", "snapshot")
        raise SnapshotError(
            f"the {held} carries {what} and the {lacking} does not; loading would either drop "
            "one or keep one from a different world"
        )
