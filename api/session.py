"""One world behind the boundary, and the only thing a client holds that has behaviour (#35).

A `Session` is what becomes a world id once #36 puts a process between the two halves. Every
method takes plain values and returns them (see `api.values`). None takes a callback, a core
object or an array the world still owns, and none returns one. That discipline, not the transport,
is the boundary. The protocol is left open on purpose (gate 1 on #35): built like this,
choosing it later costs nothing.

**Ecological facts are computed on this side.** How hungry an animal is, and what "full" means for
a biomass cell, are readings of the world's physics. A client that computed them would need the
world's config and a pass over its columns, which cannot cross a socket. That is §4's rule that
statistics are computed core-side, never in a client.

**Absent, not stubbed:** standing policies (#33) and warnings (#34) do not exist yet, so neither
appears here (§8.2).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, cast

import numpy as np

from api.values import (
    BarrierView,
    Column,
    ConditionView,
    CullRequest,
    DriveShare,
    EntityReport,
    FenceRequest,
    FieldView,
    Frame,
    GeneReading,
    Ledger,
    Quote,
    Request,
    Sample,
    Status,
    TerrainView,
    WorldInfo,
)
from core.selection import Selection
from core.world.assembly import World
from core.world.cull import Cull
from core.world.fence import Fence
from core.world.interventions import Intervention
from metrics import MetricHistory
from persistence import SnapshotError, load, save
from worlds import WORLDS, build_named_world

# Where each layer's grid lives on an assembled world: `(service attribute, field attribute)`. One
# table rather than a naming convention, because the fields genuinely live on different services,
# and `getattr(plants, layer)` stopped being true the moment a layer came from `Carrion`. It is
# consulted by `field`, and `FIELD_LAYERS` is derived from it, so a layer cannot be offered with
# nothing behind it.
_LAYER_SOURCE = {
    "biomass": ("plants", "biomass"),
    "carrion": ("carrion", "mass"),
    "soil_nutrients": ("plants", "soil_nutrients"),
    "moisture": ("plants", "moisture"),
    "potential_growth": ("plants", "potential_growth"),
}

FIELD_LAYERS: tuple[str, ...] = tuple(_LAYER_SOURCE)
"""The names `Session.field` accepts, in the order a viewer should cycle them."""

CONDITIONS: tuple[str, ...] = ("hunger", "thirst", "age")
"""The names `Session.condition` accepts. Every one is a deficit (see `ConditionView`)."""


class Refused(Exception):
    """The world declined a control call, and doing it anyway would be worse.

    Raised for a load that has no snapshot behind it, or one taken under different rules (§2.8).
    It carries only its message, so it crosses a process boundary as text. A client reports it
    rather than crashing, because refusing a load is a normal outcome.
    """


def worlds() -> tuple[str, ...]:
    """The names `create` accepts."""
    return tuple(WORLDS)


def create(world: str, seed: int, founders: int) -> Session:
    """Build the world called `world` and open a session on it.

    `seed` is logged for replaying a crash, never a promise of determinism (§2.2). An unknown name
    raises rather than falling back to some default world (§8.7).
    """
    return Session(build_named_world(world, seed, founders), world, seed)


class Session:
    """A client's handle on one world. See the module docstring for the rules it keeps."""

    def __init__(self, world: World, name: str, seed: int) -> None:
        self._world = world
        self._name = name
        self._seed = seed
        # Fixed for the session, deliberately: a ramp that rescales itself hides the slow decline a
        # field overlay exists to show. See `_layer_references`.
        self._references = _layer_references(world)

    # -- control ------------------------------------------------------------------------------

    def advance(self, ticks: int) -> Status:
        """Run `ticks` whole ticks, or none once the run has collapsed (§2.7)."""
        if ticks < 1:
            raise ValueError(f"advance at least one tick, got {ticks}")
        self._world.loop.advance(ticks)
        return self.status()

    def status(self) -> Status:
        loop = self._world.loop
        return Status(tick=loop.tick_count, collapsed=loop.collapsed, living=len(self._living()))

    def save(self, path: str) -> str:
        """Write the world to `path`, returning where it went."""
        return str(save(self._world, Path(path)))

    def load(self, path: str) -> Status:
        """Restore the snapshot at `path` into this world, in place, or raise `Refused`.

        In place, because a snapshot carries state and the config carries the rules (§3.2): a
        snapshot can only be opened into a world already built from the rules it was taken under.
        """
        snapshot = Path(path)
        if not snapshot.exists():
            raise Refused(f"no save at {snapshot}")
        try:
            load(self._world, snapshot)
        except SnapshotError as refused:
            raise Refused(str(refused)) from None
        return self.status()

    # -- read ---------------------------------------------------------------------------------

    def describe(self) -> WorldInfo:
        terrain = self._world.terrain
        rows, cols = terrain.heights.shape
        return WorldInfo(
            name=self._name,
            seed=self._seed,
            width=float(terrain.world_width),
            height=float(terrain.world_height),
            grid_shape=(int(rows), int(cols)),
            cell_size=float(terrain.cell_size),
            gene_names=tuple(self._world.genes.vocabulary.names),
            field_layers=FIELD_LAYERS,
            conditions=CONDITIONS,
        )

    def terrain(self) -> TerrainView:
        terrain = self._world.terrain
        return TerrainView(
            elevation=Column.of(terrain.heights),
            slope=Column.of(terrain.slope),
            aspect=Column.of(terrain.aspect),
            water_depth=Column.of(self._world.water.depth),
        )

    def frame(self) -> Frame:
        store = self._world.store
        rows = self._living().to_indices()
        return Frame(
            tick=self._world.loop.tick_count,
            ids=Column.of(store.row_ids()[rows]),
            x=Column.of(store.x[rows]),
            y=Column.of(store.y[rows]),
            z=Column.of(store.z[rows]),
            species=Column.of(store.species_id[rows]),
        )

    def condition(self, mode: str) -> ConditionView:
        """Each living animal's deficit in `mode`, in `frame()`'s order at the same tick.

        Clipped to [0, 1]. An animal fed past satiation wants nothing, and that is all a deficit
        can say about it.
        """
        store = self._world.store
        rows = self._living().to_indices()
        if mode == "hunger":
            # Against the pool level the hunger drive itself treats as "wants nothing", rather
            # than the population's current maximum, which would make a starving herd and a fed
            # one read the same.
            deficit = 1.0 - store.energy[rows] / self._world.config.hunger.satiation_energy
        elif mode == "thirst":
            deficit = store.dehydration[rows].astype(np.float64)
        elif mode == "age":
            # Against the oldest animal alive, which is the only scale the world offers: there is
            # no lifespan gene and no death clock (§2.5), so "old" is a fact about the population.
            ages = store.age[rows].astype(np.float64)
            deficit = ages / max(float(ages.max(initial=0.0)), 1.0)
        else:
            raise ValueError(f"unknown condition {mode!r}; expected one of {CONDITIONS}")
        values = np.clip(deficit, 0.0, 1.0).astype(np.float32)
        return ConditionView(tick=self._world.loop.tick_count, mode=mode, values=Column.of(values))

    def field(self, layer: str) -> FieldView:
        try:
            service, attribute = _LAYER_SOURCE[layer]
        except KeyError:
            raise ValueError(f"unknown field {layer!r}; expected one of {FIELD_LAYERS}") from None
        grid = getattr(getattr(self._world, service), attribute)
        return FieldView(layer=layer, values=Column.of(grid), reference=self._references[layer])

    def barriers(self) -> BarrierView:
        barriers = self._world.barriers
        return BarrierView(
            blocked_north=Column.of(barriers.blocked_north),
            blocked_west=Column.of(barriers.blocked_west),
        )

    def history(self) -> tuple[Sample, ...]:
        """The recorded series, oldest first. Recorded as the world ran, never recomputed (§4)."""
        recorder = self._world.loop.metrics
        if recorder is None:
            return ()
        # `TickLoop` names its recorder by a structural type so that `core/` never imports
        # `metrics/` (§4). The worlds a session opens attach a `MetricHistory`, so this side of the
        # boundary, which may import both, can say so.
        return tuple(cast(MetricHistory, recorder).samples)

    def inspect(self, entity_id: int) -> Optional[EntityReport]:
        """One animal, born or gestating, or None if no such id is allocated now.

        None rather than an error, because an animal a client last saw may have died since, and a
        click on a corpse is an ordinary thing for a player to do.
        """
        world = self._world
        store = world.store
        # `alive` as well as the id, because a free row's id reads -1, so asking for -1 by id
        # alone would describe whatever a dead animal left in the first free row.
        found = np.flatnonzero((store.row_ids() == entity_id) & store.alive)
        if not found.size:
            return None
        row = int(found[0])
        rows = np.array([row], dtype=np.int64)
        one = Selection.from_indices(rows, store.capacity)

        drives = sorted(
            (
                DriveShare(name=name, contribution=float(values[0]))
                for name, values in world.behaviour.breakdown(one).items()
            ),
            key=lambda share: -share.contribution,
        )
        stored = world.genetics.genes_at(rows)[0]
        expressed = world.genetics.expressed_at(rows)[0]
        genes = tuple(
            GeneReading(name=gene.name, expressed=float(expressed[i]), stored=float(stored[i]))
            for i, gene in enumerate(world.config.genes)
        )
        return EntityReport(
            id=int(entity_id),
            species=int(store.species_id[row]),
            age=int(store.age[row]),
            energy=float(store.energy[row]),
            upkeep=float(world.ecology.upkeep(one)[0]),
            exertion=float(store.exertion[row]),
            x=float(store.x[row]),
            y=float(store.y[row]),
            z=float(store.z[row]),
            speed=float(np.hypot(store.velocity_x[row], store.velocity_y[row])),
            top_speed=float(world.movement.top_speed(one)[0]),
            urge=float(store.choice_urge[row]),
            pace=float(world.movement.pace(one, store.choice_urge[rows])[0]),
            drives=tuple(drives),
            genes=genes,
        )

    def ledger(self) -> Ledger:
        interventions = self._interventions()
        return Ledger(balance=interventions.balance, records=tuple(interventions.history))

    # -- write --------------------------------------------------------------------------------

    def quote(self, request: Request) -> Quote:
        """What `request` would cost, and why it would be refused, if the boundary were now."""
        intervention = self._intervention(request)
        return Quote(
            cost=float(intervention.cost()),
            refusal=self._interventions().refusal_of(intervention),
        )

    def request(self, request: Request) -> None:
        """Queue `request` for the next tick boundary. The outcome lands in the ledger (#26)."""
        self._interventions().request(self._intervention(request))

    # -- internals ----------------------------------------------------------------------------

    def _living(self) -> Selection:
        store = self._world.store
        return Selection.from_mask(store.alive & (store.age >= 0))

    def _interventions(self):
        interventions = self._world.loop.interventions
        if interventions is None:
            raise RuntimeError(
                f"world {self._name!r} was built without an intervention ledger, so nothing can be "
                "requested of it"
            )
        return interventions

    def _intervention(self, request: Request) -> Intervention:
        world = self._world
        if isinstance(request, CullRequest):
            return Cull(
                world.ecology, world.store, request.species_id, request.count, request.survivors
            )
        if isinstance(request, FenceRequest):
            return Fence(
                world.barriers,
                request.first_x,
                request.first_y,
                request.second_x,
                request.second_y,
            )
        raise TypeError(f"not a request this world understands: {request!r}")


def _layer_references(world: World) -> dict[str, float]:
    """The value at which each layer reads as full, in that layer's own unit (§3.3).

    Computed once, when the session opens, and held for its life, because a scale that moves hides
    change. Each one means something ecological rather than "the largest value seen so far":

    - `biomass`: `potential_growth.max() / senescence_rate`, the standing crop a cell settles at
      when soil never limits it. `grow()` adds `min(potential_growth, affordable)` and removes
      `biomass × senescence_rate`, so with nutrients to spare the two balance exactly there.
    - `carrion`: `strike_power`, the most one strike can take out of an equally sized victim. It
      was a whole body once, and rendering proved that wrong: the brightest cell measured read 13%,
      so a layer for a running mechanic looked empty.
    - `soil_nutrients`: the per-cell pool every cell started with.
    - `moisture`: 1.0, since it is already a fraction.
    - `potential_growth`: its own maximum at the tick the session opened. It follows the season
      since #159, so a warmer season reads past 1.0, which is the change the fixed scale is there
      to show.
    """
    plants = world.plants
    light_ceiling = float(plants.potential_growth.max())
    if light_ceiling <= 0.0:
        raise ValueError(
            "no cell can grow anything: potential growth is zero across the whole grid, so there "
            "is no scale to draw a biomass ramp against"
        )
    if plants.config.initial_soil_nutrients <= 0.0:
        raise ValueError(
            "initial_soil_nutrients is zero, so a soil ramp has nothing to normalise against"
        )
    return {
        "biomass": light_ceiling / plants.config.senescence_rate,
        "carrion": float(world.config.predation.strike_power),
        "soil_nutrients": float(plants.config.initial_soil_nutrients),
        "moisture": 1.0,
        "potential_growth": light_ceiling,
    }
