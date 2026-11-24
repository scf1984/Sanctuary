"""Everything that crosses the boundary between a world and a client, as values (#35).

**A value, never a view.** §3.1 puts the world on a shared machine with clients asking for what to
draw, so anything returned here is something a client may hold across a tick and send over a
socket. Every type in this module is a frozen dataclass whose fields are None, bool, int, float,
str, tuples of those, other types from this module, or a `Column`. Nothing here has behaviour that
reaches back into a world, and nothing holds an array the world will go on writing to.

**Entities are named by id, never by row** (§2.3). Rows are recycled and ids never are, so an id is
the only number a client can hold across ticks and still mean the same animal by. A client never
sees a row at all. That makes #119's corpse-at-its-death-site bug unrepresentable on the client
side, rather than a rule it has to remember.

Three types are carried rather than defined here: `metrics.Sample`,
`core.world.interventions.Record` and `scheduler.Progress`. Each is already frozen and plain, and
each is owned by the issue that defines what it reports (#30, #26, #32). A copy here would be a
second definition free to drift from the first (§2.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from core.world.interventions import Record
from metrics import Sample
from scheduler.catchup import Progress

__all__ = [
    "BarrierView",
    "Column",
    "ConditionView",
    "CullRequest",
    "DriveShare",
    "EntityReport",
    "FenceRequest",
    "FieldView",
    "Frame",
    "GeneReading",
    "Ledger",
    "Pace",
    "PaceRequest",
    "Progress",
    "Quote",
    "Record",
    "Request",
    "Sample",
    "ScenarioReport",
    "SpeciesNameRequest",
    "SpeciesReport",
    "Status",
    "TerrainView",
    "WorldInfo",
]


@dataclass(frozen=True)
class Column:
    """An array crossing the boundary: a little-endian copy, as bytes, plus what it takes to read
    it back.

    Bytes rather than a list of floats, by the owner's decision on #35. Measured at 100,000
    entities, three position columns cost ~100 ms and 4.6 MB a frame as lists in JSON, against
    ~9 ms and 2.1 MB packed. That gap is a tenth of §2.1's whole tick budget per watcher frame, and
    the boundary's shape is what every client depends on, so it is not something to change later.

    It is still a plain value. `data` is a copy, so the world writing on next tick cannot reach it,
    and it goes over a socket as one binary frame or as base64 in JSON without a codec that knows
    anything about the world.

    dtype: a NumPy dtype string, always little-endian (`"<f4"`, `"<i8"`, `"|b1"`), so a reader on
        any machine decodes the same numbers.
    shape: the array's shape. `(n,)` for a per-entity column, `(rows, cols)` for a grid.
    data: the elements, C-contiguous.
    """

    dtype: str
    shape: tuple[int, ...]
    data: bytes

    @classmethod
    def of(cls, values: np.ndarray) -> Column:
        array = np.ascontiguousarray(values)
        little = array.astype(array.dtype.newbyteorder("<"), copy=False)
        return cls(
            dtype=little.dtype.str,
            shape=tuple(int(n) for n in little.shape),
            data=little.tobytes(),
        )

    def to_numpy(self) -> np.ndarray:
        """The array, **read-only**: a client that writes into a frame is editing nothing but its
        own copy, and saying so loudly is better than letting it look like it worked (§8.7)."""
        return np.frombuffer(self.data, dtype=np.dtype(self.dtype)).reshape(self.shape)


@dataclass(frozen=True)
class Status:
    """Where a world stands after a control call.

    tick: ticks completed. The only clock (§2.1).
    collapsed: whether the run has ended (§2.7). Read from the world rather than latched.
    living: animals alive and born. A gestating young is not one yet (#20).
    splits: how many splits the world has recorded (#314). It rides every catch-up progress event,
        so a client sees it rise when a split lands, live or during catch-up, and fetches
        `species()` to see what it was. A count rather than the split itself, because a client
        that missed an event, or opens after an absence, reads the same record either way.
    """

    tick: int
    collapsed: bool
    living: int
    splits: int


@dataclass(frozen=True)
class WorldInfo:
    """What does not change for the life of a session: the map's extent and the vocabulary a
    client needs to ask for anything else by name.

    width, height: world units.
    grid_shape: `(rows, cols)` of every grid this boundary returns (terrain, fields, barriers).
    cell_size: world units per cell.
    gene_names: the gene vocabulary, in column order.
    field_layers: the names `Session.field` accepts.
    conditions: the names `Session.condition` accepts.
    cull_survivors: the fewest born animals a cull may leave of a species (§2.7, #28), so a client
        can say what the guard keeps before it asks.
    """

    name: str
    seed: int
    width: float
    height: float
    grid_shape: tuple[int, int]
    cell_size: float
    gene_names: tuple[str, ...]
    field_layers: tuple[str, ...]
    conditions: tuple[str, ...]
    cull_survivors: int


@dataclass(frozen=True)
class TerrainView:
    """The static ground, as grids. Shading it is the client's business; these are the facts.

    elevation, water_depth: world units. slope: radians. aspect: radians, counterclockwise from +x.
    """

    elevation: Column
    slope: Column
    aspect: Column
    water_depth: Column


@dataclass(frozen=True)
class Frame:
    """Every living animal at one tick boundary, as parallel columns in one order.

    Only the **living**: a gestating row is allocated but is not yet a participant (#20), so it is
    not drawn. Its position is where it was conceived, and it does not move until it is born, so
    drawing it would leave a motionless animal at every conception site.

    ids: `<i8`, stable for an animal's whole life and never reused (§2.3). A client blends two
        frames by matching ids. A row index would be the bug #119 fixed.
    x, y, z: world units.
    species: species id per animal.
    """

    tick: int
    ids: Column
    x: Column
    y: Column
    z: Column
    species: Column


@dataclass(frozen=True)
class ConditionView:
    """One deficit per living animal, aligned with the `Frame` of the same tick.

    values: `<f4` in [0, 1]. 0 is an animal wanting nothing, 1 is one at the edge. Every mode is a
        deficit, so a client colouring by it darkens toward death (§3.3). The reference each mode
        is measured against is a fact of the world, which is why the reduction happens here and
        not in a client (§4).
    """

    tick: int
    mode: str
    values: Column


@dataclass(frozen=True)
class FieldView:
    """One per-cell world field, and the value at which it should read as full.

    reference: in the field's own unit, fixed when the session was opened. It is a fact about the
        world (what a cell settles at, what one strike leaves), never the field's current maximum,
        because a scale that rescales every frame hides exactly the slow decline a field exists to
        show (§3.3).
    """

    layer: str
    values: Column
    reference: float


@dataclass(frozen=True)
class BarrierView:
    """The fence edges: a cell's north and west edges, blocked or not (#27). A cell's south edge is
    its neighbour's north, so these two grids hold every edge exactly once."""

    blocked_north: Column
    blocked_west: Column


@dataclass(frozen=True)
class DriveShare:
    """One drive's contribution to the option an animal last took (#114)."""

    name: str
    contribution: float


@dataclass(frozen=True)
class GeneReading:
    """One gene, as expressed and as stored. Side by side because the expression mode is visible
    only in the pair: a magnitude folding across zero is invisible from either number alone."""

    name: str
    expressed: float
    stored: float


@dataclass(frozen=True)
class EntityReport:
    """One animal, read for a person (§3.3: "click-to-inspect").

    age: ticks. Negative is the gestation clock (#20): born in `-age` ticks.
    energy, upkeep: energy units, and energy units per tick.
    speed, top_speed: world units per tick. pace: fraction of top speed the last urge asked for.
    drives: every drive's share of the last decision, largest first.
    """

    id: int
    species: int
    age: int
    energy: float
    upkeep: float
    exertion: float
    x: float
    y: float
    z: float
    speed: float
    top_speed: float
    urge: float
    pace: float
    drives: tuple[DriveShare, ...]
    genes: tuple[GeneReading, ...]


@dataclass(frozen=True)
class Ledger:
    """The player's accounting: what is left, what came in, and every intervention's outcome in
    order (#26, #308).

    balance: currency units left to spend.
    earned: currency units of income so far. The founding balance plus `earned`, minus every
        record's cost, is `balance`.
    income_per_tick: currency units a tick at the last metrics sample: what the world earns now.
    """

    balance: float
    earned: float
    income_per_tick: float
    records: tuple[Record, ...]


@dataclass(frozen=True)
class Quote:
    """What a request would cost and whether it would be refused, **as the world stands now**.

    It is a forecast of a tick boundary that has not happened yet. A species can starve in between,
    so the outcome that counts is the one in the ledger. Refusal is checked before price, as the
    boundary checks it (#26), so the interesting reason comes first.
    """

    cost: float
    refusal: Optional[str]


@dataclass(frozen=True)
class CullRequest:
    """Remove `count` born animals of `species_id`, chosen at random (#28).

    Carries no floor: how many a cull must leave is the world's rule (`WorldInfo.cull_survivors`),
    so no client can weaken it.
    """

    species_id: int
    count: int


@dataclass(frozen=True)
class FenceRequest:
    """Enclose the rectangle between two opposite corners, in world units (#27)."""

    first_x: float
    first_y: float
    second_x: float
    second_y: float


Request = Union[CullRequest, FenceRequest]
"""What a client may ask a world to do. Data, never a core object: the session builds the
intervention, so a drag in a viewer and a standing policy firing during an absence travel the same
path (§2.7)."""


@dataclass(frozen=True)
class SpeciesReport:
    """One species, living or extinct, as the player sees it in the lineage list (#314).

    id: the species id a frame's `species` column and `EntityReport.species` carry.
    name: what the player called it, or None until they do. The placeholder shown meanwhile is the
        client's: it is presentation, and a stored placeholder would read as a name chosen.
    parent: the species it split from, or None for a founding species.
    split_tick: the tick it split off on, or None for a founding species.
    founders: how many animals it split off with, or None for a founding species.
    living: animals of this species alive and born now. Zero is extinct, which is permanent
        (§2.7), so an extinct species stays listed rather than vanishing.
    """

    id: int
    name: Optional[str]
    parent: Optional[int]
    split_tick: Optional[int]
    founders: Optional[int]
    living: int


@dataclass(frozen=True)
class ScenarioReport:
    """The scenario a world was founded with, and where its goal stands (#317).

    name: the scenario's name, as it was founded with.
    title, goal_text, briefing: what a person is told, written beside the goal they describe
        (`worlds.scenarios`). A client shows them and writes none of its own (§4).
    outcome: "open", "met" or "lost". Met and lost are final.
    progress: dimensionless. Before the split, how far the founding species' two most separated
        groups have come toward one, at the latest check that measured it; 1 is where it splits
        them. None until a check has measured it. 1 from the split on. A measurement, never a
        forecast (§2.3).
    parent, daughter: the species the goal names, once its split has happened; None before.
    split_tick, due_tick: when that split happened, and the tick of the sample at which both
        species must still be alive. None before the split.
    resolved_tick: when the goal was met or lost; None while it is open.
    reason: why it was lost, for a person; None otherwise.
    """

    name: str
    title: str
    goal_text: str
    briefing: tuple[str, ...]
    outcome: str
    progress: Optional[float]
    parent: Optional[int]
    daughter: Optional[int]
    split_tick: Optional[int]
    due_tick: Optional[int]
    resolved_tick: Optional[int]
    reason: Optional[str]


@dataclass(frozen=True)
class SpeciesNameRequest:
    """Call species `species_id` by `name` (#314). Not an intervention: it costs nothing, changes
    nothing the world does, and so is applied at once rather than at a tick boundary (§2.7)."""

    species_id: int
    name: str


@dataclass(frozen=True)
class Pace:
    """How fast a served world runs, and the range a watching player may set (#307).

    ticks_per_second: the rate now. Zero while paused.
    own_ticks_per_second: the world's own rate, §2.1's live rate. Leaving always returns to it.
    fastest_ticks_per_second: the most the world is measured to sustain, and so the most offered.
    """

    ticks_per_second: float
    own_ticks_per_second: float
    fastest_ticks_per_second: float


@dataclass(frozen=True)
class PaceRequest:
    """Run the served world at `ticks_per_second`: 0 pauses, the world's own rate resumes, and
    anything up to the fastest fast-forwards. Granted only while the asker is watching."""

    ticks_per_second: float
