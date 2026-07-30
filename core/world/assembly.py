"""World assembly: the one place a store, its services and a tick loop are wired together (#115).

Six domain services existed and **none had ever run alongside another.** The only assembly in the
repository ran `TickLoop(store, systems=())`, so every service was exercised by its own tests
against its own fixture and nothing had ever read what another wrote. This module is what makes a
world run.

**The order is the rule, and it is declared here as data.** `TICK_ORDER` below is the sequence, and
it is what actually sequences the loop — the systems are built into a mapping by name and then
ordered *by* that tuple, so there is exactly one source of truth and a system that is added without
being placed fails at assembly time rather than running wherever import order put it. Under §2.8
the order is part of the MAJOR version, frozen for the life of a world; §4 forbids declaring a rule
as data and then not consulting it, which is why the tuple sequences rather than documents.

The reasoning behind each position lives in CLAUDE.md §2.1, not here — it is a decision of
consequence and belongs where decisions are recorded (§8.6). What belongs here is the fact that
**this is the implemented prefix of that order, in that order.** Four settled steps have no system
yet and are absent rather than stubbed (§8.2):

| settled step | inserted by |
|---|---|
| feeding, after movement | #19 |
| death and decomposition, after upkeep | #21 |
| reproduction, after death | #20 |
| speciation, last; periodicity still open | #16 |

Their absence is visible in what the world does. Nothing eats, so energy only ever leaves the
animals; nothing dies or is born, so the population is fixed at its founders. This is a world that
*runs* — every service reading what the previous ones wrote, over real terrain — and not yet a world
that lives.

**Movement acts for one drive.** `Behaviour` scores all five and partitions the population by
winner, but only hunger has somewhere to walk to today: fleeing is #24's, mate-seeking is #20's,
thirst has no drinking mechanic, and fatigue's answer is to stay put. So the movement step moves
foragers and leaves everyone else standing, which is the honest reading of what exists rather than
a fabricated destination for a drive whose mechanic is not built.

**Founders are naive, never authored** (§2.5, #101). Genes are uniform draws inside a per-gene
range the caller supplies. That distinction is load-bearing rather than stylistic: hand-picking a
founder's aversion vector would be writing down *that rabbits fear wolves*, which is authoring the
outcome instead of the physics. A naive draw lets selection decide, which is the whole point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from core.behaviour.drives import (
    Fatigue,
    FatigueConfig,
    Fear,
    FearConfig,
    Hunger,
    HungerConfig,
    Lust,
    LustConfig,
    Thirst,
    ThirstConfig,
)
from core.behaviour.exertion import Exertion, ExertionConfig
from core.behaviour.movement import Movement, MovementConfig
from core.behaviour.service import Behaviour
from core.ecology.aging import Aging
from core.ecology.cues import CueField, CueFieldConfig, Scent, ScentGenes
from core.ecology.metabolism import Metabolism, MetabolismConfig
from core.ecology.plants import Plants, PlantsConfig
from core.ecology.service import Ecology
from core.entities.store import EntityStore
from core.genetics.expression import GeneticsConfig
from core.genetics.service import Genetics
from core.genetics.species import SpeciesRegistry
from core.genetics.vocabulary import GeneVocabulary
from core.invariants import default_registry
from core.selection import Selection
from core.services import ColumnRegistry
from core.world.climate import Climate, ClimateConfig
from core.world.terrain import Terrain, TerrainConfig
from core.world.tick import TickLoop
from core.world.water import Water

TICK_ORDER: tuple[str, ...] = (
    "plant_growth",
    "cue_field_rebuild",
    "drive_scoring",
    "movement",
    "exertion_recovery",
    "metabolic_upkeep",
    "age_increment",
)
"""The order systems run within a tick — a rule, not an implementation detail (CLAUDE.md §2.1).

Consulted rather than documentary: `build_world` orders the systems it builds by this tuple, and
raises if the two disagree in either direction. See the module docstring for the settled steps that
have no system yet.
"""


class SystemOrderError(Exception):
    """The systems built and `TICK_ORDER` disagree.

    Either a system was built without being placed in the order, or the order names one that does
    not exist. Both are assembly-time mistakes about a versioned rule, so both fail here rather
    than at the first tick (§8.7).
    """


@dataclass(frozen=True)
class WorldConfig:
    """Everything a world needs to exist, per world and never as constants in `core/` (§2.1).

    gene_names: the world's gene vocabulary, in column order. Per-world because §2.3 makes the
        vocabulary versioned and additive-only, and because every config below that names a gene
        (`MovementConfig.speed_gene`, `FearConfig.aversion_genes`, …) indexes into it.
    founder_gene_ranges: per-gene ``(low, high)`` for the uniform draw that seeds founders. **Every
        gene in `gene_names` must appear**, and a missing one raises: the same rule
        `MetabolismConfig` applies to costs, for the same reason — a gene left out would silently
        found the world at zero, which for a signature gene means every creature smelling
        identical and for a speed gene means a population that cannot move.
    n_founders: how many creatures the world begins with. The store is sized to hold exactly them;
        population is emergent from there (§2.3) and nothing here sets a target.
    """

    terrain: TerrainConfig
    climate: ClimateConfig
    plants: PlantsConfig
    cue_field: CueFieldConfig
    metabolism: MetabolismConfig
    genetics: GeneticsConfig
    movement: MovementConfig
    exertion: ExertionConfig
    hunger: HungerConfig
    thirst: ThirstConfig
    fear: FearConfig
    lust: LustConfig
    fatigue: FatigueConfig
    scent_genes: ScentGenes
    gene_names: tuple[str, ...]
    founder_gene_ranges: Mapping[str, tuple[float, float]]
    n_founders: int
    founder_energy: float

    def __post_init__(self) -> None:
        missing = set(self.gene_names) - set(self.founder_gene_ranges)
        if missing:
            raise ValueError(
                f"founder_gene_ranges omits {sorted(missing)}; every gene needs a founding range "
                "or the world starts with it at zero and nothing says so"
            )
        for name, (low, high) in self.founder_gene_ranges.items():
            if high < low:
                raise ValueError(f"founder range for '{name}' is inverted: ({low}, {high})")
        if self.n_founders < 1:
            raise ValueError("a world needs at least one founder")


@dataclass(frozen=True)
class World:
    """A wired world: the store, every service that owns part of it, and the loop that runs them.

    Held as one object because a world *is* the whole set — the services share one store and one
    `ColumnRegistry`, and handing a caller a subset would let it construct a second service against
    the same column and be told so only at the next assembly (§2.3). There is no singleton here and
    no global: a test builds as many of these as it likes (§4).

    `loop.advance(n)` is the entire simulation surface. Everything else on this object is for the
    diagnostic viewer and for tests to look at what a tick did (§3.3).
    """

    config: WorldConfig
    terrain: Terrain
    water: Water
    climate: Climate
    plants: Plants
    store: EntityStore
    columns: ColumnRegistry
    vocabulary: GeneVocabulary
    species: SpeciesRegistry
    genetics: Genetics
    ecology: Ecology
    exertion: Exertion
    movement: Movement
    behaviour: Behaviour
    aging: Aging
    scent: Scent
    founders: Selection
    loop: TickLoop


def build_world(config: WorldConfig, seed: int, debug_checks: bool = False) -> World:
    """Build a world from `config` and return it with its loop ready to advance.

    seed: seeds terrain generation and the founder draw. Generation is reproducible; the simulation
        that follows is not (§2.2), so this makes a *starting state* repeatable and never a run.
    debug_checks: run the invariant harness after every tick (§6). Off by default because it is a
        debug-build facility, and the loop pays one boolean per tick for it being available.
    """
    terrain = Terrain.generate(config.terrain)
    water = Water.generate(terrain)
    climate = Climate(terrain, config.climate)
    plants = Plants(terrain, climate, water, config.plants)

    vocabulary = GeneVocabulary(config.gene_names)
    # The store's two column blocks are sized from what is actually registered below, so a drive
    # added without widening the block fails in `Behaviour.register` at assembly time rather than
    # overwriting a neighbour's score.
    store = EntityStore(
        initial_capacity=config.n_founders,
        n_drives=len(_DRIVE_NAMES),
        n_genes=len(config.gene_names),
    )
    columns = ColumnRegistry()

    species = SpeciesRegistry(vocabulary)
    genetics = Genetics(store, columns, species, vocabulary, config.genetics)
    ecology = Ecology(
        store,
        columns,
        genetics,
        climate,
        # The cost table and the expression modes are two halves of one rule, and this is the only
        # place a world declares both — so it is where they are checked against each other (#136).
        Metabolism(vocabulary, config.metabolism, config.genetics.expression_modes),
    )
    exertion = Exertion(store, columns, config.exertion)
    movement = Movement(
        store, columns, ecology, exertion, genetics, terrain, vocabulary, config.movement
    )
    behaviour = Behaviour(store, columns)
    aging = Aging(store, columns)
    scent = Scent(
        store,
        genetics,
        CueField(terrain, len(config.scent_genes.signature_genes), config.cue_field),
        vocabulary,
        config.scent_genes,
    )

    hunger = Hunger(store, ecology, genetics, plants, vocabulary, config.hunger)
    drives = (
        hunger,
        Thirst(store, climate, config.thirst),
        Fear(store, genetics, scent, vocabulary, config.fear),
        Lust(store, ecology, config.lust),
        Fatigue(store, exertion, config.fatigue),
    )
    for drive in drives:
        behaviour.register(drive)
    if behaviour.drive_names != _DRIVE_NAMES:
        # The store's `drive_scores` width came from _DRIVE_NAMES, and `driven_by` addresses drives
        # by name, so the two disagreeing means a column block sized for one set of drives holding
        # another (§8.7).
        raise SystemOrderError(
            f"registered drives {behaviour.drive_names} do not match {_DRIVE_NAMES}"
        )

    founders = _found(config, store, genetics, species, movement, terrain, seed)

    systems = _build_systems(
        store,
        plants,
        scent,
        behaviour,
        hunger,
        movement,
        exertion,
        ecology,
        aging,
        config.movement.walking_pace,
    )
    loop = TickLoop(
        store,
        systems=_ordered(systems),
        invariants=default_registry(
            0.0, terrain.world_width, 0.0, terrain.world_height, plants=plants
        ),
        debug_checks=debug_checks,
    )
    return World(
        config=config,
        terrain=terrain,
        water=water,
        climate=climate,
        plants=plants,
        store=store,
        columns=columns,
        vocabulary=vocabulary,
        species=species,
        genetics=genetics,
        ecology=ecology,
        exertion=exertion,
        movement=movement,
        behaviour=behaviour,
        aging=aging,
        scent=scent,
        founders=founders,
        loop=loop,
    )


# Registration order is the column order in `drive_scores` and also the tie-break between equal
# scores (`Behaviour`), so it is as much a rule as TICK_ORDER is and is declared the same way.
_DRIVE_NAMES: tuple[str, ...] = ("hunger", "thirst", "fear", "lust", "fatigue")


def _ordered(systems: Mapping[str, Callable[[], None]]) -> tuple[Callable[[], None], ...]:
    """`systems` sequenced by `TICK_ORDER`, raising if the two do not name the same set.

    This is the line that makes the declared order load-bearing rather than decorative (§4). A
    system built and not placed would otherwise never run; a name placed with nothing behind it
    would otherwise be a silent gap in the tick.
    """
    unplaced = set(systems) - set(TICK_ORDER)
    unbuilt = set(TICK_ORDER) - set(systems)
    if unplaced or unbuilt:
        raise SystemOrderError(
            f"TICK_ORDER and the built systems disagree: unplaced={sorted(unplaced)}, "
            f"unbuilt={sorted(unbuilt)}"
        )
    return tuple(systems[name] for name in TICK_ORDER)


def _build_systems(
    store: EntityStore,
    plants: Plants,
    scent: Scent,
    behaviour: Behaviour,
    hunger: Hunger,
    movement: Movement,
    exertion: Exertion,
    ecology: Ecology,
    aging: Aging,
    walking_pace: float,
) -> dict[str, Callable[[], None]]:
    """One zero-argument callable per system, by name. `_ordered` decides the sequence.

    Each closes over the services it needs, which is the same pattern invariants and drives already
    use (§6): one uniform signature, with the collaborators bound at construction rather than
    threaded through a world-context argument enumerating domains that do not all exist yet (§8.2).

    Every system re-reads `alive` rather than closing over a population. A `Selection` is a snapshot
    (§4), and once #20 and #21 land the living set changes *within* a tick, so a population captured
    at assembly would be a stale view of the world by the second tick and a wrong one by the second
    birth.
    """

    def living() -> Selection:
        return Selection.from_mask(store.alive)

    def move_foragers() -> None:
        # Only hunger has a destination today; see the module docstring. Everyone else has been
        # scored and stands still, which is what the absent mechanics mean rather than a bug.
        foragers = behaviour.driven_by("hunger", living())
        target_x, target_y = hunger.forage_target(foragers)
        movement.step(foragers, target_x, target_y, walking_pace)

    return {
        "plant_growth": plants.grow,
        "cue_field_rebuild": lambda: scent.rebuild(living()),
        "drive_scoring": lambda: behaviour.score(living()),
        "movement": move_foragers,
        "exertion_recovery": lambda: exertion.recover(living()),
        "metabolic_upkeep": lambda: ecology.drain(living()),
        "age_increment": lambda: aging.advance(living()),
    }


def _found(
    config: WorldConfig,
    store: EntityStore,
    genetics: Genetics,
    species: SpeciesRegistry,
    movement: Movement,
    terrain: Terrain,
    seed: int,
) -> Selection:
    """Allocate the founding population: naive genes, scattered over the surface.

    One species expressing every gene. A second founding species would need a reason to differ, and
    the only honest reason is that selection made them differ — which is what the simulation is for
    (§2.5, #101).
    """
    rng = np.random.default_rng(seed)
    n = config.n_founders

    x = rng.uniform(0.0, terrain.world_width, n).astype(np.float32)
    y = rng.uniform(0.0, terrain.world_height, n).astype(np.float32)
    species_id = species.register(config.gene_names)
    store.allocate(
        n,
        x=x,
        y=y,
        energy=np.full(n, config.founder_energy, dtype=np.float32),
        # Full health: injury is a state the world produces, not one it starts in. `allocate`
        # defaults it to zero, which its own docstring calls dead (#106).
        health=np.ones(n, dtype=np.float32),
        species_id=np.full(n, species_id, dtype=np.int32),
    )
    # The living set *is* the founding set at this instant, which is why the ids `allocate` returns
    # are not needed: nothing has died and nothing has been born. Asking `alive` rather than
    # translating ids to rows also keeps row indices inside the store, where §2.3 requires them.
    founders = Selection.from_mask(store.alive)

    low = np.array([config.founder_gene_ranges[name][0] for name in config.gene_names])
    high = np.array([config.founder_gene_ranges[name][1] for name in config.gene_names])
    genes = rng.uniform(low, high, (n, len(config.gene_names))).astype(np.float32)
    genetics.set_genes(founders, genes)
    # z is the ground under (x, y): a freshly allocated row holds z = 0, which is underground
    # anywhere the terrain rises above it.
    movement.settle(founders)
    return founders
