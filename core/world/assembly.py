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
**this is the implemented prefix of that order, in that order.** Two settled steps have no system
yet and are absent rather than stubbed (§8.2):

| settled step | inserted by |
|---|---|
| speciation, last; periodicity still open | #16 |

Energy enters an animal, animals die of running out of it, and animals now breed — so a
population can rise as well as fall, and for the first time a gene can be *passed on* rather than
only filtered. `core.genetics.inheritance` had been built and tested since #14 and had never once
been called by a running world.

Conception runs after `age_increment` for the reason §2.1 gives: `age` counts whole ticks lived, and
incrementing after birth would hand a newborn an age of one having lived none. Note the two
populations the loop distinguishes — `living()` is what *acts*, and excludes the unborn by their
negative age; `gestating_or_living()` is what `Aging` advances, because ageing an unborn row toward
zero is precisely how gestation is timed (#20).

**Decomposition is not in this step**, despite §2.1 naming it there. An animal's nutrient debt is
exactly its energy and starvation empties that, so a starved carcass is worth nothing and there is
no mass for a carrion field to hold (#21). What actually closes §2.5's loop is `Ecology.spend`
returning the nutrients an animal burns to the cell it is standing in — respiration, every tick,
for every spender — and that is a property of the pool rather than a step in the order.

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
from core.behaviour.service import Behaviour, BehaviourConfig
from core.ecology.aging import Aging
from core.ecology.conception import Conception, ConceptionConfig
from core.ecology.cues import CueField, CueFieldConfig, Scent, ScentGenes
from core.ecology.death import Death
from core.ecology.diet import Diet, DietConfig
from core.ecology.feeding import Feeding, FeedingConfig
from core.ecology.metabolism import Metabolism, MetabolismConfig
from core.ecology.plants import Plants, PlantsConfig
from core.ecology.service import Ecology
from core.entities.growth import GrowthConfig
from core.entities.store import EntityStore
from core.genetics.expression import GeneticsConfig
from core.genetics.service import Genetics
from core.genetics.species import SpeciesRegistry
from core.genetics.registry import GeneRegistry, GeneSpec
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
    "forage_field_rebuild",
    "drive_scoring",
    "movement",
    "exertion_recovery",
    "feeding",
    "metabolic_upkeep",
    "death",
    "age_increment",
    "conception",
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

    genes: the world's genes, in column order, each declaring its cost, expression mode, unit and
        meaning (`core.genetics.registry`, #111). Per-world because §2.3 makes the vocabulary
        versioned and additive-only, and because every config below that names a gene
        (`MovementConfig.speed_gene`, `FearConfig.aversion_genes`, …) resolves against it — and now
        against the *unit* it expects, which is what makes the declaration load-bearing.
    founder_gene_ranges: per-gene ``(low, high)`` for the uniform draw that seeds founders. **Every
        gene must appear**, and a missing one raises, for the same reason a gene left out would silently
        found the world at zero, which for a signature gene means every creature smelling
        identical and for a speed gene means a population that cannot move.
    n_founders: how many creatures the world begins with. The store is sized to hold exactly them;
        population is emergent from there (§2.3) and nothing here sets a target.
    """

    terrain: TerrainConfig
    climate: ClimateConfig
    plants: PlantsConfig
    growth: GrowthConfig
    conception: ConceptionConfig
    diet: DietConfig
    feeding: FeedingConfig
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
    behaviour: BehaviourConfig
    scent_genes: ScentGenes
    genes: tuple[GeneSpec, ...]
    founder_gene_ranges: Mapping[str, tuple[float, float]]
    n_founders: int
    founder_energy: float

    def __post_init__(self) -> None:
        missing = {gene.name for gene in self.genes} - set(self.founder_gene_ranges)
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
    genes: GeneRegistry
    species: SpeciesRegistry
    genetics: Genetics
    ecology: Ecology
    feeding: Feeding
    death: Death
    conception: Conception
    exertion: Exertion
    movement: Movement
    behaviour: Behaviour
    aging: Aging
    scent: Scent
    # A `Selection` is a mask over a store of a particular capacity, so `founders` does not
    # survive the store growing (#127): after a capacity change its mask is shorter than the
    # columns it would index. Deliberate rather than a wart — a selection is a snapshot (§4),
    # and the founders stop being a meaningful population the moment anything is born. Read
    # `store.alive` for who is here now.
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

    genes = GeneRegistry(config.genes)
    # The store's two column blocks are sized from what is actually registered below, so a drive
    # added without widening the block fails in `Behaviour.register` at assembly time rather than
    # overwriting a neighbour's score.
    store = EntityStore(
        # The founders plus their reserve, so the world is able to breed from its first tick. Sized
        # to the founders exactly, a store has zero free rows, and `Conception` truncates to what is
        # available rather than raising — so the world would be born sterile until the first tick
        # boundary grew it (#127).
        initial_capacity=config.n_founders
        + max(1, int(config.n_founders * config.growth.reserve_fraction)),
        n_drives=len(_DRIVE_NAMES),
        n_genes=len(genes),
    )
    columns = ColumnRegistry()

    species = SpeciesRegistry(genes.vocabulary)
    genetics = Genetics(store, columns, species, genes, config.genetics)
    ecology = Ecology(
        store,
        columns,
        genetics,
        climate,
        Metabolism(genes, config.metabolism),
        plants,
    )
    feeding = Feeding(
        store,
        plants,
        genetics,
        ecology,
        Diet(genes, config.diet),
        genes,
        config.feeding,
    )
    death = Death(store, ecology)
    conception = Conception(store, ecology, genetics, genes, config.conception)
    exertion = Exertion(store, columns, config.exertion)
    movement = Movement(
        store, columns, ecology, exertion, genetics, terrain, genes, config.movement
    )
    behaviour = Behaviour(
        store, columns, genetics, genes, terrain, config.behaviour
    )
    aging = Aging(store, columns)
    scent = Scent(
        store,
        genetics,
        CueField(terrain, len(config.scent_genes.signature_genes), config.cue_field),
        genes,
        config.scent_genes,
    )

    hunger = Hunger(store, ecology, genetics, plants, genes, config.hunger)
    drives = (
        hunger,
        Thirst(store, climate, genetics, genes, config.thirst),
        Fear(store, genetics, scent, genes, config.fear),
        Lust(store, ecology, genetics, scent, genes, config.lust),
        Fatigue(store, exertion, genetics, genes, config.fatigue),
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

    rng = np.random.default_rng(seed)
    founders = _found(config, store, genetics, species, movement, terrain, rng)
    # Founders hold an endowment the field never supplied, and every excretion and carcass
    # returns nutrients *against* the export ledger (#21). Recording their bodies here is what
    # makes `total_nutrients()` constant from tick zero rather than from whenever they first
    # happen to eat — and it is the only call in the world that moves that total.
    plants.record_founding_stock(config.n_founders * config.founder_energy)

    systems = _build_systems(
        store,
        plants,
        scent,
        behaviour,
        hunger,
        movement,
        exertion,
        ecology,
        feeding,
        death,
        aging,
        conception,
        rng,
        config.movement.walking_pace,
    )
    loop = TickLoop(
        store,
        systems=_ordered(systems),
        invariants=default_registry(
            0.0, terrain.world_width, 0.0, terrain.world_height, plants=plants
        ),
        debug_checks=debug_checks,
        growth=config.growth,
    )
    return World(
        config=config,
        terrain=terrain,
        water=water,
        climate=climate,
        plants=plants,
        store=store,
        columns=columns,
        genes=genes,
        species=species,
        genetics=genetics,
        ecology=ecology,
        feeding=feeding,
        death=death,
        conception=conception,
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
    feeding: Feeding,
    death: Death,
    aging: Aging,
    conception: Conception,
    rng: np.random.Generator,
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
        """Everything that acts this tick: allocated, and born.

        A gestating row is allocated and holds a **negative age** until its term is up (#20), so
        excluding it here keeps it out of sensing, movement, feeding, upkeep, death and scent at
        once — one condition rather than six. It is also the sharpest statement of §2.1's rule that
        a newborn does not act in the tick it is born: an unborn animal is not half-simulated, it is
        simply not yet a participant.
        """
        return Selection.from_mask(store.alive & (store.age >= 0))

    def gestating_or_living() -> Selection:
        """Everything allocated, born or not — the only population `age` itself advances over.

        `Aging` is the gestation clock. It counts an unborn row up toward zero exactly as it counts
        a living one up from it, which is why gestation needs no countdown of its own (#20).
        """
        return Selection.from_mask(store.alive)

    def move_chosen() -> None:
        # Everyone moves, because everyone chose — including the animals that chose to stay, whose
        # target is their own position and whose step therefore costs nothing (#114). There is no
        # longer a subset "driven by" one drive: a heading is the sum of every drive's opinion.
        population = living()
        target_x, target_y = behaviour.chosen_target(population)
        movement.step(population, target_x, target_y, walking_pace)

    return {
        "plant_growth": plants.grow,
        "cue_field_rebuild": lambda: scent.rebuild(living()),
        "forage_field_rebuild": plants.rebuild_forage,
        "drive_scoring": lambda: behaviour.choose(living(), rng),
        "movement": move_chosen,
        "exertion_recovery": lambda: exertion.recover(living()),
        "feeding": lambda: feeding.feed(living()),
        "metabolic_upkeep": lambda: ecology.drain(living()),
        "death": lambda: death.reap(living()),
        "age_increment": lambda: aging.advance(gestating_or_living()),
        "conception": lambda: conception.conceive(living(), rng),
    }


def _found(
    config: WorldConfig,
    store: EntityStore,
    genetics: Genetics,
    species: SpeciesRegistry,
    movement: Movement,
    terrain: Terrain,
    rng: np.random.Generator,
) -> Selection:
    """Allocate the founding population: naive genes, scattered over the surface.

    One species expressing every gene. A second founding species would need a reason to differ, and
    the only honest reason is that selection made them differ — which is what the simulation is for
    (§2.5, #101).
    """
    n = config.n_founders

    x = rng.uniform(0.0, terrain.world_width, n).astype(np.float32)
    y = rng.uniform(0.0, terrain.world_height, n).astype(np.float32)
    species_id = species.register(tuple(gene.name for gene in config.genes))
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

    names = tuple(gene.name for gene in config.genes)
    low = np.array([config.founder_gene_ranges[name][0] for name in names])
    high = np.array([config.founder_gene_ranges[name][1] for name in names])
    genes = rng.uniform(low, high, (n, len(names))).astype(np.float32)
    genetics.set_genes(founders, genes)
    # z is the ground under (x, y): a freshly allocated row holds z = 0, which is underground
    # anywhere the terrain rises above it.
    movement.settle(founders)
    return founders
