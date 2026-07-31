"""Benchmark harness for issue #114: what option sampling actually costs per tick.

Throwaway spike code -- not part of the simulation core (CLAUDE.md 8.3). It exists to produce the
numbers docs/spikes/option-sampling-cost.md is built from, and should not be imported by anything
else.

#114 says the cost is `N candidates x n_drives x n_entities` field reads per tick and calls it "the
largest single cost multiplier in the tick", requiring a measurement rather than an assertion
(8.5). This measures three things the design rests on:

  1. **Scaling in N.** If cost were linear in N with a large slope, N would be a knob nobody could
     afford to turn up, and the "raise N and rely on jitter" answer to angular resolution (#114's
     stated alternative to a two-stage refinement pass) would be empty.

  2. **Where the time actually goes**, split into: generating and positioning candidates, each
     drive's `appeal`, and the Boltzmann draw. The claim under test is that the *field reads*
     dominate -- which is what makes "sample the candidate positions once per entity and let every
     drive read its own field there" the right structure.

  3. **The share that does not depend on N at all.** `Plants.forage_field()` rebuilds the
     cost-aware diffusion once per tick regardless of how many options each animal weighs, so if it
     dominates, N is close to free and the issue's worry is aimed at the wrong term.

Against 2.1's budget: one tick is one sim-minute and the live rate is one tick per real second, so
the whole tick has ~1000 ms at whatever population the world holds.

Usage:
    python docs/spikes/option_sampling_bench.py
"""

from __future__ import annotations

import statistics
import time

import numpy as np

from core.behaviour.drives import (
    Fatigue,
    FatigueConfig,
    Hunger,
    HungerConfig,
    Lust,
    LustConfig,
    Thirst,
    ThirstConfig,
)
from core.behaviour.exertion import Exertion, ExertionConfig
from core.behaviour.service import Behaviour, BehaviourConfig
from core.ecology.metabolism import Metabolism, MetabolismConfig
from core.ecology.plants import Plants, PlantsConfig
from core.ecology.service import Ecology
from core.entities.store import EntityStore
from core.genetics.expression import GeneticsConfig
from core.genetics.registry import ExpressionMode, GeneRegistry, GeneSpec, Unit
from core.genetics.service import Genetics
from core.genetics.species import SpeciesRegistry
from core.selection import Selection
from core.services import ColumnRegistry
from core.world.climate import Climate, ClimateConfig
from core.world.diffusion import DiffusionConfig
from core.world.terrain import Terrain
from core.world.water import Water

SIZES = (1_000, 10_000, 100_000)
CANDIDATE_COUNTS = (4, 8, 16, 32)
REPEATS = 7
GRID = 256  # cells per side; held fixed so the field cost is constant across population sizes


def _spec(
    name: str, cost: float = 0.0, mode: ExpressionMode = ExpressionMode.MAGNITUDE
) -> GeneSpec:
    return GeneSpec(
        name=name, cost=cost, expression_mode=mode, unit=Unit.DIMENSIONLESS, description=name
    )


GENES = (
    _spec("size", cost=0.01),
    _spec("sight", cost=0.01),
    # A gene that only ever reduces upkeep must charge a positive cost, or it is a free lunch --
    # Metabolism refuses to be built otherwise (#136).
    _spec("insulation", cost=0.01),
    _spec("mutability"),
    _spec("choice_temperature", mode=ExpressionMode.EXPONENTIAL),
    _spec("commitment"),
)
GENE_NAMES = tuple(gene.name for gene in GENES)


class Bench:
    """A world of `n` entities scattered over a fixed grid, with four of the five drives.

    Fear is left out deliberately: its cost is the cue field, which #22 already benchmarked and
    which #114 does not change -- its `appeal` is a constant. Including it would measure #22.
    """

    def __init__(self, n: int, n_candidates: int) -> None:
        self.store = EntityStore(initial_capacity=n, n_drives=5, n_genes=len(GENES))
        columns = ColumnRegistry()
        genes = GeneRegistry(GENES)
        species = SpeciesRegistry(genes.vocabulary)
        species_id = species.register(GENE_NAMES)
        self.genetics = Genetics(
            self.store, columns, species, genes,
            GeneticsConfig(mutability_gene="mutability", drift_margin=2.0),
        )
        heights = np.zeros((GRID, GRID), dtype=np.float32)
        terrain = Terrain(heights, cell_size=1.0)
        climate = Climate(
            terrain, ClimateConfig(equator_y=0.0, equator_temperature=20.0, latitude_gradient=0.0)
        )
        water = Water(
            depth=np.zeros((GRID, GRID), dtype=np.float32),
            flow_direction=np.full((GRID, GRID), -1, dtype=np.int8),
            flow_accumulation=np.ones((GRID, GRID), dtype=np.float32),
            cell_size=1.0,
        )
        self.plants = Plants(
            terrain, climate, water,
            PlantsConfig(
                solar_constant=1.0, latitude_tilt=0.0, min_growth_temperature=0.0,
                optimal_growth_temperature=20.0, max_growth_temperature=40.0,
                nutrient_per_biomass=1.0, initial_soil_nutrients=100.0, senescence_rate=0.01,
                saturation_accumulation=10.0, max_rooting_depth=1.0,
                forage_diffusion=DiffusionConfig(range=8.0, climb_penalty=0.5),
            ),
        )
        ecology = Ecology(
            self.store, columns, self.genetics, climate,
            Metabolism(genes, MetabolismConfig(
                basal_rate=1.0, thermoregulation_rate=0.5, neutral_temperature=20.0,
                insulation_gene="insulation",
            )),
        )
        exertion = Exertion(self.store, columns, ExertionConfig(recovery_rate=0.5))
        self.behaviour = Behaviour(
            self.store, columns, self.genetics, genes, terrain,
            BehaviourConfig(
                n_candidates=n_candidates, look_ahead=4.0, commitment_gene="commitment",
                choice_temperature_gene="choice_temperature",
            ),
        )
        self.hunger = Hunger(
            self.store, ecology, self.genetics, self.plants, genes,
            HungerConfig(
                weight=1.0, satiation_energy=100.0, detection_threshold=0.1, sight_gene="sight"
            ),
        )
        self.behaviour.register(self.hunger)
        self.behaviour.register(Thirst(
            self.store, climate,
            ThirstConfig(weight=1.0, onset_temperature=25.0, saturation_temperature=40.0),
        ))
        self.behaviour.register(Lust(
            self.store, ecology,
            LustConfig(
                weight=1.0, maturity_age=100, breeding_energy=20.0, abundant_energy=70.0
            ),
        ))
        self.behaviour.register(Fatigue(
            self.store, exertion, FatigueConfig(weight=1.0, exertion_saturation=1.0)
        ))

        rng = np.random.default_rng(0)
        side = terrain.world_width
        ids = self.store.allocate(
            n,
            x=rng.uniform(0.0, side, n).astype(np.float32),
            y=rng.uniform(0.0, side, n).astype(np.float32),
            energy=rng.uniform(0.0, 100.0, n).astype(np.float32),
            health=np.ones(n, dtype=np.float32),
            species_id=np.full(n, species_id, dtype=np.int32),
        )
        rows = np.array([self.store._id_to_row[i] for i in ids.tolist()], dtype=np.int64)
        self.population = Selection.from_indices(rows, self.store.capacity)
        matrix = np.zeros((n, len(GENES)), dtype=np.float32)
        matrix[:, GENE_NAMES.index("sight")] = 1.0
        self.genetics.set_genes(self.population, matrix)
        self.rng = np.random.default_rng(1)


def timed(fn, repeats: int = REPEATS) -> float:
    """Median wall-clock milliseconds over `repeats` runs, after one warmup."""
    fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def main() -> None:
    print(f"grid {GRID}x{GRID}, {REPEATS} timed runs each, median ms\n")

    print("== whole choose() call, ms/tick ==")
    header = "     n | " + " | ".join(f"N={n:>2}" for n in CANDIDATE_COUNTS)
    print(header)
    print("-" * len(header))
    whole: dict[tuple[int, int], float] = {}
    for n in SIZES:
        cells = []
        for n_candidates in CANDIDATE_COUNTS:
            bench = Bench(n, n_candidates)
            ms = timed(lambda b=bench: b.behaviour.choose(b.population, b.rng))
            whole[(n, n_candidates)] = ms
            cells.append(f"{ms:>7.1f}")
        print(f"{n:>6} | " + " | ".join(cells))

    print("\n== where the time goes at N=8 (9 options), ms/tick ==")
    print("  diffusion: Plants.forage_field(), rebuilt once per tick, independent of N")
    print("  reads:     forage_at() over the whole (n, 9) option block -- the N-scaled term")
    print("  flat:      the three drives whose appeal is a constant (thirst, lust, fatigue)")
    columns = ("diffusion", "positions", "reads", "flat", "draw", "measured total", "choose()")
    print("     n | " + " | ".join(f"{c:>14}" for c in columns))
    print("-" * (8 + 17 * len(columns)))
    for n in SIZES:
        bench = Bench(n, 8)
        population, rng = bench.population, bench.rng

        diffusion_ms = timed(lambda b=bench: b.plants.forage_field())
        positions_ms = timed(
            lambda b=bench: b.behaviour.candidate_positions(
                population, b.behaviour.candidate_headings(population, rng)
            )
        )
        headings = bench.behaviour.candidate_headings(population, rng)
        x, y = bench.behaviour.candidate_positions(population, headings)
        field = bench.plants.forage_field()
        reads_ms = timed(lambda b=bench: b.plants.forage_at(field, x, y))
        flat = [d for d in bench.behaviour._drives if d.name != "hunger"]
        flat_ms = timed(
            lambda: [(d.urgency(population), d.appeal(population, x, y)) for d in flat]
        )
        draw_ms = timed(
            lambda b=bench: np.argmax(
                np.zeros((len(population), 9)) + rng.gumbel(size=(len(population), 9)), axis=1
            )
        )
        measured = diffusion_ms + positions_ms + reads_ms + flat_ms + draw_ms
        cells = (diffusion_ms, positions_ms, reads_ms, flat_ms, draw_ms, measured, whole[(n, 8)])
        print(f"{n:>6} | " + " | ".join(f"{c:>14.1f}" for c in cells))

    print("\n== cost of one candidate, ms per 1,000 entities ==")
    print("  The slope in N. A flat slope means N is a knob that can be turned up.")
    for n in SIZES:
        slope = (whole[(n, 32)] - whole[(n, 4)]) / (32 - 4) / (n / 1000.0)
        base = whole[(n, 4)] / (n / 1000.0)
        print(f"{n:>6} | per-candidate {slope:>6.3f} ms/1k | fixed floor at N=4 {base:>6.3f} ms/1k")


if __name__ == "__main__":
    main()
