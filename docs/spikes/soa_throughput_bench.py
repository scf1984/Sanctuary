"""Benchmark harness for issue #1: measure SoA throughput and validate the catch-up budget.

Throwaway spike code -- not part of the simulation core (CLAUDE.md 8.3). It exists to produce
the numbers docs/spikes/soa-throughput.md is built from, and should not be imported by anything
else.

Prototypes one representative tick five ways the real core would touch every entity per tick
(CLAUDE.md 2.3):
  - position integration
  - a spatial-hash neighbour lookup (stand-in for InteractionGrid, CLAUDE.md 1)
  - an energy upkeep decrement
  - a threshold comparison (starving)
  - a masked selection applying the consequence of that comparison

It runs the same workload twice -- once over global NumPy arrays (structure-of-arrays), once
over a plain Python list of objects -- at several population sizes, plus a doubling-copy growth
cost at each size, and prints a markdown table.

Usage:
    python3 docs/spikes/soa_throughput_bench.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

SIZES = (1_000, 5_000, 20_000, 100_000)
SOA_TICKS = 200
PY_TICKS = 20  # plain-Python is orders of magnitude slower; fewer ticks keeps this runnable
DT = 1.0  # one sim-minute, CLAUDE.md 2.1
WORLD_SIZE = 1_000.0
CELL_SIZE = 20.0  # spatial hash cell size, see InteractionGrid, CLAUDE.md 1
METABOLISM_UPKEEP = 0.05  # energy/tick baseline
CROWDING_UPKEEP = 0.01  # additional energy/tick per same-cell neighbour
STARTING_ENERGY = 100.0
SEED = 0


# ---------------------------------------------------------------------------
# Structure-of-arrays (NumPy) representative tick
# ---------------------------------------------------------------------------

@dataclass
class SoaPopulation:
    x: np.ndarray  # (n,) float32, world units
    y: np.ndarray  # (n,) float32, world units
    vx: np.ndarray  # (n,) float32, world units / tick
    vy: np.ndarray  # (n,) float32, world units / tick
    energy: np.ndarray  # (n,) float32, joules

    @classmethod
    def random(cls, n, rng):
        return cls(
            x=rng.uniform(0, WORLD_SIZE, n).astype(np.float32),
            y=rng.uniform(0, WORLD_SIZE, n).astype(np.float32),
            vx=rng.uniform(-1, 1, n).astype(np.float32),
            vy=rng.uniform(-1, 1, n).astype(np.float32),
            energy=np.full(n, STARTING_ENERGY, dtype=np.float32),
        )


def soa_neighbour_counts(x, y, cell_size):
    """Grid-hash neighbour lookup: same-cell occupant count per entity."""
    cell_x = np.floor(x / cell_size).astype(np.int64)
    cell_y = np.floor(y / cell_size).astype(np.int64)
    keys = cell_x * 1_000_003 + cell_y  # arbitrary large prime-ish multiplier to spread hashes
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    _, counts = np.unique(sorted_keys, return_counts=True)
    counts_per_entity = np.empty_like(keys)
    counts_per_entity[order] = np.repeat(counts, counts)
    return counts_per_entity


def soa_tick(pop: SoaPopulation, dt: float) -> int:
    # Position integration.
    pop.x += pop.vx * dt
    pop.y += pop.vy * dt
    np.clip(pop.x, 0, WORLD_SIZE, out=pop.x)
    np.clip(pop.y, 0, WORLD_SIZE, out=pop.y)

    # Neighbour lookup.
    neighbours = soa_neighbour_counts(pop.x, pop.y, CELL_SIZE)

    # Energy decrement: crowding raises upkeep, standing in for contested foraging.
    pop.energy -= METABOLISM_UPKEEP * (1.0 + CROWDING_UPKEEP * neighbours)

    # Threshold comparison + masked selection: starving entities are refed, standing in for
    # the mortality/replacement a real tick would apply via the free list (CLAUDE.md 2.3).
    starving = pop.energy <= 0
    pop.energy[starving] = STARTING_ENERGY
    return int(starving.sum())


def measure_growth_cost(pop: SoaPopulation) -> float:
    """Cost of doubling capacity by copy, the mitigation in CLAUDE.md 2.3 item 1."""
    n = pop.x.shape[0]
    start = time.perf_counter()
    SoaPopulation(
        x=np.concatenate([pop.x, np.zeros(n, dtype=pop.x.dtype)]),
        y=np.concatenate([pop.y, np.zeros(n, dtype=pop.y.dtype)]),
        vx=np.concatenate([pop.vx, np.zeros(n, dtype=pop.vx.dtype)]),
        vy=np.concatenate([pop.vy, np.zeros(n, dtype=pop.vy.dtype)]),
        energy=np.concatenate([pop.energy, np.zeros(n, dtype=pop.energy.dtype)]),
    )
    return time.perf_counter() - start


# ---------------------------------------------------------------------------
# Plain Python objects, same workload, for the ratio
# ---------------------------------------------------------------------------

class PyEntity:
    __slots__ = ("x", "y", "vx", "vy", "energy")

    def __init__(self, x, y, vx, vy, energy):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.energy = energy


def py_population(n, rng):
    return [
        PyEntity(
            x=rng.uniform(0, WORLD_SIZE),
            y=rng.uniform(0, WORLD_SIZE),
            vx=rng.uniform(-1, 1),
            vy=rng.uniform(-1, 1),
            energy=STARTING_ENERGY,
        )
        for _ in range(n)
    ]


def py_tick(entities, dt) -> int:
    grid: dict[tuple[int, int], list[PyEntity]] = {}
    for e in entities:
        e.x = min(max(e.x + e.vx * dt, 0.0), WORLD_SIZE)
        e.y = min(max(e.y + e.vy * dt, 0.0), WORLD_SIZE)
        cell = (int(e.x // CELL_SIZE), int(e.y // CELL_SIZE))
        grid.setdefault(cell, []).append(e)

    starving = 0
    for e in entities:
        cell = (int(e.x // CELL_SIZE), int(e.y // CELL_SIZE))
        neighbours = len(grid[cell])
        e.energy -= METABOLISM_UPKEEP * (1.0 + CROWDING_UPKEEP * neighbours)
        if e.energy <= 0:
            e.energy = STARTING_ENERGY
            starving += 1
    return starving


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

def time_soa(n, rng):
    pop = SoaPopulation.random(n, rng)
    for _ in range(5):  # warmup: first touches page in the arrays
        soa_tick(pop, DT)
    start = time.perf_counter()
    for _ in range(SOA_TICKS):
        soa_tick(pop, DT)
    elapsed = time.perf_counter() - start
    return elapsed / SOA_TICKS


def time_py(n, rng):
    entities = py_population(n, rng)
    for _ in range(2):
        py_tick(entities, DT)
    start = time.perf_counter()
    for _ in range(PY_TICKS):
        py_tick(entities, DT)
    elapsed = time.perf_counter() - start
    return elapsed / PY_TICKS


SIM_MINUTES_PER_TICK = 1
TICKS_PER_SEVEN_DAY_ABSENCE = 7 * 24 * 60 // SIM_MINUTES_PER_TICK


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for n in SIZES:
        soa_seconds_per_tick = time_soa(n, rng)
        py_seconds_per_tick = time_py(n, rng)
        growth_seconds = measure_growth_cost(SoaPopulation.random(n, rng))

        soa_updates_per_sec = n / soa_seconds_per_tick
        py_updates_per_sec = n / py_seconds_per_tick
        catchup_seconds = TICKS_PER_SEVEN_DAY_ABSENCE * soa_seconds_per_tick

        rows.append(
            dict(
                n=n,
                soa_updates_per_sec=soa_updates_per_sec,
                py_updates_per_sec=py_updates_per_sec,
                ratio=soa_updates_per_sec / py_updates_per_sec,
                growth_ms=growth_seconds * 1000,
                catchup_seconds=catchup_seconds,
            )
        )

    header = (
        f"{'n':>8} | {'SoA updates/s':>15} | {'Python updates/s':>17} | "
        f"{'SoA/Python':>10} | {'growth copy (ms)':>16} | {'7-day catch-up (s)':>19}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['n']:>8} | {r['soa_updates_per_sec']:>15,.0f} | "
            f"{r['py_updates_per_sec']:>17,.0f} | {r['ratio']:>10,.1f} | "
            f"{r['growth_ms']:>16,.2f} | {r['catchup_seconds']:>19,.2f}"
        )


if __name__ == "__main__":
    main()
