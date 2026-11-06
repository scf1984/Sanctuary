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
from dataclasses import replace

import numpy as np

from clients.viewer.demo_world import demo_world_config
from core.world.assembly import build_world

SIZES = (1_000, 10_000, 100_000)
CANDIDATE_COUNTS = (4, 8, 16, 32)
REPEATS = 7
GRID = 256  # cells per side; held fixed so the field cost is constant across population sizes


class Bench:
    """A world of `n` founders on a fixed GRID x GRID map, built the way the game builds one.

    Assembled through `build_world` from the demo world's coefficients rather than wired service by
    service. The hand-wired original drifted out of step with every constructor it called (#257);
    the composition root is the one place that has to stay in step anyway. The cost: all five
    drives are registered, so fear's per-candidate reads are now inside `choose()`. The original
    left fear out to avoid re-measuring #22's cue-field rebuild, which is a separate system and
    is still outside these timings.
    """

    def __init__(self, n: int, n_candidates: int) -> None:
        config = demo_world_config(n_entities=n, seed=0)
        config = replace(
            config,
            terrain=replace(config.terrain, width=GRID, height=GRID),
            behaviour=replace(config.behaviour, n_candidates=n_candidates),
        )
        world = build_world(config, seed=0)
        self.behaviour = world.behaviour
        self.plants = world.plants
        self.population = world.founders
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
    print("  others:    urgency and appeal of the drives other than hunger")
    columns = ("diffusion", "positions", "reads", "others", "draw", "measured total", "choose()")
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
