"""Spike for #258 (and #174): what a whole tick costs at 10k and 100k, system by system.

Throwaway spike code, not part of the simulation core (CLAUDE.md §8.3). It produces the numbers
docs/spikes/tick-budget.md is built from and nothing should import it.

Until now #258 measured only `Behaviour.choose`. §2.1's budget is the *whole* tick (1,000 ms at the
live rate), so this times every system in `TICK_ORDER`, on a world built the way the game builds one
(`build_world` over `demo_world_config`). Nothing in `core/` is changed: the loop's `systems` tuple is
replaced on the instance with timing wrappers in the same order, and `Genetics`' read methods are
wrapped on the instance for the phenotype census.

Three modes:

    systems    per-system median ms over the measured ticks, plus the live population
    series     whole-tick ms against the acting population, tick by tick
    profile    cProfile over the measured ticks; `Terrain.cell_indices` calls and share
    phenotype  every phenotype read per tick, by system and by caller, with time; and the cost of
               one whole-block build to compare against

Usage (from the repository root):
    python docs/spikes/tick_budget.py systems   --n 100000 --grid 256
    python docs/spikes/tick_budget.py profile   --n 100000 --grid 256
    python docs/spikes/tick_budget.py phenotype --n 100000 --grid 256

`--grid 0` keeps the demo world's own size (80 x 80). The option-sampling bench uses 256 x 256.
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import replace

import numpy as np

from clients.viewer.demo_world import demo_world_config
from core.selection import Selection
from core.world.assembly import TICK_ORDER, World, build_world

BUDGET_MS = 1_000.0


def build(n: int, grid: int, seed: int) -> World:
    config = demo_world_config(n_entities=n, seed=seed)
    if grid:
        config = replace(config, terrain=replace(config.terrain, width=grid, height=grid))
    return build_world(config, seed=seed)


def living_count(world: World) -> tuple[int, int]:
    """(acting, allocated): born animals, and every allocated row including the gestating."""
    store = world.store
    return int((store.alive & (store.age >= 0)).sum()), int(store.alive.sum())


class SystemClock:
    """Replaces the loop's systems with wrappers that time each one and note which is running."""

    def __init__(self, world: World) -> None:
        self.samples: dict[str, list[float]] = defaultdict(list)
        self.current: str | None = None
        self.recording = False
        systems = world.loop.systems
        assert len(systems) == len(TICK_ORDER)
        world.loop.systems = tuple(
            self._wrap(name, system) for name, system in zip(TICK_ORDER, systems)
        )

    def _wrap(self, name, system):
        def timed() -> None:
            self.current = name
            start = time.perf_counter()
            system()
            elapsed = (time.perf_counter() - start) * 1000.0
            self.current = None
            if self.recording:
                self.samples[name].append(elapsed)

        return timed


def run_systems(args: argparse.Namespace) -> None:
    """Per-system medians, pooled over `--rebuilds` fresh worlds (seeds seed, seed+1, ...).

    Fresh worlds rather than one long run because no map holds 100,000 animals: the founders
    starve from about tick 6, so only the first handful of ticks are a tick *at* the size asked
    for. Pooling early ticks of several worlds is what gives >= 20 samples at that size.
    """
    samples: dict[str, list[float]] = defaultdict(list)
    whole, acting = [], []
    for i in range(args.rebuilds):
        world = build(args.n, args.grid, args.seed + i)
        clock = SystemClock(world)
        world.loop.advance(args.warmup)
        clock.recording = True
        for _ in range(args.ticks):
            acting.append(living_count(world)[0])
            start = time.perf_counter()
            world.loop.advance(1)
            whole.append((time.perf_counter() - start) * 1000.0)
        for name, values in clock.samples.items():
            samples[name].extend(values)
    print(f"n={args.n} grid={world.terrain.heights.shape} seeds {args.seed}.."
          f"{args.seed + args.rebuilds - 1}, ticks {args.warmup}..{args.warmup + args.ticks - 1} "
          f"of each ({len(whole)} ticks)")
    print(f"acting at tick start: min {min(acting)} median {int(statistics.median(acting))} "
          f"max {max(acting)}")
    print(f"{'system':>22} | {'median':>8} | {'p10':>8} | {'p90':>8} | share")
    medians = {name: statistics.median(samples[name]) for name in TICK_ORDER}
    total = sum(medians.values())
    for name in TICK_ORDER:
        s = sorted(samples[name])
        p10, p90 = s[len(s) // 10], s[(9 * len(s)) // 10]
        print(f"{name:>22} | {medians[name]:8.1f} | {p10:8.1f} | {p90:8.1f} | "
              f"{100 * medians[name] / total:5.1f}%")
    print(f"{'sum of medians':>22} | {total:8.1f} | {100 * total / BUDGET_MS:.0f}% of budget")
    print(f"{'whole advance(1)':>22} | {statistics.median(whole):8.1f} | "
          f"{min(whole):8.1f} | {max(whole):8.1f}  (median, min, max)")


def run_series(args: argparse.Namespace) -> None:
    """Whole-tick ms against the acting population, tick by tick, as the founders starve.

    Not a budget measurement: it shows how tick cost tracks n, which the per-system table cannot
    because it holds n fixed.
    """
    world = build(args.n, args.grid, args.seed)
    print(f"n={args.n} grid={world.terrain.heights.shape} seed={args.seed}")
    print("tick | acting | ms")
    for tick in range(args.ticks):
        acting = living_count(world)[0]
        start = time.perf_counter()
        world.loop.advance(1)
        print(f"{tick:>4} | {acting:>6} | {(time.perf_counter() - start) * 1000.0:7.1f}")


def run_profile(args: argparse.Namespace) -> None:
    world = build(args.n, args.grid, args.seed)
    world.loop.advance(args.warmup)

    # First without the profiler, which inflates small calls: wrap `cell_indices` on the instance
    # and attribute each call to the system running it.
    clock = SystemClock(world)
    clock.recording = True
    terrain = world.terrain
    cells: dict[str, list[tuple[float, int]]] = defaultdict(list)
    original = terrain.cell_indices

    def counted(x, y):
        start = time.perf_counter()
        result = original(x, y)
        cells[clock.current or "(outside systems)"].append(
            ((time.perf_counter() - start) * 1000.0, int(np.size(x)))
        )
        return result

    terrain.cell_indices = counted
    start = time.perf_counter()
    world.loop.advance(args.ticks)
    wall = (time.perf_counter() - start) * 1000.0 / args.ticks
    terrain.cell_indices = original
    calls = sum(len(v) for v in cells.values()) / args.ticks
    ms = sum(t for v in cells.values() for t, _ in v) / args.ticks
    print(f"unprofiled, {args.ticks} ticks: tick {wall:.1f} ms; Terrain.cell_indices "
          f"{calls:.1f} calls/tick, {ms:.1f} ms/tick ({100 * ms / wall:.1f}% of the tick)")
    for system in TICK_ORDER:
        if system in cells:
            v = cells[system]
            print(f"  {system:>22}: {len(v) / args.ticks:5.1f} calls/tick, "
                  f"{sum(t for t, _ in v) / args.ticks:6.1f} ms/tick, "
                  f"median {int(statistics.median(e for _, e in v))} positions per call")
    world.loop.systems = tuple(  # unwrap, so the profile below sees the systems themselves
        getattr(s, "__wrapped__", s) for s in world.loop.systems
    )

    profiler = cProfile.Profile()
    profiler.enable()
    world.loop.advance(args.ticks)
    profiler.disable()
    stats = pstats.Stats(profiler)
    total = stats.total_tt
    print(f"n={args.n} grid={world.terrain.heights.shape}, {args.ticks} ticks profiled, "
          f"acting at end {living_count(world)[0]}")
    print(f"profiled total {1000 * total / args.ticks:.1f} ms/tick (inflated by cProfile)")
    for (filename, _, function), (cc, nc, tt, ct, callers) in stats.stats.items():
        if function == "cell_indices" and filename.endswith("terrain.py"):
            print(f"Terrain.cell_indices: {nc / args.ticks:.1f} calls/tick, "
                  f"{1000 * ct / args.ticks:.1f} ms/tick cumulative "
                  f"({100 * ct / total:.1f}% of profiled time)")
            print("  by caller (calls/tick, cumulative ms/tick):")
            for (cf, _, cfn), (_, c_nc, _, c_ct) in sorted(
                callers.items(), key=lambda kv: -kv[1][3]
            ):
                print(f"    {cf.split('core')[-1]}:{cfn:<28} {c_nc / args.ticks:5.1f} "
                      f"{1000 * c_ct / args.ticks:8.1f}")
    if args.top:
        stats.sort_stats("cumulative").print_stats(args.top)


# Public reads of `Genetics` that build or slice a phenotype or touch genes / species ids.
GENETICS_READS = (
    "expressed", "expressed_at", "expressed_columns", "expressed_columns_at", "heritable",
    "heritable_at",
    "vigour", "senescence_decay_at", "genes", "genes_at", "species_ids", "species_ids_at",
    "inherit", "set_genes", "speciate",
)


class PhenotypeCensus:
    """Wraps `Genetics`' reads on the instance and records each outermost call.

    Outermost only: `expressed` calls `expressed_at` and `vigour` calls `_remaining`, and counting
    the inner call too would double the time.
    """

    def __init__(self, world: World, clock: SystemClock) -> None:
        self.clock = clock
        self.depth = 0
        self.recording = False
        # (system, method, caller) -> list of (ms, rows, columns)
        self.calls: dict[tuple[str, str, str], list[tuple[float, int, int]]] = defaultdict(list)
        genetics = world.genetics
        n_genes = world.store.genes.shape[1]
        # `hasattr`, so the same census runs against a revision older than a read (#174's
        # before-and-after: `expressed_columns_at` is new there).
        for name in GENETICS_READS:
            if hasattr(genetics, name):
                setattr(genetics, name, self._wrap(name, getattr(genetics, name), n_genes))
        expression = genetics.expression
        expression.phenotype = self._wrap("expression.phenotype", expression.phenotype, n_genes)

    def _wrap(self, method, bound, n_genes):
        def wrapped(*a, **kw):
            self.depth += 1
            start = time.perf_counter()
            try:
                result = bound(*a, **kw)
            finally:
                self.depth -= 1
            elapsed = (time.perf_counter() - start) * 1000.0
            if self.depth == 0 and self.recording:
                frame = sys._getframe(1)
                owner = frame.f_locals.get("self")
                owner = type(owner).__name__ + "." if owner is not None else ""
                caller = f"{frame.f_code.co_filename.split('core')[-1][1:]}:{owner}" \
                         f"{frame.f_code.co_name}"
                rows = int(result.shape[0]) if hasattr(result, "shape") and result.ndim else 0
                cols = (int(result.shape[1]) if hasattr(result, "shape") and result.ndim == 2
                        else 1)
                system = self.clock.current or "(outside systems)"
                self.calls[(system, method, caller)].append((elapsed, rows, cols))
            return result

        return wrapped


def run_phenotype(args: argparse.Namespace) -> None:
    world = build(args.n, args.grid, args.seed)
    clock = SystemClock(world)
    census = PhenotypeCensus(world, clock)
    world.loop.advance(args.warmup)
    census.recording = True
    acting = []
    for _ in range(args.ticks):
        acting.append(living_count(world)[0])
        world.loop.advance(1)
    census.recording = False
    ticks = args.ticks
    print(f"n={args.n} grid={world.terrain.heights.shape}, {ticks} ticks, acting median "
          f"{int(statistics.median(acting))}")
    print(f"{'system':>20} | {'method':>20} | {'caller':<44} | calls/tick | rows | cols | "
          f"ms/tick")
    total = 0.0
    by_system: dict[str, float] = defaultdict(float)
    order = {name: i for i, name in enumerate(TICK_ORDER)}
    for (system, method, caller), samples in sorted(
        census.calls.items(), key=lambda kv: (order.get(kv[0][0], 99), -sum(s[0] for s in kv[1]))
    ):
        ms = sum(s[0] for s in samples) / ticks
        total += ms
        by_system[system] += ms
        rows = int(statistics.median(s[1] for s in samples))
        cols = int(statistics.median(s[2] for s in samples))
        print(f"{system:>20} | {method:>20} | {caller:<44} | {len(samples) / ticks:10.1f} | "
              f"{rows:>6} | {cols:>4} | {ms:7.1f}")
    print(f"total Genetics reads: {total:.1f} ms/tick")
    for system, ms in sorted(by_system.items(), key=lambda kv: order.get(kv[0], 99)):
        print(f"  {system:>20}: {ms:7.1f} ms/tick")

    # What one whole-block build costs, for the living at the end of the run: the price a
    # once-per-tick block would pay in place of the reads above, before its slicing costs.
    living = Selection.from_mask(world.store.alive & (world.store.age >= 0))
    genetics = world.genetics

    def median_ms(fn, repeats=7):
        fn()
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - start) * 1000.0)
        return statistics.median(samples)

    block = genetics.expressed(living)
    print(f"\none expressed(living), {len(living)} rows x {block.shape[1]} genes: "
          f"{median_ms(lambda: genetics.expressed(living)):.1f} ms")
    print(f"one heritable(living): {median_ms(lambda: genetics.heritable(living)):.1f} ms")
    print(f"one vigour(living):    {median_ms(lambda: genetics.vigour(living)):.1f} ms")
    for k in (1, 2, 4, 8, 17):
        cols = np.arange(k)
        print(f"slice block[:, {k} cols]: {median_ms(lambda c=cols: block[:, c]):.2f} ms")
    rows = np.sort(np.random.default_rng(0).choice(len(living), size=len(living) // 2,
                                                   replace=False))
    print(f"gather block[half the rows]: {median_ms(lambda: block[rows]):.2f} ms")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("systems", "series", "profile", "phenotype"))
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--grid", type=int, default=256, help="cells per side; 0 = demo's own")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--ticks", type=int, default=8)
    parser.add_argument("--rebuilds", type=int, default=3, help="systems mode: fresh worlds")
    parser.add_argument("--top", type=int, default=0, help="also print the top-N cumulative rows")
    args = parser.parse_args()
    {"systems": run_systems, "series": run_series, "profile": run_profile, "phenotype": run_phenotype}[args.mode](args)


if __name__ == "__main__":
    main()
