"""How small can a population get and still come back? (#29)

Throwaway spike code (CLAUDE.md §8.3), not part of `core/` and imported by nothing. It produces the
numbers in `docs/spikes/viable-population.md`, which #29's collapse detection and #28's cull guard
read. Run it from a repository root (or worktree) with `PYTHONPATH=.`:

    python docs/spikes/viable_population.py establish --out <dir>
    python docs/spikes/viable_population.py sweep --out <dir> --mode bottleneck --reps 40
    python docs/spikes/viable_population.py sweep --out <dir> --mode naive --reps 40
    python docs/spikes/viable_population.py sweep --out <dir> --mode naive --grid 160 --reps 20

**Two ways a population gets small, measured separately because they are different questions.**

- `bottleneck`: an established, already-selected population is knocked down to `n` born animals,
  chosen at random, as a cull or a die-off would. Every pregnancy is lost with them. This is the
  situation #29 and #28 actually ask about.
- `naive`: `n` founders drawn from the demo world's founder ranges, as `build_demo_world` does.
  Founders are unselected, and a large share die before breeding (the docs measure ~40% at
  n=200). So this curve mixes demographic risk with founder maladaptation, and it is reported to
  show how big that second effect is, not as the threshold.

Bodies removed by a bottleneck go back into the soil at the animal's position, exactly as
`core.world.cull.Cull.apply` does. The victims are chosen at random rather than by row order,
which `Cull` uses and which correlates with allocation time.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from clients.viewer.demo_world import build_demo_world, demo_world_config
from core.world.assembly import build_world
from persistence import snapshot

# The founding size the demo world is built at, and the size its source worlds are grown from.
ESTABLISH_FOUNDERS = 200
# Ticks to grow a source world before it is snapshotted. Measured on seed 1: the founder die-off
# bottoms out near tick 300 and the population passes 1,500 by tick 1,400, so 1,500 is well into
# sustained growth while still far below the ~3,000 the world was still climbing past at tick 3,000.
ESTABLISH_TICKS = 1_500
SOURCE_SEEDS = (1, 2, 3)

# A run is RECOVERED once the born population reaches this size. 200 is the demo world's own
# founding size, and the sweep includes n=200 bottlenecks, so the choice is checked rather than
# assumed: if a population of 200 itself sometimes dies out, this definition is too low.
RECOVERY_SIZE = 200
# A run is UNRESOLVED if it neither recovers nor dies out within this many generations.
HORIZON_GENERATIONS = 10

N_GRID = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 100, 200)


def _config(n: int, seed: int, grid: int):
    """The demo world's config, with the grid resized for the density check."""
    config = demo_world_config(n, seed)
    if grid == config.terrain.width:
        return config
    extent = (grid - 1) * config.terrain.cell_size
    terrain = dataclasses.replace(
        config.terrain, width=grid, height=grid, max_elevation=extent / 10.0
    )
    climate = dataclasses.replace(config.climate, equator_y=extent / 2.0)
    return dataclasses.replace(config, terrain=terrain, climate=climate)


def _born(world) -> np.ndarray:
    store = world.store
    return store.alive & (store.age >= 0)


def generation_ticks(world) -> float:
    """Mean expressed `maturity_age + gestation_length` over the living: the shortest a lineage can
    turn over. A floor on generation time rather than its mean, since real parents breed after
    maturity rather than at it. That makes the horizon built on it conservative (shorter)."""
    names = [gene.name for gene in world.config.genes]
    rows = np.flatnonzero(_born(world))
    phenotype = world.genetics.expressed_at(rows)
    maturity = phenotype[:, names.index("maturity_age")]
    gestation = phenotype[:, names.index("gestation_length")]
    return float(maturity.mean() + gestation.mean())


def establish(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for seed in SOURCE_SEEDS:
        start = time.perf_counter()
        world = build_demo_world(seed=seed, n_entities=ESTABLISH_FOUNDERS)
        world.loop.advance(ESTABLISH_TICKS)
        snapshot.save(world, out / f"source_{seed}.npz")
        summary[seed] = {
            "born": int(_born(world).sum()),
            "generation_ticks": generation_ticks(world),
            "wall_s": time.perf_counter() - start,
        }
        print(seed, summary[seed], flush=True)
    (out / "sources.json").write_text(json.dumps(summary, indent=2))


def _bottleneck(world, n: int, rng: np.random.Generator) -> None:
    """Keep `n` born animals at random, and release everything else alive, unborn included."""
    store = world.store
    born = np.flatnonzero(_born(world))
    keep = rng.choice(born, size=min(n, born.size), replace=False)
    doomed = store.alive.copy()
    doomed[keep] = False
    rows = np.flatnonzero(doomed)
    # Same two calls, in the same order, as `Cull.apply`: return each body to the ground it lies
    # on, then free the rows.
    world.plants.return_nutrients(
        store.x[rows].astype(np.float64),
        store.y[rows].astype(np.float64),
        store.energy[rows].astype(np.float64),
    )
    store.release(store.row_ids()[rows])


def _run(job: dict) -> dict:
    mode, seed, n, rep, grid, horizon = (
        job["mode"],
        job["seed"],
        job["n"],
        job["rep"],
        job["grid"],
        job["horizon"],
    )
    start = time.perf_counter()
    rep_seed = 1_000_003 * seed + 7_919 * n + rep
    if mode == "bottleneck":
        world = build_demo_world(seed=seed, n_entities=ESTABLISH_FOUNDERS)
        snapshot.load(world, Path(job["source"]))
        world.rng.bit_generator.state = np.random.default_rng(rep_seed).bit_generator.state
        _bottleneck(world, n, np.random.default_rng(rep_seed + 1))
    else:
        world = build_world(_config(n, rep_seed, grid), seed=rep_seed)

    outcome, ticks, peak = "unresolved", horizon, n
    step = 10
    for elapsed in range(step, horizon + step, step):
        try:
            world.loop.advance(step)
        except ValueError as error:
            # #240: a world starving out raises from `Plants.return_nutrients` once float32
            # rounding in the energy column outruns a nearly empty export ledger. By then the whole
            # population holds well under one energy unit, so it is starving out. Recorded as its
            # own outcome rather than folded into `extinct`, so the count stays visible.
            if "cannot return more nutrients" not in str(error):
                raise
            outcome, ticks = "crashed", elapsed
            break
        born = int(_born(world).sum())
        peak = max(peak, born)
        if not world.store.alive.any():
            outcome, ticks = "extinct", elapsed
            break
        if born >= RECOVERY_SIZE:
            outcome, ticks = "recovered", elapsed
            break
    return {
        "mode": mode,
        "seed": seed,
        "grid": grid,
        "n": n,
        "rep": rep,
        "outcome": outcome,
        "ticks": ticks,
        "peak": peak,
        "final": int(_born(world).sum()),
        "wall_s": round(time.perf_counter() - start, 2),
    }


def sweep(
    out: Path,
    mode: str,
    reps: int,
    grid: int,
    ns: tuple[int, ...],
    workers: int,
    generations: int = HORIZON_GENERATIONS,
) -> None:
    sources = json.loads((out / "sources.json").read_text())
    generation = float(np.mean([s["generation_ticks"] for s in sources.values()]))
    horizon = int(round(generations * generation, -1))
    jobs = [
        {
            "mode": mode,
            "seed": seed,
            "n": n,
            "rep": rep,
            "grid": grid,
            "horizon": horizon,
            "source": str(out / f"source_{seed}.npz"),
        }
        for n in ns
        for rep in range(reps)
        for seed in (SOURCE_SEEDS[rep % len(SOURCE_SEEDS)],)
    ]
    suffix = "" if generations == HORIZON_GENERATIONS else f"_gen{generations}"
    path = out / f"{mode}_grid{grid}{suffix}.csv"
    start = time.perf_counter()
    print(f"{len(jobs)} runs, horizon {horizon} ticks ({generation:.0f}/generation)", flush=True)
    with path.open("w", newline="") as handle, ProcessPoolExecutor(workers) as pool:
        writer = None
        for i, row in enumerate(pool.map(_run, jobs, chunksize=1)):
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            if i % 20 == 0:
                print(i, row, flush=True)
    print(f"wall {time.perf_counter() - start:.0f}s -> {path}", flush=True)


def summarise(out: Path) -> None:
    """P(recovery) per n with a Wilson 95% interval, for every sweep file present."""
    for path in sorted(out.glob("*_grid*.csv")):
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        print(f"\n{path.name}")
        print(
            "   n  runs  recovered  extinct  crashed  stalled  births?  P(recover)  95% CI"
            "        median ticks"
        )
        for n in sorted({int(r["n"]) for r in rows}):
            group = [r for r in rows if int(r["n"]) == n]
            k = sum(r["outcome"] == "recovered" for r in group)
            ext = sum(r["outcome"] == "extinct" for r in group)
            crashed = sum(r["outcome"] == "crashed" for r in group)
            stalled = [r for r in group if r["outcome"] == "unresolved"]
            # A stalled run whose peak exceeded its start bred at least once.
            bred = sum(int(r["peak"]) > n for r in stalled)
            low, high = _wilson(k, len(group))
            ticks = np.median([int(r["ticks"]) for r in group])
            print(
                f"{n:>4} {len(group):>5} {k:>10} {ext:>8} {crashed:>8} {len(stalled):>8}"
                f" {bred:>8}  {k / len(group):>10.2f}  [{low:.2f}, {high:.2f}]  {ticks:>12.0f}"
            )


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("establish", "sweep", "summarise"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("bottleneck", "naive"), default="bottleneck")
    parser.add_argument("--reps", type=int, default=40)
    parser.add_argument("--grid", type=int, default=80)
    parser.add_argument("--ns", type=str, default=",".join(map(str, N_GRID)))
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--generations", type=int, default=HORIZON_GENERATIONS)
    args = parser.parse_args()
    if args.command == "establish":
        establish(args.out)
    elif args.command == "sweep":
        ns = tuple(int(v) for v in args.ns.split(","))
        sweep(args.out, args.mode, args.reps, args.grid, ns, args.workers, args.generations)
    else:
        summarise(args.out)


if __name__ == "__main__":
    main()
