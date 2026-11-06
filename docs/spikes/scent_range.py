"""How far can a lone animal smell a mate, and what does widening the scent field cost? (#251, #262)

Throwaway spike code (CLAUDE.md §8.3), not part of `core/` and imported by nothing. It produces the
numbers in `docs/spikes/scent-range.md`. Run it from a repository root (or worktree) with
`PYTHONPATH=.`:

    python docs/spikes/scent_range.py profile
    python docs/spikes/scent_range.py cost
    python docs/spikes/viable_population.py establish --out <dir>      # source worlds, once
    python docs/spikes/scent_range.py readers --sources <dir>
    python docs/spikes/scent_range.py pairs --sources <dir> --reps 10
    python docs/spikes/scent_range.py founders

`--configs A,B` restricts any command to some of `CONFIGS`. The configs measure the exponential
kernel #262 introduced. #251's box-blur tables came from this script at 1b57c0e, whose `CONFIGS`
named box-blur configs, and only reproduce from that commit.

**Every config is applied by swapping two knobs on a live world**: the cue field (`diffusion_range`)
and lust's `detection_threshold`. Snapshots refuse a world whose config fingerprint differs (§2.8),
so a source world is loaded under the config it was saved with and then retuned. The cue field is
rebuilt from scratch every tick, so swapping it carries no state across. A source world only loads
into the demo config it was grown under, so `readers` and `pairs` must run with the same
`clients/viewer/demo_world.py` checked out as `establish` did (the doc's tables used d16c681's).
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from clients.viewer.demo_world import build_demo_world, demo_world_config
from core.ecology.cues import CueField, CueFieldConfig, _axis_operator
from core.selection import Selection
from core.world.assembly import build_world
from core.world.terrain import Terrain
from persistence import snapshot

# (diffusion_range, lust detection_threshold), under #262's exponential kernel. The first is
# #251's shipped pair of numbers carried over unchanged. A typical mate (emission 1, own-signature
# dot 2) reads 2c² at its own cell, c = (1 − a) / (1 + a) and a = exp(−1 / range), so
# t = 2c² · exp(−reach / range) sets the reach at acuity 1. The others hold that reach at the 27
# units #251 validated (t3e-4), or between (t1e-4), and at range 7 the kernel's spread matches the
# box blur's at range 10 (per-axis standard deviation √2 · 7 ≈ 9.9 against √110 ≈ 10.5).
CONFIGS = {
    "E10_t5e-5": (10.0, 5e-5),
    "E10_t1e-4": (10.0, 1e-4),
    "E10_t3e-4": (10.0, 3e-4),
    "E7_t5e-5": (7.0, 5e-5),
    "E7_t2e-4": (7.0, 2e-4),
}
SOURCE_SEEDS = (1, 2, 3)
# Ticks a bottlenecked pair is followed for, about 27 founding generations.
PAIR_HORIZON = 3_000
# Two fresh founders this far apart, as in tests/core/ecology/test_mates_find_each_other.py.
FOUNDER_SEPARATION = 30.0
FOUNDER_HORIZON = 2_000


def retune(world, diffusion_range: float, threshold: float) -> None:
    world.scent.field = CueField(
        world.terrain, world.scent.field.n_channels, CueFieldConfig(diffusion_range=diffusion_range)
    )
    lust = _drives(world)["lust"]
    lust.config = dataclasses.replace(lust.config, detection_threshold=threshold)


def _drives(world) -> dict:
    return {drive.name: drive for drive in world.behaviour._drives}


def _lone_mate(length: int, centre: int, diffusion_range: float) -> np.ndarray:
    """One typical mate's lust reading (emission 1, own-signature dot 2) along an axis, acuity 1."""
    response = _axis_operator(length, float(np.exp(-1.0 / diffusion_range)))[:, centre]
    return 2.0 * float(response[centre]) * response.astype(np.float64)


def profile(names: list[str]) -> None:
    """A lone mate's reading by distance, then each config's reach by acuity."""
    length, centre = 321, 160
    distances = (0, 2, 5, 10, 15, 20, 30, 40, 50, 60)
    print("R   " + "".join(f"{d:>10}" for d in distances))
    for diffusion_range in (5, 7, 10, 15, 20):
        reading = _lone_mate(length, centre, diffusion_range)
        print(f"{diffusion_range:<4}" + "".join(f"{reading[centre + d]:>10.2e}" for d in distances))
    acuities = (0.5, 1.0, 1.5, 3.0)
    print("\nreach (world units) at acuity " + ", ".join(f"{a:g}" for a in acuities))
    for name in names:
        diffusion_range, threshold = CONFIGS[name]
        reading = _lone_mate(length, centre, diffusion_range)[centre:]
        reach = [int(np.nonzero(a * reading >= threshold)[0].max()) for a in acuities]
        print(f"{name:<10} " + "  ".join(f"{r:>4}" for r in reach))


def cost() -> None:
    """ms per cue rebuild plus one (n, 9) candidate sample, median of 15."""
    rng = np.random.default_rng(0)
    for grid, n in ((80, 2_000), (256, 20_000)):
        terrain = Terrain(np.zeros((grid, grid), dtype=np.float32), cell_size=1.0)
        x, y = rng.uniform(0, grid - 1, n), rng.uniform(0, grid - 1, n)
        emission = rng.uniform(0.5, 1.5, n).astype(np.float32)
        signature = rng.uniform(0, 1, (n, 8)).astype(np.float32)
        cx = np.clip(x[:, None] + rng.uniform(-3, 3, (n, 9)), 0, grid - 1)
        cy = np.clip(y[:, None] + rng.uniform(-3, 3, (n, 9)), 0, grid - 1)
        row = []
        for diffusion_range in (3.0, 6.0, 10.0, 15.0, 20.0, 25.0):
            field = CueField(terrain, 8, CueFieldConfig(diffusion_range=diffusion_range))
            times = []
            for _ in range(15):
                start = time.perf_counter()
                field.rebuild(x, y, emission, signature)
                field.sample_excluding_source(cx, cy, x, y, emission, signature)
                times.append((time.perf_counter() - start) * 1e3)
            row.append(f"R{diffusion_range:g}={np.median(times):.1f}")
        print(f"{grid}x{grid}, n={n}: " + "  ".join(row))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def readers(sources: Path, names: list[str]) -> None:
    """What each config does to the field's other readers, on established worlds."""
    for name in names:
        diffusion_range, threshold = CONFIGS[name]
        rows = []
        for seed in SOURCE_SEEDS:
            world = build_demo_world(seed=seed, n_entities=200)
            snapshot.load(world, sources / f"source_{seed}.npz")
            retune(world, diffusion_range, threshold)
            store = world.store
            alive = np.flatnonzero(store.alive & (store.age >= 0))
            selection = Selection.from_indices(alive, capacity=store.alive.shape[0])
            world.scent.rebuild(Selection.from_mask(store.alive.copy()))
            drives = _drives(world)
            channels = np.stack(drives["fear"]._channels(selection))
            fear = 1.0 - np.prod(1.0 - channels, axis=0)
            x, y = store.x[alive].astype(np.float64), store.y[alive].astype(np.float64)
            squared = (x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2
            rows.append([
                fear.mean(),
                _spearman(channels.max(axis=0), (squared < 25).sum(1) - 1),
                _spearman(channels.max(axis=0), (squared < 400).sum(1) - 1),
            ])
        mean = np.mean(rows, axis=0)
        print(
            f"{name:<10} fear p mean {mean[0]:.3f} | rho(danger, neighbours < 5) {mean[1]:+.2f}"
            f" | rho(danger, neighbours < 20) {mean[2]:+.2f}"
        )


def _pair(job: dict) -> dict:
    name, seed, rep, sources = job["name"], job["seed"], job["rep"], Path(job["sources"])
    world = build_demo_world(seed=seed, n_entities=200)
    snapshot.load(world, sources / f"source_{seed}.npz")
    retune(world, *CONFIGS[name])
    run_seed = 1_000_003 * seed + 7_919 * 2 + rep
    world.rng.bit_generator.state = np.random.default_rng(run_seed).bit_generator.state
    store = world.store
    born = np.flatnonzero(store.alive & (store.age >= 0))
    keep = np.random.default_rng(run_seed + 1).choice(born, size=2, replace=False)
    doomed = store.alive.copy()
    doomed[keep] = False
    released = np.flatnonzero(doomed)
    # As `Cull.apply`: bodies back to the ground they lie on, then the rows freed.
    world.plants.return_nutrients(
        store.x[released].astype(np.float64),
        store.y[released].astype(np.float64),
        store.energy[released].astype(np.float64),
    )
    store.release(store.row_ids()[released])
    return {"name": name, **_follow_pair(world, PAIR_HORIZON)}


def _follow_pair(world, horizon: int) -> dict:
    store = world.store
    for tick in range(10, horizon + 10, 10):
        world.loop.advance(10)
        alive = store.alive
        if (alive & (store.age < 0)).any() or alive.sum() > 2:
            return {"outcome": "conceived", "tick": tick}
        if alive.sum() < 2:
            return {"outcome": "died", "tick": tick}
    return {"outcome": "apart", "tick": None}


def pairs(sources: Path, reps: int, workers: int, names: list[str]) -> None:
    """Established worlds knocked down to a random pair (#251's evidence), per config."""
    jobs = [
        {"name": name, "seed": seed, "rep": rep, "sources": str(sources)}
        for name in names
        for seed in SOURCE_SEEDS
        for rep in range(reps)
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_pair, jobs))
    _report(results, names)


def _founder(job: dict) -> dict:
    name, seed = job["name"], job["seed"]
    world = build_world(demo_world_config(2, seed), seed=seed)
    retune(world, *CONFIGS[name])
    store = world.store
    rows = np.flatnonzero(store.alive)
    store.x[rows] = [40.0 - FOUNDER_SEPARATION / 2.0, 40.0 + FOUNDER_SEPARATION / 2.0]
    store.y[rows] = 40.0
    store.age[rows] = 200
    topped_up = world.config.lust.abundant_energy - store.energy[rows]
    store.energy[rows] = world.config.lust.abundant_energy
    world.plants.record_founding_stock(float(topped_up.sum()))
    return {"name": name, **_follow_pair(world, FOUNDER_HORIZON)}


def founders(seeds: int, workers: int, names: list[str]) -> None:
    """Two fresh founders `FOUNDER_SEPARATION` apart, as the regression test sets up."""
    jobs = [{"name": name, "seed": seed} for name in names for seed in range(seeds)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_founder, jobs))
    _report(results, names)


def _report(results: list[dict], names: list[str]) -> None:
    for name in names:
        mine = [r for r in results if r["name"] == name]
        ticks = sorted(r["tick"] for r in mine if r["outcome"] == "conceived")
        counts = {k: sum(r["outcome"] == k for r in mine) for k in ("conceived", "died", "apart")}
        median = f"{np.median(ticks):.0f}" if ticks else "-"
        print(f"{name:<10} {counts}  of {len(mine)}  median conception tick {median}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("profile", "cost", "readers", "pairs", "founders"))
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--configs", default=",".join(CONFIGS))
    args = parser.parse_args()
    names = args.configs.split(",")
    if args.command == "profile":
        profile(names)
    elif args.command == "cost":
        cost()
    elif args.command == "readers":
        readers(args.sources, names)
    elif args.command == "pairs":
        pairs(args.sources, args.reps, args.workers, names)
    else:
        founders(args.seeds, args.workers, names)


if __name__ == "__main__":
    main()
