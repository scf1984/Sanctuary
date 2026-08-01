"""Does the capacity reserve actually clear a tick's births? (#200, #127)

`GrowthConfig.reserve_fraction` was derived from an allocation rate measured while
`speciation_threshold` was rejecting 94% of couples (#199), so the rate it was sized against was
roughly 24x too low. Too small a reserve does not raise — `Conception.conceive` truncates to
`store.available` — so the array silently caps the population, which is the failure #127 existed to
remove.

This instruments the thing that matters: **how often conception runs out of rows**, and what the
allocation rate actually is now.

    python -m docs.spikes.capacity_rate_bench --ticks 1500

Reports the population curve, the peak per-tick allocation as a share of occupancy, the peak
concurrent gestation share, and — the point — the number of ticks on which conception was clipped.
"""

from __future__ import annotations

import argparse

import numpy as np

from clients.viewer.demo_world import demo_world_config
from core.world.assembly import build_world


def measure(seed: int, founders: int, ticks: int, sample: int) -> dict:
    world = build_world(demo_world_config(founders, seed), seed=seed)
    store = world.store

    # Wrap the conception system to see what it had to work with. `_build_systems` closes over the
    # service and resolves the method per call, so patching the instance is enough.
    original = world.conception.conceive
    per_tick: list[tuple[int, int, int]] = []

    def watched(selection, rng):
        free_before = store.available
        occupied_before = int(store.alive.sum())
        original(selection, rng)
        per_tick.append(
            (free_before, occupied_before, int(store.alive.sum()) - occupied_before)
        )

    world.conception.conceive = watched  # type: ignore[method-assign]

    curve = []
    for tick in range(ticks):
        if tick % sample == 0:
            born = store.alive & (store.age >= 0)
            curve.append(
                (
                    tick,
                    int(born.sum()),
                    int((store.alive & (store.age < 0)).sum()),
                    store.available,
                    store.capacity,
                )
            )
        world.loop.advance(1)

    free, occupied, conceived = (np.array(column) for column in zip(*per_tick))
    # A tick that conceived exactly as many young as it had rows is a tick that was probably
    # clipped: `conceive` truncates to `store.available` and there is no way to ask it afterwards.
    clipped = int(((conceived > 0) & (conceived == free)).sum())
    rate = np.divide(conceived, np.maximum(occupied, 1))
    gestating = np.array([row[2] for row in curve])
    living = np.array([row[1] for row in curve])

    return {
        "curve": curve,
        "peak_rate": float(rate.max()),
        "p99_rate": float(np.percentile(rate, 99)),
        "mean_rate": float(rate.mean()),
        "clipped_ticks": clipped,
        "zero_free_ticks": int((free == 0).sum()),
        "conceptions": int(conceived.sum()),
        "peak_gestating_share": float(
            (gestating / np.maximum(living + gestating, 1)).max()
        ),
        "final_capacity": store.capacity,
        "peak_living": int(living.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--founders", type=int, default=150)
    parser.add_argument("--ticks", type=int, default=1500)
    parser.add_argument("--sample", type=int, default=150)
    args = parser.parse_args()

    for seed in args.seeds:
        result = measure(seed, args.founders, args.ticks, args.sample)
        print(f"\n=== seed {seed} ===")
        print("%7s %8s %10s %8s %9s" % ("tick", "living", "gestating", "free", "capacity"))
        for row in result["curve"]:
            print("%7d %8d %10d %8d %9d" % row)
        print(
            "\nconceptions %d | peak alloc/tick %.4f of occupancy | p99 %.4f | mean %.4f"
            % (
                result["conceptions"],
                result["peak_rate"],
                result["p99_rate"],
                result["mean_rate"],
            )
        )
        print(
            "peak gestating share %.3f | final capacity %d | peak living %d"
            % (result["peak_gestating_share"], result["final_capacity"], result["peak_living"])
        )
        print(
            "TICKS WITH NO FREE ROWS AT CONCEPTION: %d | ticks probably clipped: %d"
            % (result["zero_free_ticks"], result["clipped_ticks"])
        )


if __name__ == "__main__":
    main()
