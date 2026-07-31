"""Watch every gene in a running world, and see what genetic distance is actually made of (#193).

`distance.between` is an unweighted Euclidean norm over the expressed phenotype, so a gene's share
of "genetic distance" is proportional to its natural magnitude. #193 asks how genes should be
commensurated before that norm is taken, and that is not a question to answer from first principles:
it depends on which genes actually drift, how fast, and how much each one contributes in practice.

So this exports the data rather than arguing about it. Two CSVs per run:

  <out>-genes.csv     one row per (tick, gene): the expressed distribution across the living
                      population, and how far it has moved from where the founders started.
  <out>-distance.csv  one row per (tick, gene): that gene's share of mean squared pairwise
                      distance — which is the thing #193 is actually about.

Deliberately dependency-free: numpy and the standard library, so running it needs nothing the core
already lacks. Plot the CSVs with whatever you like.

    python -m docs.spikes.gene_observatory --ticks 3000 --out run

Nothing here is a shipped metric. #30 owns metric *definitions* and #157 owns showing trait
distributions to a player; this is a spike producing evidence for one decision, in the same way
`speciation_drift_spike.py` produced the numbers behind #104's inheritance rule.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from clients.viewer.demo_world import demo_world_config
from core.world.assembly import build_world

# Enough pairs that a percentile is stable, few enough that sampling them every interval is free
# against the tick it interrupts.
_PAIRS = 2000


def _living(store) -> np.ndarray:
    """Rows of everything born and alive — the unborn hold negative ages and are not participants."""
    return np.flatnonzero(store.alive & (store.age >= 0))


def observe(seed: int, founders: int, ticks: int, every: int, out: Path) -> None:
    world = build_world(demo_world_config(founders, seed), seed=seed)
    store = world.store
    names = [gene.name for gene in world.config.genes]
    rng = np.random.default_rng(seed)

    opening = world.genetics.expressed_at(_living(store)).mean(axis=0)

    genes_path = out.with_name(out.name + "-genes.csv")
    distance_path = out.with_name(out.name + "-distance.csv")
    with genes_path.open("w", newline="") as genes_file, distance_path.open(
        "w", newline=""
    ) as distance_file:
        genes_out = csv.writer(genes_file)
        genes_out.writerow(
            ["tick", "population", "gene", "p10", "median", "p90", "mean", "std", "drift"]
        )
        distance_out = csv.writer(distance_file)
        distance_out.writerow(
            ["tick", "gene", "mean_square_gap", "share_of_distance", "total_distance_median"]
        )

        for tick in range(0, ticks + 1, every):
            rows = _living(store)
            if rows.size < 2:
                break
            phenotype = world.genetics.expressed_at(rows)

            low, mid, high = np.percentile(phenotype, [10, 50, 90], axis=0)
            mean = phenotype.mean(axis=0)
            std = phenotype.std(axis=0)
            for i, name in enumerate(names):
                genes_out.writerow(
                    [
                        tick,
                        rows.size,
                        name,
                        f"{low[i]:.6g}",
                        f"{mid[i]:.6g}",
                        f"{high[i]:.6g}",
                        f"{mean[i]:.6g}",
                        f"{std[i]:.6g}",
                        f"{mean[i] - opening[i]:.6g}",
                    ]
                )

            # What genetic distance is *made of*. `between` is a Euclidean norm, so each gene's
            # contribution to the squared distance is the squared gap in that gene — summing them
            # and dividing gives the share, which is exactly what #193 is asking about.
            a = world.genetics.expressed_at(rng.choice(rows, _PAIRS))
            b = world.genetics.expressed_at(rng.choice(rows, _PAIRS))
            square_gap = ((a - b) ** 2).mean(axis=0)
            total = square_gap.sum()
            median_distance = float(np.median(np.linalg.norm(a - b, axis=1)))
            for i, name in enumerate(names):
                distance_out.writerow(
                    [
                        tick,
                        name,
                        f"{square_gap[i]:.6g}",
                        f"{square_gap[i] / total:.6g}" if total > 0 else "0",
                        f"{median_distance:.6g}",
                    ]
                )

            print(
                f"tick {tick:>6}  living {rows.size:>6}  median pairwise distance "
                f"{median_distance:8.3f}",
                flush=True,
            )
            world.loop.advance(every)

    print(f"\nwrote {genes_path} and {distance_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--founders", type=int, default=200)
    parser.add_argument("--ticks", type=int, default=3000)
    parser.add_argument("--every", type=int, default=100, help="ticks between samples")
    parser.add_argument("--out", type=Path, default=Path("gene-observatory"))
    args = parser.parse_args()
    observe(args.seed, args.founders, args.ticks, args.every, args.out)


if __name__ == "__main__":
    main()
