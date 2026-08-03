"""Does a world survive predation, and does carnivory pay for itself (#179)?

Two questions this issue cannot answer by reasoning, so §8.1's explore-then-lock-in applies.

**Does the world survive?** Every founder carries a real flesh allocation already — the demo world
founds `diet_animal_derived` over (-1, 1), which the logistic reading turns into 0.27-0.73 — so
switching predation on does not introduce predators, it makes an allocation that already existed
start acting. Whether that is a transient the population walks out of, or mutual annihilation, is a
contact-rate question and not a design one.

**Does carnivory pay without hunting?** A predator here eats what it bumps into and does not seek
prey. If the flesh allocation falls steadily toward zero, that says contact alone is too rare to
support the strategy and that locating prey (#179's open Decision 1) is required rather than
optional. If it holds or rises anywhere, predation is viable as built.

Run from a repository root with `PYTHONPATH=.`.
"""

from __future__ import annotations

import numpy as np

from clients.viewer.demo_world import build_demo_world
from core.selection import Selection

TICKS = 2000
SAMPLE_EVERY = 250
SEEDS = (1, 2)


def living_of(world):
    return Selection.from_mask(world.store.alive & (world.store.age >= 0))


def sample(world):
    living = living_of(world)
    if not len(living):
        return 0, float("nan"), float("nan"), float("nan")
    phenotype = world.genetics.expressed(living)
    share = world.feeding.diet.animal_share(phenotype)
    return (
        len(living),
        float(np.mean(share)),
        float(np.max(share)),
        float(world.carrion.mass.sum()),
    )


def main():
    for seed in SEEDS:
        world = build_demo_world(seed=seed, n_entities=200)
        print(f"\nseed {seed}")
        print(f"{'tick':>6}{'living':>8}{'flesh share':>13}{'max':>7}{'carrion':>10}")
        for tick in range(0, TICKS + 1, SAMPLE_EVERY):
            if tick:
                world.loop.advance(SAMPLE_EVERY)
            n, mean_share, max_share, carrion = sample(world)
            print(f"{tick:>6}{n:>8}{mean_share:>13.3f}{max_share:>7.3f}{carrion:>10.1f}")
            if not n:
                break


if __name__ == "__main__":
    main()
