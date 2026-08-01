"""Which drive actually decides where an animal goes, and what decisiveness buys.

Issue #205, which asked whether founders start too *indecisive*: `Behaviour.choose` samples from
`exp(utility / temperature)`, and at the founding temperature animals took their best option 13–27%
of the time against 11% by chance. The obvious reading was that the drives barely steer and the
founding range should be tightened.

This script exists because that reading was wrong, and it prints the three tables that show why.
Run from a repository root with `PYTHONPATH=.`; each section is independent.

1. `sweep()` — decisiveness against ecology. Founds every animal at one fixed temperature and runs
   the world, reporting population, condition, and how good the chosen heading was.
2. `attribute()` — the same, per drive: how good the heading *each drive on its own* would have
   picked was. This is what names the culprit.
3. `spreads()` — how far each drive's appeal varies **across options**, which is the quantity that
   decides who steers. A drive's urgency sets how much it contributes; only its spread moves a
   ranking.

The measure throughout is the **forage rank of a chosen heading**: where it sits, by the forage
field, among the candidates that animal considered. Chance is 0.5. It is used in preference to
"standing crop under the population over the field mean", which was tried first and is confounded —
animals eat what they stand on, so a *better* forager depresses its own numerator.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from clients.viewer.demo_world import demo_world_config
from core.selection import Selection
from core.world.assembly import build_world

# Stored gene values, read through `exp` — temperatures of 0.08 to 1.0, the last being where
# founders sit today. An order of magnitude either side rather than a bracket around a guess.
STORED = (-2.5, -2.0, -1.5, -1.0, -0.5, 0.0)
SEEDS = (1, 2, 3)
TICKS = 400


def founded_at(stored, seed, n_entities=200):
    """A demo world whose founders all carry `stored` as their choice temperature.

    Fixed rather than drawn from a band, so one row of the sweep reads one temperature. The shipped
    config draws from a range for the reason §2.5 gives — selection needs founders to differ.
    """
    config = demo_world_config(n_entities, seed)
    ranges = dict(config.founder_gene_ranges)
    ranges["choice_temperature"] = (stored, stored)
    return build_world(dataclasses.replace(config, founder_gene_ranges=ranges), seed=seed)


def scored(world, population, rng):
    """One fresh decision, decomposed: (chosen, best, per-drive contributions, forage at each).

    Re-scores rather than reading `choice_heading`, because the question is about the *option set*
    an animal faced and the store keeps only the option it took.
    """
    behaviour = world.behaviour
    headings = behaviour.candidate_headings(population, rng)
    x, y = behaviour.candidate_positions(population, headings)
    expressed = world.genetics.expressed(population)
    commitment = expressed[:, world.genes.index_of("commitment")].astype(np.float64)
    total, contributions = behaviour.utilities(population, headings, x, y, commitment)
    temperature = expressed[:, world.genes.index_of("choice_temperature")].astype(np.float64)
    scaled = total / temperature[:, None]
    chosen = np.argmax(scaled + rng.gumbel(size=scaled.shape), axis=1)
    best = np.argmax(scaled, axis=1)
    return chosen, best, contributions, world.plants.forage_at(world.plants.forage, x, y)


def forage_rank(crop, picked):
    """Mean rank of `picked` by forage among the options it was picked from; chance is 0.5.

    Mid-ranked, so a flat neighbourhood — every candidate reading identical — scores 0.5 rather
    than crediting the draw for a distinction that was not there to make.
    """
    rows = np.arange(crop.shape[0])
    chosen = crop[rows, picked][:, None]
    better = (crop < chosen).sum(axis=1)
    tied = (crop == chosen).sum(axis=1)
    return float(np.mean((better + 0.5 * (tied - 1)) / (crop.shape[1] - 1)))


def sweep():
    print("\n1. decisiveness against ecology\n")
    print(f"{'temp':>7}{'seed':>6}{'living':>8}{'energy p50':>12}{'takes best':>12}{'forage rank':>13}")
    for stored in STORED:
        for seed in SEEDS:
            world = founded_at(stored, seed)
            world.loop.advance(TICKS)
            population = Selection.from_mask(world.store.alive & (world.store.age >= 0))
            chosen, best, _, crop = scored(world, population, np.random.default_rng(99))
            print(
                f"{np.exp(stored):>7.2f}{seed:>6}{len(population):>8}"
                f"{np.median(world.ecology.energy(population)):>12.1f}"
                f"{float((chosen == best).mean()):>12.3f}{forage_rank(crop, chosen):>13.3f}",
                flush=True,
            )


def attribute():
    print("\n2. where each drive on its own would have gone\n")
    print(f"{'temp':>7}{'seed':>6}{'chosen':>9}   per-drive argmax rank")
    for stored in (-2.5, 0.0):
        for seed in SEEDS[:2]:
            world = founded_at(stored, seed)
            world.loop.advance(TICKS)
            population = Selection.from_mask(world.store.alive & (world.store.age >= 0))
            chosen, _, contributions, crop = scored(world, population, np.random.default_rng(99))
            per_drive = "  ".join(
                f"{name} {forage_rank(crop, np.argmax(c, axis=1)):.3f}"
                for name, c in contributions.items()
            )
            print(
                f"{np.exp(stored):>7.2f}{seed:>6}{forage_rank(crop, chosen):>9.3f}   {per_drive}",
                flush=True,
            )


def spreads():
    print("\n3. how loudly each drive speaks about direction\n")
    print(f"{'seed':>6}{'drive':>12}{'urgency':>10}{'spread over options':>22}")
    for seed in SEEDS[:2]:
        world = founded_at(0.0, seed)
        world.loop.advance(TICKS)
        population = Selection.from_mask(world.store.alive & (world.store.age >= 0))
        _, _, contributions, _ = scored(world, population, np.random.default_rng(99))
        for name, contribution in contributions.items():
            spread = (contribution.max(axis=1) - contribution.min(axis=1)).mean()
            print(
                f"{seed:>6}{name:>12}{contribution.max(axis=1).mean():>10.4f}{spread:>22.4f}",
                flush=True,
            )


if __name__ == "__main__":
    sweep()
    attribute()
    spreads()
