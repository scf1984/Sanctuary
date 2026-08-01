"""Hunger knows which way the food is — asserted end to end, over a world that has been grazed.

This is the half of #205 that turned out to be *working*, and it had no test. `core.world.diffusion`
and `core.ecology.plants` are each covered in isolation, and `Hunger.appeal` is covered against a
hand-made field, but nothing checked the one property the whole of #93 exists to produce: that in a
running world, the heading hunger prefers is the heading with the most food behind it.

It is a statistical test rather than a golden one (§2.2, §6): the assertion is a direction against
chance, over seeds, and no exact value appears.

**Why it is worth its runtime.** The diagnosis in `docs/spikes/who-steers.md` rests entirely on
this number being ~0.99 — that is what makes "hunger is perfectly informed and still never steers"
a statement about the drive contest rather than about a broken field. A regression in the forage
field, the diffusion, the acuity gate or the candidate positions would move it, and *nothing else
in the suite would notice*: every downstream population figure would stay plausible, because a
world of animals foraging at random still eats, breeds and stabilises. That is exactly the failure
§8.1 means by a test you would miss.
"""

import numpy as np
import pytest

from clients.viewer.demo_world import build_demo_world
from core.selection import Selection

# Long enough that the founders have grazed the field into something with structure — an ungrazed
# world grows uniformly from a smooth terrain, so every candidate reads nearly the same and the
# rank is 0.5 for want of anything to prefer rather than for want of a working drive.
TICKS = 150


def ranked_by_forage(crop, picked):
    """Mean rank of `picked` among the options it was picked from, by forage; chance is 0.5.

    Mid-ranked, so a neighbourhood where every candidate reads the same scores 0.5 rather than
    crediting a choice for a distinction that was not there to make. That matters here: it is what
    stops an ungrazed patch of world inflating the result.
    """
    rows = np.arange(crop.shape[0])
    chosen = crop[rows, picked][:, None]
    better = (crop < chosen).sum(axis=1)
    tied = (crop == chosen).sum(axis=1)
    return float(np.mean((better + 0.5 * (tied - 1)) / (crop.shape[1] - 1)))


def grazed_world(seed):
    world = build_demo_world(seed=seed, n_entities=200)
    world.loop.advance(TICKS)
    return world


def options(world):
    """(population, per-drive contributions, forage reading at every candidate).

    Scores a fresh decision rather than reading `choice_heading`, because the question is about the
    option *set* an animal faced and the store keeps only the option it took.
    """
    population = Selection.from_mask(world.store.alive & (world.store.age >= 0))
    rng = np.random.default_rng(99)
    headings = world.behaviour.candidate_headings(population, rng)
    x, y = world.behaviour.candidate_positions(population, headings)
    commitment = world.genetics.expressed(population)[
        :, world.genes.index_of("commitment")
    ].astype(np.float64)
    _, contributions = world.behaviour.utilities(population, headings, x, y, commitment)
    return population, contributions, world.plants.forage_at(world.plants.forage, x, y)


@pytest.mark.parametrize("seed", [1, 2])
def test_the_heading_hunger_prefers_is_the_one_with_the_most_food(seed):
    """The whole of #93 in one number: a diffused, cost-aware crop field read at candidate
    positions ranks those candidates by how much grazing is *reachable* from each.

    Asserted well above chance rather than at a value, but the margin is deliberately wide — this
    measures ~0.99 across seeds, so a floor of 0.9 is far below the observed figure and far above
    anything a broken field could reach by accident.
    """
    _, contributions, crop = options(grazed_world(seed))

    hungriest = np.argmax(contributions["hunger"], axis=1)

    assert ranked_by_forage(crop, hungriest) > 0.9


@pytest.mark.parametrize("seed", [1, 2])
def test_a_drive_that_perceives_nothing_ranks_no_better_than_chance(seed):
    """The control, and the reason the test above means anything.

    Fear is flat by construction — flight has never existed (§2.5, #24) — so its "preferred"
    heading is whichever the argmax tie-break lands on. If *that* also scored 0.99 the measure
    would be reading something other than the drive, and the whole diagnosis on #205 would be
    built on an artefact.
    """
    _, contributions, crop = options(grazed_world(seed))

    indifferent = np.argmax(contributions["fear"], axis=1)

    assert ranked_by_forage(crop, indifferent) == pytest.approx(0.5, abs=0.1)
