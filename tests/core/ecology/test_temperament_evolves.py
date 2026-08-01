"""Drive weights are genes, so temperament is selected rather than designed (#23, CLAUDE.md §2.5).

Locked in after the fact (§8.1): whether selection actually moves a weight is an ecological question
with no failing test to write in advance. It was observed first, and what was observed was not what
was expected — see `docs/spikes/temperament.md`.

Directions and distributions only, never values (§2.2).

**A weight is only under selection if its drive has a direction.** `Behaviour.choose` scores
`utility(option) = Σ urgency × appeal(option)`, so a drive whose appeal is flat adds the *same*
number to every option and cannot change which one wins. Hunger steers by the forage field and lust
by the cue field (#188); fatigue steers toward the null option; thirst and fear are still flat, and
their weights are therefore very nearly neutral genes. That is a fact about what is built, not about
temperament.
"""

import numpy as np
import pytest

from clients.viewer.demo_world import demo_world_config
from core.selection import Selection
from core.world.assembly import build_world

# Long enough that founders have been replaced by descendants — maturity is drawn from (40, 120)
# ticks — and small enough that three seeds stay inside a CI budget.
TICKS = 600
FOUNDERS = 120
SEEDS = (0, 1, 2, 3, 4)

_RUNS: dict[int, tuple] = {}


def run(seed):
    """One world per seed, advanced once and shared by every assertion below.

    Cached because these runs are the expensive part: rebuilding a world per test turned a
    fourteen-assertion module into eight minutes.
    """
    if seed not in _RUNS:
        world = build_world(demo_world_config(FOUNDERS, seed), seed=seed)
        opening = {name: float(weights(world, name).mean()) for name in WEIGHTS}
        world.loop.advance(TICKS)
        _RUNS[seed] = (world, opening)
    return _RUNS[seed]


WEIGHTS = ("hunger_weight", "thirst_weight", "fear_weight", "lust_weight", "fatigue_weight")


def weights(world, gene):
    living = Selection.from_mask(world.store.alive & (world.store.age >= 0))
    return world.genetics.expressed(living)[:, world.genes.index_of(gene)]


class TestAWeightIsCarriedAndPassedOn:
    def test_founders_differ_in_temperament(self):
        """Nothing can select on a distribution that does not exist."""
        world = build_world(demo_world_config(FOUNDERS, 0), seed=0)

        assert weights(world, "hunger_weight").std() > 0.0

    @pytest.mark.parametrize("seed", SEEDS)
    def test_descendants_still_vary(self, seed):
        """A population that converged on one temperament would make every assertion below
        vacuous — and #104's inheritance floor exists precisely to prevent that."""
        world, _ = run(seed)

        assert weights(world, "fear_weight").std() > 0.0

    @pytest.mark.parametrize("seed", SEEDS)
    def test_every_weight_stays_non_negative(self, seed):
        """Read as a magnitude, so a lineage whose stored value drifted below zero still wants its
        drive rather than wanting the opposite of it (§8.7)."""
        world, _ = run(seed)

        for gene in WEIGHTS:
            assert (weights(world, gene) >= 0.0).all()


class TestOnlyASteeringDriveCanChangeTheChoice:
    """Why some weights are under selection and others are not, asserted **mechanically**.

    `Behaviour.choose` samples from `utility(option) = Σ urgency × appeal(option)`. A drive whose
    appeal is *flat* adds the same number to every option, and the Boltzmann sampling is invariant
    to a shift shared by all of them — so that drive's weight cannot change which option wins, at
    any magnitude. It is a neutral gene by construction rather than by measurement.

    This replaces a population-level statistical claim that did not survive contact with a fixed
    world: see `docs/spikes/temperament.md`. The mechanism is provable, so proving it beats
    measuring a shadow of it over 600 ticks.
    """

    def chosen(self, seed, scale=None, gene=None):
        world = build_world(demo_world_config(FOUNDERS, seed), seed=seed)
        # Advanced first, identically, because a brand-new world has no plants: the field starts
        # empty and grows, so hunger's appeal is all zeros and *every* drive is flat at tick 0.
        world.loop.advance(120)
        population = Selection.from_mask(world.store.alive & (world.store.age >= 0))
        if gene is not None:
            genes = world.genetics.genes(population)
            genes[:, world.genes.index_of(gene)] *= scale
            world.genetics.set_genes(population, genes)
        world.behaviour.choose(population, np.random.default_rng(seed))
        return world.store.choice_heading[population.to_mask()].copy()

    @pytest.mark.parametrize("flat", ["fear_weight", "thirst_weight"])
    def test_a_flat_drives_weight_cannot_change_the_decision(self, flat):
        """Ten times the weight, identical headings. Fear and thirst are still flat — flight has
        never existed and nothing drinks — so scaling them moves nothing."""
        np.testing.assert_array_equal(self.chosen(0), self.chosen(0, 10.0, flat))

    def test_a_steering_drives_weight_does_change_it(self):
        """The control: hunger reads the forage field, so its weight reaches the choice."""
        assert not np.array_equal(self.chosen(0), self.chosen(0, 10.0, "hunger_weight"))


class TestTheDegenerateAttractor:
    """#23 names the risk: "populations can evolve into degenerate strategies (never eating, never
    fleeing)". The one this world could reach is *never moving* — resting is free (#107) and with
    food underfoot an animal that never walks pays no locomotion at all.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_the_population_survives(self, seed):
        world, _ = run(seed)

        assert world.store.alive.sum() > 0

    @pytest.mark.parametrize("seed", SEEDS)
    def test_animals_are_still_moving_at_the_end(self, seed):
        """Asserted on behaviour rather than on the gene: a world collapsed into permanent rest
        would be motionless, whatever its weights read."""
        world, _ = run(seed)
        alive = world.store.alive.copy()
        before = world.store.x[alive].copy()

        world.loop.advance(5)

        assert (world.store.x[: alive.shape[0]][alive] != before).any()
