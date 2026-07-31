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


def shift(seed, gene):
    """How far a weight moved from where the founders started, as a fraction of it."""
    world, opening = run(seed)
    return abs(float(weights(world, gene).mean()) - opening[gene]) / opening[gene]


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


class TestOnlyASteeringDriveIsSelected:
    """The result that was measured rather than assumed.

    A flat drive adds a constant to every option, so it cannot change the choice and its weight is
    nearly a free random walk. A steering drive changes which option wins, so its weight is paid
    for or rewarded every tick.
    """

    @pytest.mark.parametrize("flat", ["fear_weight", "thirst_weight"])
    def test_hunger_moves_further_than_a_flat_drive(self, flat):
        """Hunger reads the forage field and steers; fear and thirst are still flat, because
        flight does not exist (#188 gave lust its direction and left fear's for #24) and nothing
        drinks. So hunger's weight is paid for every tick and theirs is drifting.

        **Asserted across replicates, not per seed.** §2.2 is explicit that variance between two
        runs of the same state can exceed the difference being measured, so a per-seed assertion
        here is a coin flip dressed as a result — it failed on one seed of three while holding
        comfortably in aggregate. This compares the median shift over `SEEDS` runs, which is the
        form §2.2 asks competitions to use.
        """
        steering = [shift(seed, "hunger_weight") for seed in SEEDS]
        drifting = [shift(seed, flat) for seed in SEEDS]

        assert float(np.median(steering)) > float(np.median(drifting))


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
