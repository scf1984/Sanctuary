"""Two animals far apart in an empty world find each other and breed (#251).

Before #251 a mature, well-fed pair a few tens of units apart never met: lust steers by the scent
field, and one animal's plume read below lust's detection threshold even standing next to it, while
the field's support ended about three diffusion ranges out. The pair's fate was decided by chance
encounters, and that, not demography, set #29's viable-population threshold
(`docs/spikes/scent-range.md`).

Statistical rather than golden (§2.2): the assertion is a count over seeds against a horizon, and
no exact tick appears. The horizon and the bar are set from the measured rate in the spike, with
margin, so the test fails on a regression to chance encounters rather than on noise.
"""

import numpy as np

from clients.viewer.demo_world import build_demo_world

SEEDS = range(16)
# Pair separation, world units. Inside the 15-57 units #251 observed between bottlenecked pairs
# that never met, and well beyond the 2-unit contact range, so no pair starts within reach.
SEPARATION = 30.0
# Ticks, about four founding generations (#29's spike measures ~113 ticks per generation). Every
# measured conception came by tick 420; a pair that has not met by then has usually died, because
# these are naive founders and most of them do. Kept short because this is the slowest ecology test.
HORIZON = 450
# Of 16 pairs. Measured 0/16 before #251, 4/16 at a 6.0 range and 10/16 as shipped
# (docs/spikes/scent-range.md). At the shipped rate, fewer than 5 happens about 0.3% of the time.
REQUIRED = 5


def _paired_world(seed):
    """The demo world reduced to two mature, well-fed animals `SEPARATION` apart."""
    world = build_demo_world(seed=seed, n_entities=2)
    store = world.store
    rows = np.flatnonzero(store.alive)
    centre_x = centre_y = 40.0
    store.x[rows] = [centre_x - SEPARATION / 2.0, centre_x + SEPARATION / 2.0]
    store.y[rows] = [centre_y, centre_y]
    # Past any founder's maturity, and with energy at lust's saturation, so the only thing between
    # these two and a conception is finding each other.
    store.age[rows] = 200
    topped_up = world.config.lust.abundant_energy - store.energy[rows]
    store.energy[rows] = world.config.lust.abundant_energy
    # The top-up arrives from outside the field, as a founder's endowment does, so it goes on the
    # export ledger the same way. Without it the first upkeep returns nutrients that never left.
    world.plants.record_founding_stock(float(topped_up.sum()))
    return world


def _conceives(world):
    store = world.store
    for _ in range(0, HORIZON, 10):
        world.loop.advance(10)
        alive = store.alive
        # A gestating young is a row with a negative age (#20); a birth adds a third living row.
        if (alive & (store.age < 0)).any() or alive.sum() > 2:
            return True
        if alive.sum() < 2:
            return False
    return False


def test_a_separated_pair_finds_each_other_and_breeds():
    conceived = sum(_conceives(_paired_world(seed)) for seed in SEEDS)
    assert conceived >= REQUIRED, f"only {conceived}/{len(SEEDS)} separated pairs bred"
