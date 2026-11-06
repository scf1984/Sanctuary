"""A world that starves to extinction ends as `collapsed`, not as an exception (#240).

Starvation is the main way a world ends today, and the last animals to starve are the ones whose
bodies the export ledger is closest to owing in full. When `energy` was float32 against a float64
ledger, the rounding each meal left behind did not shrink with the population: by the time a world
was down to its last fraction of a body, the final unit burned read as more than had ever left the
field, and `Plants.return_nutrients` raised instead of the loop reaching §2.7's run-end. With the
debug harness on, #236's `exported_nutrients_are_held` tripped first, on the same gap.

These worlds are #240's own reproductions: two naive founders on the demo map, which starve out
within about a thousand ticks. Run with the invariant harness on, so both guards are exercised.
"""

import pytest

from clients.viewer.demo_world import demo_world_config
from core.world.assembly import build_world

# Comfortably past every observed extinction (ticks 408-689 for these seeds before the fix). A
# collapsed loop refuses to advance, so overshooting costs nothing; falling short would make the
# test pass without the world ever reaching the tick that used to raise.
HORIZON = 5_000


@pytest.mark.parametrize("seed", [2, 4, 5])
def test_a_world_that_starves_out_collapses_cleanly(seed):
    world = build_world(demo_world_config(2, seed), seed=seed, debug_checks=True)

    world.loop.advance(HORIZON)

    assert world.loop.collapsed
