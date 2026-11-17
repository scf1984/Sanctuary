"""One frame of the viewer, end to end through the boundary, without a display (#110, #35).

The bug this file exists for was not in any single render function, since each is unit-tested,
but in the seam between the world and them, which nothing exercised. Chaining them over a real
session is what makes a shape, dtype or signature drift fail here rather than at the first
keypress. `app.py` itself imports pygame and cannot be collected in CI, so this is the closest a
test gets to `run()`.
"""

import api
from clients.viewer.render import (
    CONDITION_MODES,
    condition_colors,
    elevation_shading,
    apply_water_overlay,
    live_positions,
    species_colors,
    world_to_screen,
)

_SCREEN = (640, 480)


def test_one_frames_worth_of_rendering_runs_headlessly():
    """Everything `run()` does per frame except the pygame calls themselves."""
    session = api.create("demo", seed=4, founders=40)
    info = session.describe()
    terrain = session.terrain()
    background = apply_water_overlay(elevation_shading(terrain), terrain.water_depth.to_numpy())
    previous = session.frame()
    session.advance(1)
    current = session.frame()

    x, y, _z = live_positions(previous, current, 0.5)
    px, py = world_to_screen(x, y, info.width, info.height, *_SCREEN)

    n_live = session.status().living
    assert background.shape == info.grid_shape + (3,)
    assert px.shape == py.shape == (n_live,)
    assert ((0 <= px) & (px < _SCREEN[0])).all()
    assert ((0 <= py) & (py < _SCREEN[1])).all()
    for mode in CONDITION_MODES:
        colors = (
            species_colors(current.species.to_numpy())
            if mode == "species"
            else condition_colors(session.condition(mode).values.to_numpy())
        )
        assert colors.shape == (n_live, 3)
    # Every founder is given a species, so none falls back to the unset-species gray.
    assert (current.species.to_numpy() >= 0).all()


def test_a_death_removes_an_animal_from_the_frame():
    """End-to-end proof that the ghost is gone (#119).

    Releases an animal directly rather than waiting for starvation to arrive on its own schedule.
    The point is that the *frame path* stops drawing it, not that any particular system caused it.
    """
    session = api.create("demo", seed=4, founders=40)
    session.advance(1)
    previous = session.frame()
    doomed = previous.ids.to_numpy()[:1]

    session._world.store.release(doomed)
    current = session.frame()
    x, _y, _z = live_positions(previous, current, 0.5)

    assert len(x) == previous.ids.shape[0] - 1
    assert doomed[0] not in current.ids.to_numpy()
