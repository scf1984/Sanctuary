"""Headless coverage of the world the viewer assembles and of the frame path that draws it.

This file must stay importable **without pygame**, which is the whole point of it: CI installs
`.[dev]` and never the viewer extra, deliberately, so that the core is provably runnable headless.
`app.py` imports pygame at module scope, so a test that reached into it would be uncollectable in
CI — which is exactly how a `TypeError` in `_build_demo_world` survived unnoticed since the gene
matrix landed (#110). World-building lives in `clients.viewer.demo_world` for that reason, and
these tests import it directly.
"""

import numpy as np

from clients.viewer.demo_world import build_demo_world
from clients.viewer.render import interpolate_positions, species_colors, world_to_screen
from core.invariants import default_registry

_SCREEN = (640, 480)


class TestBuildDemoWorld:
    def test_allocates_every_entity_it_was_asked_for(self):
        _terrain, _water, store, _loop = build_demo_world(seed=0, n_entities=25)

        assert int(store.alive.sum()) == 25

    def test_entities_stand_on_the_terrain_surface(self):
        """§2.6 stores z from the start but movement is surface-locked, so a demo entity's z is
        the ground beneath it rather than an independent coordinate it could drift from.
        """
        terrain, _water, store, _loop = build_demo_world(seed=1, n_entities=25)
        alive = store.alive

        surface = terrain.elevation_at(store.x[alive], store.y[alive])

        np.testing.assert_allclose(store.z[alive], surface, rtol=1e-6)

    def test_a_freshly_built_world_satisfies_every_invariant_checkable_today(self):
        terrain, _water, store, loop = build_demo_world(seed=2, n_entities=50)
        registry = default_registry(0.0, terrain.world_width, 0.0, terrain.world_height)

        registry.check_all(store, loop.tick_count)  # must not raise

    def test_the_same_seed_rebuilds_the_same_world(self):
        """The simulation is non-deterministic by design (§2.2), but *generation* is seeded so a
        crash can be replayed. A demo world that could not be rebuilt would make the viewer
        useless for exactly the diagnosis §3.3 exists for.
        """
        _terrain, _water, first, _loop = build_demo_world(seed=7, n_entities=30)
        _terrain, _water, second, _loop = build_demo_world(seed=7, n_entities=30)

        np.testing.assert_array_equal(first.x, second.x)
        np.testing.assert_array_equal(first.y, second.y)
        np.testing.assert_array_equal(first.species_id, second.species_id)

    def test_the_loop_advances_the_store_it_was_built_around(self):
        terrain, _water, store, loop = build_demo_world(seed=3, n_entities=10)
        registry = default_registry(0.0, terrain.world_width, 0.0, terrain.world_height)

        loop.advance(5)

        assert loop.tick_count == 5
        registry.check_all(store, loop.tick_count)  # must not raise
        # No system is registered yet (#115 owns the assembly), so a tick is observably a no-op
        # on position — the assertion that matters is that advancing does not raise.
        np.testing.assert_array_equal(loop.current_positions[0], store.x)


class TestFramePath:
    def test_one_frames_worth_of_rendering_runs_headlessly(self):
        """Everything `run()` does per frame except the pygame calls themselves.

        The bug this file exists for was not in any of these functions — each is unit-tested —
        but in the seam between the world and them, which nothing exercised. Chaining them over a
        real demo world is what makes a shape, dtype or signature drift fail here rather than at
        the first keypress.
        """
        terrain, _water, store, loop = build_demo_world(seed=4, n_entities=40)
        loop.advance(1)

        x, y, _z = interpolate_positions(loop.previous_positions, loop.current_positions, 0.5)
        px, py = world_to_screen(x, y, terrain.world_width, terrain.world_height, *_SCREEN)
        colors = species_colors(store.species_id)

        assert px.shape == py.shape == (store.capacity,)
        assert ((0 <= px) & (px < _SCREEN[0])).all()
        assert ((0 <= py) & (py < _SCREEN[1])).all()
        assert colors.shape == (store.capacity, 3)
        # Every demo entity is given a species, so none falls back to the unset-species gray.
        assert (store.species_id[store.alive] >= 0).all()
