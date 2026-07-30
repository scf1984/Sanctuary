"""Headless coverage of the world the viewer assembles and of the frame path that draws it.

This file must stay importable **without pygame**, which is the whole point of it: CI installs
`.[dev]` and never the viewer extra, deliberately, so that the core is provably runnable headless.
`app.py` imports pygame at module scope, so a test that reached into it would be uncollectable in
CI — which is exactly how a `TypeError` in `_build_demo_world` survived unnoticed since the gene
matrix landed (#110). World-building lives in `clients.viewer.demo_world` for that reason, and
these tests import it directly.

Since #115 that module is config only — `core.world.assembly.build_world` does the wiring — so what
these tests cover is the viewer's *world config* being one a world can actually be built from, and
the frame path still reading it correctly now that entities move.
"""

import numpy as np

from clients.viewer.demo_world import build_demo_world
from clients.viewer.render import live_positions, species_colors, world_to_screen
from core.invariants import default_registry

_SCREEN = (640, 480)


class TestBuildDemoWorld:
    def test_allocates_every_entity_it_was_asked_for(self):
        world = build_demo_world(seed=0, n_entities=25)

        assert int(world.store.alive.sum()) == 25

    def test_entities_stand_on_the_terrain_surface(self):
        """§2.6 stores z from the start but movement is surface-locked, so a demo entity's z is
        the ground beneath it rather than an independent coordinate it could drift from.
        """
        world = build_demo_world(seed=1, n_entities=25)
        alive = world.store.alive

        surface = world.terrain.elevation_at(world.store.x[alive], world.store.y[alive])

        np.testing.assert_allclose(world.store.z[alive], surface, rtol=1e-6)

    def test_a_freshly_built_world_satisfies_every_invariant_checkable_today(self):
        world = build_demo_world(seed=2, n_entities=50)
        registry = default_registry(
            0.0, world.terrain.world_width, 0.0, world.terrain.world_height, plants=world.plants
        )

        registry.check_all(world.store, world.loop.tick_count)  # must not raise

    def test_the_same_seed_rebuilds_the_same_world(self):
        """The simulation is non-deterministic by design (§2.2), but *generation* is seeded so a
        crash can be replayed. A demo world that could not be rebuilt would make the viewer
        useless for exactly the diagnosis §3.3 exists for.
        """
        first = build_demo_world(seed=7, n_entities=30)
        second = build_demo_world(seed=7, n_entities=30)

        np.testing.assert_array_equal(first.store.x, second.store.x)
        np.testing.assert_array_equal(first.store.y, second.store.y)
        np.testing.assert_array_equal(first.store.genes, second.store.genes)

    def test_the_loop_advances_the_store_it_was_built_around(self):
        world = build_demo_world(seed=3, n_entities=10)
        registry = default_registry(
            0.0, world.terrain.world_width, 0.0, world.terrain.world_height, plants=world.plants
        )

        world.loop.advance(5)

        assert world.loop.tick_count == 5
        registry.check_all(world.store, world.loop.tick_count)  # must not raise
        # The loop now runs the settled order (#115), so five ticks are five ticks of a live world:
        # ages advance, pools drain, and the position snapshot the renderer interpolates from is
        # the one this store holds.
        np.testing.assert_array_equal(world.loop.current_positions[0], world.store.x)
        assert (world.store.age[world.store.alive] == 5).all()


class TestFramePath:
    def test_one_frames_worth_of_rendering_runs_headlessly(self):
        """Everything `run()` does per frame except the pygame calls themselves.

        The bug this file exists for was not in any of these functions — each is unit-tested —
        but in the seam between the world and them, which nothing exercised. Chaining them over a
        real demo world is what makes a shape, dtype or signature drift fail here rather than at
        the first keypress.
        """
        world = build_demo_world(seed=4, n_entities=40)
        world.loop.advance(1)

        x, y, _z, drawn = live_positions(
            world.loop.previous_positions,
            world.loop.previous_row_ids,
            world.loop.current_positions,
            world.loop.current_row_ids,
            0.5,
        )
        px, py = world_to_screen(
            x, y, world.terrain.world_width, world.terrain.world_height, *_SCREEN
        )
        colors = species_colors(world.store.species_id[drawn])

        n_live = int(world.store.alive.sum())
        assert px.shape == py.shape == (n_live,)
        assert ((0 <= px) & (px < _SCREEN[0])).all()
        assert ((0 <= py) & (py < _SCREEN[1])).all()
        assert colors.shape == (n_live, 3)
        # Every founder is given a species, so none falls back to the unset-species gray.
        assert (world.store.species_id[world.store.alive] >= 0).all()

    def test_a_death_removes_an_entity_from_the_frame(self):
        """End-to-end proof that the ghost is gone (#119).

        Nothing in an assembled world dies yet — #21 is unbuilt — so this releases a row directly.
        That is exactly what death will do, and the point is that the *frame path* stops drawing it
        rather than that any particular system caused it. Capacity-wide rendering was accidentally
        correct only because the demo world allocated exactly its capacity and never released a row.
        """
        world = build_demo_world(seed=4, n_entities=40)
        world.loop.advance(1)
        before = int(world.store.alive.sum())

        doomed = world.store.row_ids()[world.store.alive][:1]
        world.store.release(doomed)
        world.loop.advance(1)

        x, _y, _z, drawn = live_positions(
            world.loop.previous_positions,
            world.loop.previous_row_ids,
            world.loop.current_positions,
            world.loop.current_row_ids,
            0.5,
        )

        assert int(drawn.sum()) == before - 1
        assert len(x) == before - 1
        assert world.store.capacity == before, "capacity is unchanged; only the drawn set shrank"
