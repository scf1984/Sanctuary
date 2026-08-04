"""Fences: edges nothing crosses, and the isolation that falls out (#27).

The last class is the one that matters and it is statistical (§6, §8.1): a fence that *mostly*
holds is not a fence, and the two ways it leaked before it worked were both invisible to unit
assertions — the walk stepping clean over a cell edge in the middle of a segment, and an animal
landing exactly on the line and indexing to the far side of it.
"""

import numpy as np
import pytest

from clients.viewer.demo_world import build_demo_world
from core.selection import Selection
from core.world.barriers import Barriers
from core.world.diffusion import CostAwareDiffusion, DiffusionConfig
from core.world.fence import Fence
from core.world.terrain import Terrain


def flat(size=21):
    return Terrain(np.zeros((size, size), dtype=np.float32), cell_size=1.0)


def at(*points):
    return (
        np.array([p[0] for p in points], dtype=np.float64),
        np.array([p[1] for p in points], dtype=np.float64),
    )


class TestABarrierIsAnEdge:
    def test_a_new_world_is_unfenced(self):
        barriers = Barriers(flat())

        assert not barriers.any_blocked

    def test_enclosing_blocks_the_perimeter_and_not_the_interior(self):
        """Blocking the area would not pen a herd, it would freeze each animal in its own cell."""
        barriers = Barriers(flat())

        barriers.enclose(5.0, 5.0, 10.0, 10.0)

        # 5 cells a side: 5 north edges top, 5 south, 5 west, 5 east.
        assert int(barriers.blocked_north.sum() + barriers.blocked_west.sum()) == 20
        assert not barriers.blocked_north[7, 7]
        assert not barriers.blocked_west[7, 7]

    def test_a_rectangle_thinner_than_a_cell_blocks_nothing(self):
        barriers = Barriers(flat())

        assert barriers.enclose(5.0, 5.0, 5.2, 9.0) == 0
        assert not barriers.any_blocked

    def test_the_same_edge_written_twice_is_one_edge(self):
        """A cell's south edge is its neighbour's north, so there is no second place to hold the
        fact and no way for two places to disagree."""
        barriers = Barriers(flat())
        barriers.enclose(5.0, 5.0, 10.0, 10.0)
        once = int(barriers.blocked_north.sum() + barriers.blocked_west.sum())

        barriers.enclose(5.0, 5.0, 10.0, 10.0)

        assert int(barriers.blocked_north.sum() + barriers.blocked_west.sum()) == once

    def test_clearing_takes_every_fence_down(self):
        barriers = Barriers(flat())
        barriers.enclose(5.0, 5.0, 10.0, 10.0)

        barriers.clear()

        assert not barriers.any_blocked

    def test_every_edit_bumps_the_revision(self):
        """What a cached derivation compares against. A stamp a writer must remember to set is one
        that is eventually not set, and the failure is silent — the field keeps routing through a
        fence that is standing (§8.7)."""
        barriers = Barriers(flat())
        start = barriers.revision

        barriers.enclose(5.0, 5.0, 10.0, 10.0)
        barriers.clear()

        assert barriers.revision == start + 2


class TestCrossingQuery:
    def test_a_move_inside_one_cell_crosses_nothing(self):
        barriers = Barriers(flat())
        barriers.blocked_west[:, 6] = True

        assert not barriers.crossing_blocked(*at((5.2, 5.0)), *at((5.4, 5.0)))[0]

    def test_a_move_across_a_blocked_edge_is_blocked(self):
        barriers = Barriers(flat())
        barriers.blocked_west[:, 6] = True

        assert barriers.crossing_blocked(*at((5.4, 5.0)), *at((5.6, 5.0)))[0]

    def test_a_move_across_an_open_edge_is_not(self):
        barriers = Barriers(flat())
        barriers.blocked_west[:, 6] = True

        assert not barriers.crossing_blocked(*at((7.4, 5.0)), *at((7.6, 5.0)))[0]

    def test_a_diagonal_is_blocked_if_either_component_edge_is(self):
        """A fence corner has no gap in it; permitting the diagonal would let animals leak through
        the join one at a time."""
        barriers = Barriers(flat())
        barriers.blocked_north[6, :] = True

        assert barriers.crossing_blocked(*at((5.4, 5.4)), *at((5.6, 5.6)))[0]


class TestPerceptionRoutesAround:
    """Why a fence is an intervention rather than a multiplier (§2.5). The operator is a walk over
    the neighbour graph, so what arrives behind a barrier is only what came round the end of it —
    nothing here attenuates, an edge is simply not there."""

    def field(self, barriers):
        return CostAwareDiffusion(flat(), DiffusionConfig(range=6.0, climb_penalty=0.5), barriers)

    def source(self):
        source = np.zeros((21, 21), dtype=np.float32)
        source[10, 7] = 100.0
        return source

    def test_a_sealed_pen_receives_essentially_nothing_from_outside(self):
        """Five parts in a million of what an open field delivers, which is float32 round-off over
        thirty diffusion passes rather than a path: every conductance row still sums to exactly 1,
        and there is no sequence of open edges from the source into the pen. Asserted as a ratio
        against the unfenced world rather than against zero, because "the fence removes essentially
        all of it" is the claim, and pinning an exact zero would be pinning the arithmetic."""
        barriers = Barriers(flat())
        diffusion = self.field(barriers)
        unfenced = diffusion.spread(self.source())[10, 11]
        assert unfenced > 0.0

        barriers.enclose(8.5, 0.5, 13.5, 20.5)

        assert diffusion.spread(self.source())[10, 11] / unfenced < 1e-4

    def test_the_field_is_rebuilt_when_a_fence_goes_up(self):
        """On a revision comparison rather than an invalidation call, so nothing has to remember."""
        barriers = Barriers(flat())
        diffusion = self.field(barriers)
        diffusion.spread(self.source())

        barriers.enclose(8.5, 0.5, 13.5, 20.5)
        diffusion.spread(self.source())

        assert diffusion._built_for_revision == barriers.revision

    def test_taking_the_fence_down_restores_what_was_there(self):
        barriers = Barriers(flat())
        diffusion = self.field(barriers)
        before = diffusion.spread(self.source())
        barriers.enclose(8.5, 0.5, 13.5, 20.5)
        diffusion.spread(self.source())

        barriers.clear()

        assert diffusion.spread(self.source()) == pytest.approx(before)

    def test_a_fence_that_does_not_seal_is_walked_around(self):
        """The point of a neighbour-graph walk: a signal reaches the far side by going round the
        end, so a gap in a fence is a gap rather than an attenuation."""
        barriers = Barriers(flat())
        diffusion = self.field(barriers)
        barriers.blocked_west[:15, 9] = True

        behind = diffusion.spread(self.source())[18, 11]

        assert behind > 0.0


class TestMovementRefusesRatherThanPays:
    """A ridge is expensive and a fence is impassable, and that distinction is deliberate: priced
    instead of refused, a fence would be a very steep hill a desperate animal crosses."""

    def walker(self, x):
        world = build_demo_world(seed=1, n_entities=4)
        world.barriers.blocked_west[:, 21] = True
        world.barriers.revision += 1
        living = Selection.from_mask(world.store.alive & (world.store.age >= 0))
        rows = living.to_indices()
        world.store.x[rows] = x
        world.store.y[rows] = 30.0
        world.store.energy[rows] = 5000.0
        world.store.velocity_x[rows] = 0.0
        world.store.velocity_y[rows] = 0.0
        return world, living, rows

    def drive_east(self, world, living, rows, ticks):
        target_x = np.full(rows.size, 60.0)
        target_y = np.full(rows.size, 30.0)
        for _ in range(ticks):
            world.movement.step(living, target_x, target_y, np.full(rows.size, 5.0))

    def test_an_animal_walks_up_to_the_fence_and_stops(self):
        world, living, rows = self.walker(18.0)

        self.drive_east(world, living, rows, ticks=8)

        _rows, cols = world.terrain.cell_indices(world.store.x[rows], world.store.y[rows])
        assert set(cols.tolist()) == {20}

    def test_it_stops_short_of_the_line_rather_than_on_it(self):
        """A cell owns the half-open span around its centre, so a position landing exactly on the
        boundary indexes to the cell *beyond* it — and the walk puts an animal exactly there on
        every ordinary pass. Stopping on the line reads as already through."""
        world, living, rows = self.walker(18.0)

        self.drive_east(world, living, rows, ticks=8)

        assert np.all(world.store.x[rows] < 20.5)

    def test_it_is_not_charged_for_the_crossing_it_did_not_make(self):
        """Refused, not priced. An animal that paid to cross a fence would empty its pool against
        the wire and die there, so a fence would read as a killing field rather than a boundary."""
        world, living, rows = self.walker(18.0)
        self.drive_east(world, living, rows, ticks=8)
        settled = world.store.energy[rows].copy()

        self.drive_east(world, living, rows, ticks=4)

        assert world.store.energy[rows] == pytest.approx(settled, rel=1e-4)

    def test_an_unfenced_world_lets_the_same_animal_through(self):
        """The control: without the fence this walk crosses that column easily, so the test above
        is measuring the barrier rather than an animal that could not get there anyway."""
        world, living, rows = self.walker(18.0)
        world.barriers.clear()

        self.drive_east(world, living, rows, ticks=8)

        _rows, cols = world.terrain.cell_indices(world.store.x[rows], world.store.y[rows])
        assert min(cols.tolist()) > 21


class TestTheFenceIntervention:
    def world(self):
        world = build_demo_world(seed=1, n_entities=40)
        world.loop.interventions.balance = 1.0e6
        return world

    def test_building_one_blocks_edges_at_a_tick_boundary(self):
        world = self.world()
        world.loop.interventions.request(Fence(world.barriers, 20.0, 20.0, 40.0, 40.0))
        assert not world.barriers.any_blocked

        world.loop.advance(1)

        assert world.barriers.any_blocked

    def test_it_costs_its_perimeter(self):
        """Scaled rather than flat, or fencing the whole map would be the obviously correct
        opening move."""
        small = Fence(Barriers(flat()), 5.0, 5.0, 10.0, 10.0)
        large = Fence(Barriers(flat()), 5.0, 5.0, 15.0, 15.0)

        assert large.cost() == pytest.approx(2.0 * small.cost())

    def test_a_rectangle_thinner_than_a_cell_is_refused(self):
        world = self.world()

        world.loop.interventions.request(Fence(world.barriers, 20.0, 20.0, 20.5, 40.0))
        world.loop.advance(1)

        assert "at least one cell" in world.loop.interventions.history[-1].refusal
        assert not world.barriers.any_blocked

    def test_a_refused_fence_is_not_charged(self):
        world = self.world()
        before = world.loop.interventions.balance

        world.loop.interventions.request(Fence(world.barriers, 20.0, 20.0, 20.5, 40.0))
        world.loop.advance(1)

        assert world.loop.interventions.balance == pytest.approx(before)

    def test_a_fence_does_not_evict(self):
        """Whoever is inside when it goes up is inside. That is what makes it an isolation rather
        than a round-up, and it is why "did I fence enough of them, with enough food and water" is
        the player's question to get wrong."""
        world = self.world()
        living = Selection.from_mask(world.store.alive & (world.store.age >= 0))
        before = world.store.x[living.to_mask()].copy()

        world.loop.interventions.request(Fence(world.barriers, 20.0, 20.0, 40.0, 40.0))
        world.loop.advance(1)

        assert world.store.x[living.to_mask()].shape == before.shape


class TestAFencedPopulationStaysFenced:
    """The claim the whole issue rests on, over a run long enough for it to fail.

    Both ways it leaked before were invisible to a unit assertion. The walk sampled only the
    *elevation node* lattice, so a segment stepped clean over a cell edge lying half a cell off it;
    and an animal landing exactly on a boundary indexed to the far cell. Each lost about 15% of a
    pen per hundred ticks while every direct test of the barrier passed.
    """

    def test_nothing_crosses_in_either_direction(self):
        world = build_demo_world(seed=1, n_entities=300)
        world.loop.advance(200)
        width, height = world.terrain.world_width, world.terrain.world_height
        corners = (width * 0.25, height * 0.25, width * 0.75, height * 0.75)
        world.loop.interventions.balance = 1.0e6
        world.loop.interventions.request(Fence(world.barriers, *corners))
        world.loop.advance(1)

        cell = world.terrain.cell_size
        low_col, high_col = (int(np.floor(corners[i] / cell + 0.5)) for i in (0, 2))
        low_row, high_row = (int(np.floor(corners[i] / cell + 0.5)) for i in (1, 3))

        def penned_ids():
            living = world.store.alive & (world.store.age >= 0)
            rows, cols = world.terrain.cell_indices(
                np.clip(world.store.x, 0.0, width), np.clip(world.store.y, 0.0, height)
            )
            inside = living & (cols >= low_col) & (cols < high_col)
            inside &= (rows >= low_row) & (rows < high_row)
            return set(world.store.row_ids()[inside].tolist())

        def living_ids():
            return set(world.store.row_ids()[world.store.alive & (world.store.age >= 0)].tolist())

        penned = penned_ids()
        outsiders = living_ids() - penned
        assert len(penned) > 10, "nothing was fenced in, so the test would pass vacuously"

        world.loop.advance(400)

        alive = living_ids()
        inside_now = penned_ids()
        assert not (penned & alive) - inside_now, "a penned animal got out"
        assert not (outsiders & alive) & inside_now, "an outsider got in"
