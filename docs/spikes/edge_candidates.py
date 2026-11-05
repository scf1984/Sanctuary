"""What the world edge does to the options an animal is offered, under each way of keeping them in (#209).

Throwaway spike code (CLAUDE.md §8.3), not imported by anything. Run from a worktree root with
`PYTHONPATH=.`. It patches `Behaviour.candidate_headings` with each variant in turn and measures:

1. **Folded options.** For animals standing exactly on the boundary (a corner, or mid-wall), the
   fraction of travelling options whose point clips onto the animal itself, which is a stay-put
   option under a travelling index.
2. **Distribution.** An *indifferent* population (every drive removed, so only the commitment bonus
   shapes a choice) is stepped through the real `choose` -> `chosen_target` -> `Movement.step`
   pipeline. With no drives, any departure from a uniform density is the boundary rule's doing.
   Reported as density in the edge band (within one look-ahead of a wall) and in the corner
   squares, each relative to its share of the area, plus the mean resultant length of movers'
   headings (directional skew; 0 is none).

Variants, all applied to the same evenly spaced, jittered headings:
- clip:    before #209. Points are clipped into the world and headings are left alone.
- shipped: clip, but a heading clipping would fold onto the animal is turned to the nearest
           heading whose full-reach point is inside (`Behaviour._unfold`).
- mirror:  every outward heading component is negated, so the option points inward at full reach.
- redraw:  every outward heading is replaced by a uniform draw from the headings that stay inside.
- turn:    every outward heading is turned to the nearest heading that stays inside.
"""

import numpy as np

from clients.viewer.demo_world import build_demo_world
from core.behaviour.service import Behaviour
from core.selection import Selection

SHIPPED = Behaviour.candidate_headings


def raw(self, selection, rng):
    n = len(selection)
    spacing = 2.0 * np.pi / self.config.n_candidates
    base = np.arange(self.config.n_candidates, dtype=np.float64) * spacing
    return base[None, :] + rng.uniform(0.0, spacing, size=n)[:, None]


def _outward(self, selection, headings):
    mask = selection.to_mask()
    x = self.store.x[mask].astype(np.float64)[:, None]
    y = self.store.y[mask].astype(np.float64)[:, None]
    reach = self.config.look_ahead
    tx, ty = x + reach * np.cos(headings), y + reach * np.sin(headings)
    w, h = self.terrain.world_width, self.terrain.world_height
    return (tx < 0.0) | (tx > w), (ty < 0.0) | (ty > h)


def mirror(self, selection, rng):
    headings = raw(self, selection, rng)
    out_x, out_y = _outward(self, selection, headings)
    cos = np.where(out_x, -np.cos(headings), np.cos(headings))
    sin = np.where(out_y, -np.sin(headings), np.sin(headings))
    return np.mod(np.arctan2(sin, cos), 2.0 * np.pi)


def redraw(self, selection, rng):
    headings = raw(self, selection, rng)
    for _ in range(64):
        out_x, out_y = _outward(self, selection, headings)
        out = out_x | out_y
        if not out.any():
            break
        headings = np.where(out, rng.uniform(0.0, 2.0 * np.pi, size=headings.shape), headings)
    return headings


def turn(self, selection, rng):
    """`_unfold`'s rule applied to every outward heading, not only the folded ones."""
    headings = raw(self, selection, rng)
    out_x, out_y = _outward(self, selection, headings)
    out = out_x | out_y
    # Force `_unfold` to treat each outward heading as folded by asking it about those alone.
    mask = selection.to_mask()
    x = self.store.x[mask].astype(np.float64)
    y = self.store.y[mask].astype(np.float64)
    w, h = self.terrain.world_width, self.terrain.world_height
    reach = self.config.look_ahead
    rows, cols = np.nonzero(out)
    here_x, here_y, wanted = x[rows], y[rows], headings[rows, cols]
    left = np.arccos(np.clip(-here_x / reach, -1, 1))
    right = np.arccos(np.clip((w - here_x) / reach, -1, 1))
    bottom = np.arcsin(np.clip(-here_y / reach, -1, 1))
    top = np.arcsin(np.clip((h - here_y) / reach, -1, 1))
    ends = np.stack([left, -left, right, -right, bottom, np.pi - bottom, top, np.pi - top], axis=1)
    ex, ey = here_x[:, None] + reach * np.cos(ends), here_y[:, None] + reach * np.sin(ends)
    s = 1e-9 * max(w, h)
    inside = (ex >= -s) & (ex <= w + s) & (ey >= -s) & (ey <= h + s)
    gap = np.abs(np.angle(np.exp(1j * (ends - wanted[:, None]))))
    best = ends[np.arange(len(rows)), np.argmin(np.where(inside, gap, np.inf), axis=1)]
    headings = headings.copy()
    headings[rows, cols] = np.mod(best, 2.0 * np.pi)
    return headings


VARIANTS = {"clip": raw, "shipped": SHIPPED, "mirror": mirror, "redraw": redraw, "turn": turn}


def folded(world, rng):
    """Fraction of travelling options folded onto the animal, for animals at corners and mid-wall."""
    b = world.behaviour
    w, h = b.terrain.world_width, b.terrain.world_height
    alive = np.flatnonzero(world.store.alive)
    spots = {"corner": [(0, 0), (w, 0), (0, h), (w, h)],
             "mid-wall": [(0, h / 2), (w, h / 2), (w / 2, 0), (w / 2, h)]}
    result = {}
    for kind, points in spots.items():
        share = []
        for px, py in points:
            world.store.x[alive] = px
            world.store.y[alive] = py
            population = Selection.from_mask(world.store.alive)
            x, y = b.candidate_positions(population, b.candidate_headings(population, rng))
            share.append((np.hypot(x[:, :-1] - x[:, -1:], y[:, :-1] - y[:, -1:]) == 0.0).mean())
        result[kind] = float(np.mean(share))
    return result["corner"], result["mid-wall"]


def wander(world, rng, ticks):
    """Step an indifferent population; report edge and corner density ratios and heading skew."""
    b, m = world.behaviour, world.movement
    b._drives = []
    w, h = b.terrain.world_width, b.terrain.world_height
    band = b.config.look_ahead
    edge_area = 1.0 - (w - 2 * band) * (h - 2 * band) / (w * h)
    corner_area = 4 * band * band / (w * h)
    edge_share, corner_share, sx, sy, n_moving = [], [], 0.0, 0.0, 0
    for t in range(ticks):
        population = Selection.from_mask(world.store.alive)
        b.choose(population, rng)
        tx, ty = b.chosen_target(population)
        m.step(population, tx, ty, b.chosen_urge(population))
        if t >= ticks // 2:
            mask = population.to_mask()
            x = world.store.x[mask].astype(np.float64)
            y = world.store.y[mask].astype(np.float64)
            near_x = (x < band) | (x > w - band)
            near_y = (y < band) | (y > h - band)
            edge_share.append((near_x | near_y).mean())
            corner_share.append((near_x & near_y).mean())
            moving = world.store.choice_moving[mask]
            heading = world.store.choice_heading[mask][moving].astype(np.float64)
            sx += np.cos(heading).sum()
            sy += np.sin(heading).sum()
            n_moving += int(moving.sum())
    return (
        float(np.mean(edge_share) / edge_area),
        float(np.mean(corner_share) / corner_area),
        float(np.hypot(sx, sy) / max(n_moving, 1)),
    )


def main(seeds=(1, 2, 3, 4), n=2000, ticks=400):
    print(f"n={n} ticks={ticks} seeds={seeds}")
    print("variant  | folded:corner mid-wall | density:edge  corner | skew")
    for name, variant in VARIANTS.items():
        Behaviour.candidate_headings = variant
        rows = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            world = build_demo_world(seed=seed, n_entities=n)
            # Founders spread uniformly, so the run starts at the density it is judged against.
            alive = world.store.alive
            world.store.x[alive] = rng.uniform(0, world.terrain.world_width, alive.sum())
            world.store.y[alive] = rng.uniform(0, world.terrain.world_height, alive.sum())
            density = wander(world, rng, ticks)
            rows.append(folded(world, rng) + density)
        mean = np.mean(rows, axis=0)
        spread = np.std(rows, axis=0)
        print(
            f"{name:8s} | {mean[0]:13.4f} {mean[1]:8.4f} | {mean[2]:6.3f}±{spread[2]:.3f} "
            f"{mean[3]:6.3f}±{spread[3]:.3f} | {mean[4]:.4f}"
        )
    Behaviour.candidate_headings = SHIPPED


if __name__ == "__main__":
    main()
