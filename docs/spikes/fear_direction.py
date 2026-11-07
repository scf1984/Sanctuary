"""Does fear steer once it has a direction, and does it cost the world anything (#265)?

Before #265 `Fear.appeal` returned flat ones, so fear added the same number to every option and a
frightened animal never moved away from anything. #265 gives it relative safety: danger read at
every candidate, `appeal = 1 − danger / max danger`. This runs the demo world and asks:

- **Viability.** Final and minimum population.
- **How often fear fires.** The share of the living with fear urgency above zero.
- **Whether flight happens.** For each animal frightened at a sample, the danger it perceives where
  it stands, and the same animal's reading one tick later. Flight shows as that reading falling
  more often, and further, than it does with fear flat.
- **The #207 spread worry.** How often fear's contribution *decided* the chosen option: the same
  Gumbel draw is taken with and without fear's contribution, and a sample counts as decided when
  the two picks differ. With a flat appeal that is zero by construction, which is the control.
  Also fear's spread across options (max − min of its contribution) against hunger's.

Run it on this branch and on master by pointing PYTHONPATH at each checkout; it reads nothing that
#265 added, so the same file measures both.

Throwaway spike code (CLAUDE.md §8.3), not imported by anything. Run from a repository root:

    PYTHONPATH=. python docs/spikes/fear_direction.py SEED [TICKS] [LABEL]

It prints one line per 20-tick sample and a summary line; docs/spikes/fear-direction.md has the
results.
"""

from __future__ import annotations

import sys

import numpy as np

from clients.viewer.demo_world import demo_world_config
from core.behaviour.candidates import Candidates
from core.selection import Selection
from core.world.assembly import build_world

N_ENTITIES = 200
EVERY = 20


def living(world) -> Selection:
    return Selection.from_mask(world.store.alive & (world.store.age >= 0))


def fear_of(world):
    return next(drive for drive in world.behaviour._drives if drive.name == "fear")


def danger_here(world, fear, selection: Selection) -> np.ndarray:
    """(n,) the summed gated danger each animal perceives where it stands: urgency's input, before
    the saturating curve. Computed here rather than read from #265's code so master measures it
    the same way."""
    expressed = world.genetics.expressed_columns(selection, fear._columns).astype(np.float64)
    smelled = world.scent.perceive(world.scent.broadcast(selection)).astype(np.float64)
    acuity = expressed[:, 0]
    total = np.zeros(len(selection))
    for direction in fear._aversion_positions:
        reading = acuity * (expressed[:, direction] * smelled).sum(axis=1)
        total += np.where(reading >= fear.config.detection_threshold, reading, 0.0)
    return total


def decision(world, selection: Selection, rng: np.random.Generator):
    """(decided, fear spread, hunger spread) for one fresh scoring of `selection`."""
    behaviour = world.behaviour
    headings = behaviour.candidate_headings(selection, rng)
    x, y = behaviour.candidate_positions(selection, headings)
    commitment, temperature = world.genetics.expressed_columns(
        selection, behaviour._choice_columns
    ).T
    total, contributions = behaviour.utilities(
        selection, headings, x, y, commitment.astype(np.float64)
    )
    t = temperature.astype(np.float64)[:, None]
    gumbel = rng.gumbel(size=total.shape)
    with_fear = np.argmax(total / t + gumbel, axis=1)
    without = np.argmax((total - contributions["fear"]) / t + gumbel, axis=1)
    fear = contributions["fear"]
    hunger = contributions["hunger"]
    return (
        with_fear != without,
        fear.max(axis=1) - fear.min(axis=1),
        hunger.max(axis=1) - hunger.min(axis=1),
    )


def main() -> None:
    seed = int(sys.argv[1])
    ticks = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
    label = sys.argv[3] if len(sys.argv) > 3 else "branch"
    world = build_world(demo_world_config(N_ENTITIES, seed), seed=seed)
    fear = fear_of(world)
    rng = np.random.default_rng(10_000 + seed)
    populations = []
    firing, decided_all, decided_frightened = [], [], []
    fear_spread, hunger_spread = [], []
    scared_urgency, appeal_spread = [], []
    deltas, fell = [], []
    tick = 0
    while tick < ticks:
        # Advance to one tick before the sample, measure, advance the last tick, measure again:
        # the second reading is the same animals one decision later.
        world.loop.advance(EVERY - 1)
        population = living(world)
        if not len(population):
            print(f"seed={seed} {label} tick={tick + EVERY - 1} extinct", flush=True)
            break
        # The cue field is rebuilt at the top of each tick from where animals stood then; refreshed
        # here so a reading between ticks sees where they stand now. The next tick rebuilds it from
        # the same positions, so this changes nothing the world does.
        world.scent.rebuild(population)
        mask = population.to_mask()
        here = Candidates(
            population,
            world.store.x[mask].astype(np.float64)[:, None],
            world.store.y[mask].astype(np.float64)[:, None],
        )
        urgency = fear.urgency(population, here)
        frightened = urgency > 0.0
        before = danger_here(world, fear, population)
        ids_before = world.store.row_ids()[population.to_indices()]
        decided, f_spread, h_spread = decision(world, population, rng)

        world.loop.advance(1)
        tick += EVERY
        after_population = living(world)
        world.scent.rebuild(after_population)
        after = danger_here(world, fear, after_population)
        ids_after = world.store.row_ids()[after_population.to_indices()]
        lookup = dict(zip(ids_after.tolist(), after.tolist()))
        pairs = [
            (lookup[i], b)
            for i, b, scared in zip(ids_before.tolist(), before.tolist(), frightened.tolist())
            if scared and i in lookup
        ]
        n = len(population)
        populations.append((tick, len(after_population)))
        firing.append(float(frightened.mean()))
        decided_all.append(float(decided.mean()))
        if frightened.any():
            decided_frightened.append(float(decided[frightened].mean()))
            fear_spread.append(float(f_spread[frightened].mean()))
            # Fear's spread is urgency times the spread of its appeal; split so the two can be
            # told apart (relative safety bounds the second by 1, nothing bounds the first below).
            scared_urgency.append(float(urgency[frightened].mean()))
            appeal_spread.append(float((f_spread[frightened] / urgency[frightened]).mean()))
        hunger_spread.append(float(h_spread.mean()))
        if pairs:
            a, b = np.array(pairs).T
            deltas.append(float(np.mean((a - b) / b)))
            fell.append(float(np.mean(a < b)))
        print(
            f"seed={seed} {label} tick={tick} n={len(after_population)} "
            f"firing={frightened.mean():.3f} decided={decided.mean():.4f} "
            f"decided|scared={decided[frightened].mean() if frightened.any() else float('nan'):.3f} "
            f"fear_spread={f_spread[frightened].mean() if frightened.any() else float('nan'):.3f} "
            f"hunger_spread={h_spread.mean():.3f} "
            f"danger_rel_change={deltas[-1] if pairs else float('nan'):+.3f} "
            f"danger_fell={fell[-1] if pairs else float('nan'):.3f} n_scared={int(frightened.sum())}/{n}",
            flush=True,
        )
    pop = np.array(populations)

    def mean(values):
        return float(np.mean(values)) if values else float("nan")

    print(
        f"SUMMARY seed={seed} {label} final_n={int(pop[-1, 1])} min_n={int(pop[:, 1].min())} "
        f"min_at={int(pop[pop[:, 1].argmin(), 0])} firing={mean(firing):.3f} "
        f"decided_all={mean(decided_all):.4f} decided_scared={mean(decided_frightened):.3f} "
        f"fear_spread={mean(fear_spread):.3f} hunger_spread={mean(hunger_spread):.3f} "
        f"danger_rel_change={mean(deltas):+.3f} danger_fell={mean(fell):.3f} "
        f"urgency_scared={mean(scared_urgency):.4f} appeal_spread_scared={mean(appeal_spread):.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
