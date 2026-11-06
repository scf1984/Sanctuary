"""Demo-world viability across #176: living count and mean thermal genes over a fixed horizon.

Reads whichever thermal genes the vocabulary has (`insulation` before #176, `heat_tolerance` and
`cold_tolerance` after), so the same script measures both sides of the change. The before column
in docs/spikes/thermal-divergence.md was taken by running it on the parent commit.

Throwaway spike code (CLAUDE.md §8.3). Run from a repository root with `PYTHONPATH=.`:

    python docs/spikes/thermal_viability.py SEED [TICKS] [EVERY]
"""

import sys
import time

import numpy as np

from clients.viewer.demo_world import build_demo_world
from core.selection import Selection

seed = int(sys.argv[1])
ticks = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
every = int(sys.argv[3]) if len(sys.argv) > 3 else 500

world = build_demo_world(seed=seed, n_entities=200)
names = list(world.genes.vocabulary.names)
thermal = [g for g in ("insulation", "heat_tolerance", "cold_tolerance") if g in names]
start = time.perf_counter()
for tick in range(0, ticks + 1, every):
    if tick:
        world.loop.advance(every)
    living = Selection.from_mask(world.store.alive & (world.store.age >= 0))
    n = len(living)
    genes = ""
    if n:
        phen = world.genetics.expressed(living)
        genes = " ".join(f"{g}={float(np.mean(phen[:, names.index(g)])):.3f}" for g in thermal)
    print(f"seed={seed} tick={tick} living={n} {genes}", flush=True)
    if not n:
        break
print(f"seed={seed} wall={time.perf_counter() - start:.0f}s", flush=True)
