"""The world the diagnostic viewer looks at: real terrain and water, a scatter of entities on it.

**Separate from `app.py` because `app.py` imports pygame at module scope.** CI installs `.[dev]`
and never the viewer extra — deliberately, since a run that never installs pygame is a standing
check that the core is runnable headless (CLAUDE.md §3) — so nothing importing `app.py` is
collectable there. World-building sitting inside that module is precisely why its `EntityStore`
call could go a whole gene-matrix release without `n_genes` and without anyone noticing (#110):
the one entry point that runs the simulation as a program was the one module no test could load.
The rule `render.py` already states — pygame lives in `app.py` and nowhere else — earns its keep
only if everything worth testing stays on this side of it.

This is a **demo scatter, not a world assembly.** The services all exist now — `Behaviour` (#22),
`Ecology` (#17), `Genetics` (#13), `Plants` (#18), `Movement` (#25), `Aging` (#109) — but building
them against one store and registering them in the settled tick order (§2.1) is #115's scope, and
a second assembly competing with that one is what §7.2 exists to prevent. Until then the loop runs
no systems: entities hold position, which still exercises the render and interpolation path this
viewer was built to prove out (§3.3).
"""

from __future__ import annotations

import numpy as np

from core.entities.store import EntityStore
from core.world.terrain import Terrain, TerrainConfig
from core.world.tick import TickLoop
from core.world.water import Water

# Width of the store's two column blocks. Nothing here registers a drive or reads a gene — this
# world has no services (see the module docstring) — and the store's constructor requires at least
# one column of each, so these are the minimum that constructs rather than a claim about how many
# drives or genes a real world has. #115's assembly sets both from what it actually registers.
_N_DRIVES = 1
_N_GENES = 1

_N_SPECIES = 5


def build_demo_world(
    seed: int, n_entities: int
) -> tuple[Terrain, Water, EntityStore, TickLoop]:
    """A freshly generated terrain, its derived water, and entities scattered across the surface.

    Generation is a pure function of `seed` (§2.2): the simulation itself is non-deterministic, but
    a world the viewer cannot rebuild is one whose crash cannot be replayed.
    """
    terrain = Terrain.generate(TerrainConfig(width=80, height=80, seed=seed))
    water = Water.generate(terrain)
    store = EntityStore(initial_capacity=n_entities, n_drives=_N_DRIVES, n_genes=_N_GENES)

    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, terrain.world_width, n_entities).astype(np.float32)
    y = rng.uniform(0.0, terrain.world_height, n_entities).astype(np.float32)
    # Surface-locked: §2.6 stores z from the start, but nothing flies or swims yet, so an entity's
    # z is the ground under it rather than a coordinate it can drift from.
    z = terrain.elevation_at(x, y).astype(np.float32)
    species_id = rng.integers(0, _N_SPECIES, n_entities).astype(np.int32)
    store.allocate(n_entities, x=x, y=y, z=z, species_id=species_id)

    return terrain, water, store, TickLoop(store, systems=())
