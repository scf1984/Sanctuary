"""Standing water laid over a terrain by hand, for tests that need to say exactly how deep it is.

`Water.generate` derives depth from the heightmap, which is right for a world and wrong for a test
that wants to price a step through water of a known depth on ground of a known shape: the two are
then not independent, and a test of what depth costs would also be a test of where a basin fills.
`Water` takes its arrays directly for snapshot restore (#31), so a test can do the same.

The flow arrays are inert placeholders. Nothing that reads depth reads them, and the one consumer
that does — `Plants`' soil moisture — builds its own `Water` in `tests/support/plants.py`.
"""

from __future__ import annotations

import numpy as np

from core.world.terrain import Terrain
from core.world.water import Water


def standing_water(terrain: Terrain, depth: np.ndarray | None = None) -> Water:
    """A `Water` over `terrain` holding `depth` (world units), or dry ground everywhere."""
    shape = terrain.heights.shape
    if depth is None:
        depth = np.zeros(shape, dtype=np.float32)
    return Water(
        depth=np.asarray(depth, dtype=np.float32),
        flow_direction=np.full(shape, -1, dtype=np.int8),
        flow_accumulation=np.ones(shape, dtype=np.float32),
        cell_size=terrain.cell_size,
    )
