"""Per-species population concentration over the terrain grid (#22).

The animal counterpart to `core.ecology.plants`: a field, advanced as whole-array operations, that
answers "how much of species *s* is around here" without any per-pair work. Where the plant field
exists because plant *identity* buys the ecology nothing, this one exists because several questions
that look pairwise are not — mate-finding (#20), a predator locating prey (#19), and threat
perception (#22) all want a local abundance, not a list of individuals.

**This is deliberately not fear's private property.** #22 is its first reader and weights it by a
threat matrix, but nothing here knows what danger is. Burying it inside a drive would leave #19 and
#20 either reaching into a drive or building their own copy, which is the duplication CLAUDE.md
§7.1 warns about arriving by a different road.

**Why a field and not a spatial query.** A per-observer nearest-neighbour query over the whole
population measured 6.3 s/tick at 100,000 entities against a 1 s tick — see #96 for the numbers and
both implementations. This is O(n) in the population and O(cells) in the blur, with no per-pair term
at all. For the scent channel it is also the *physically* right model, not merely the affordable
one: scent diffuses, so a diffused field is what scent is (CLAUDE.md §2.5).

Concentration is currently isotropic. Wind advection — which makes the plume asymmetric and a
downwind approach stealthy — is #97, blocked on a wind field that does not yet exist. It enters as a
drift term in `rebuild()`; nothing outside this module changes when it does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.entities.store import EntityStore
from core.genetics.species import SpeciesRegistry
from core.selection import Selection
from core.world.terrain import Terrain


@dataclass(frozen=True)
class PopulationsConfig:
    """Per-world tuning for the population field (CLAUDE.md §2.1: a table, not scattered literals).

    diffusion_range: world units. How far a creature's presence spreads before it is negligible —
        the plume's characteristic length, not a hard cutoff. Must be positive; a zero range would
        make concentration a raw per-cell headcount, so an animal one cell away would be as
        invisible as one across the map.
    """

    diffusion_range: float

    def __post_init__(self) -> None:
        if self.diffusion_range <= 0:
            raise ValueError(
                f"diffusion_range must be positive, got {self.diffusion_range}; see the config "
                "docstring — zero collapses the field to a per-cell headcount"
            )


class Populations:
    """Diffused per-species presence over the terrain grid.

    concentration: ``(n_species, height, width)`` float32, entities per cell after diffusion.
        Aligned cell-for-cell with `terrain.heights`. Rebuilt from entity positions each tick;
        this holds no state between rebuilds, so a stale field is impossible rather than merely
        unlikely.

    Memory is ``n_species × cells``: 8 MB at 50 species on a 200×200 grid, 80 MB at 500. That is
    the one term that grows with speciation, which CLAUDE.md §2.3 warns must not degrade — it is
    left dense because blurring a sparse field spreads its support every pass, and 80 MB is
    affordable. Species with no living members cost nothing beyond their (zeroed) plane, since
    binning is driven by the entities that exist; extinction is permanent (§2.7) and a long-lived
    world would otherwise pay for every species it has ever had, forever.

    float32 rather than the plant field's float64: this is rebuilt from scratch every tick and
    carries no conservation invariant, so there is no pool for rounding to drift against.
    """

    concentration: np.ndarray

    def __init__(
        self, terrain: Terrain, species: SpeciesRegistry, config: PopulationsConfig
    ) -> None:
        self.terrain = terrain
        self.species = species
        self.config = config
        self.concentration = np.zeros(
            (species.n_species, *terrain.heights.shape), dtype=np.float32
        )

    def rebuild(self, store: EntityStore, population: Selection) -> None:
        """Re-bin `population` by species and cell, then diffuse.

        Sized to the registry on every call, so a species registered since the last rebuild — by
        speciation, which is a mask row and an id write (§2.3) — simply gets a plane. Nothing here
        needs telling that the world speciated.
        """
        rows = population.to_mask()
        shape = self.terrain.heights.shape
        density = np.zeros((self.species.n_species, *shape), dtype=np.float32)

        species_ids = store.species_id[rows].astype(np.int64)
        grid_rows, grid_cols = self._cell_indices(store.x[rows], store.y[rows])
        # One scatter-add over a flattened (species, row, col) index: binning a mixed-species
        # population of any size is a single vectorized pass, never a loop over species (§2.3).
        flat = (species_ids * shape[0] + grid_rows) * shape[1] + grid_cols
        np.add.at(density.reshape(-1), flat, 1.0)

        self.concentration = _diffuse(
            density, self.config.diffusion_range / self.terrain.cell_size
        )

    def sample(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """(n, n_species) float32: every species' concentration at each world position.

        Returned for all species at once rather than one species at a time, because the callers
        that matter weight across species — fear against a threat matrix, a predator across its
        diet — and asking per species would reintroduce the per-species loop §2.3 rejects.

        Sampled to the containing cell rather than interpolated, matching `Plants.biomass_at`: a
        diffused field is already smooth at cell scale, so interpolation would add cost and no
        information.

        The result is ``n × n_species``, which is 20 MB at 100,000 entities and 50 species. A
        caller scoring the whole population at a much larger species count should sample in
        chunks; nothing here does, because nothing yet needs it (§8.2).
        """
        grid_rows, grid_cols = self._cell_indices(x, y)
        return self.concentration[:, grid_rows, grid_cols].T

    def _cell_indices(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Grid (row, col) of the cell containing each world position.

        Raises ValueError outside the world, matching `Plants._cell_indices` — a position off the
        map is a bug in whatever moved the entity, and clamping it to an edge cell would pile
        phantom animals against the border (§8.7).
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        width = self.terrain.world_width
        height = self.terrain.world_height
        if not (np.all((x >= 0) & (x <= width)) and np.all((y >= 0) & (y <= height))):
            raise ValueError("position outside terrain bounds")

        cell_size = self.terrain.cell_size
        # floor(v + 0.5) rather than np.round, which is banker's rounding: a position exactly on a
        # cell boundary would otherwise land in different cells depending on the boundary's parity.
        cols = np.floor(x / cell_size + 0.5).astype(np.int64)
        rows = np.floor(y / cell_size + 0.5).astype(np.int64)
        return rows, cols


def _diffuse(density: np.ndarray, radius_in_cells: float) -> np.ndarray:
    """Spread each species' plane by three successive box blurs.

    Three boxes rather than a true Gaussian kernel: repeated box convolution converges on a
    Gaussian (the central limit theorem applied to the kernel), and each box is a pair of
    cumulative-sum differences, which costs O(cells) *regardless of blur width*. A direct kernel
    would cost O(cells × width) and buy a shape difference nothing here can observe — what matters
    ecologically is only that presence falls off smoothly with distance.

    Every plane blurs in the same vectorized pass, so species count never becomes a loop.
    """
    # Each box spans 2r+1 cells; three of them give a standard deviation near `radius_in_cells`,
    # so the config's range reads as the distance at which presence has mostly faded.
    half_width = max(1, int(round(radius_in_cells)))
    blurred = density.astype(np.float32)
    for _ in range(3):
        blurred = _box_blur(blurred, half_width)
    return blurred


def _box_blur(field: np.ndarray, half_width: int) -> np.ndarray:
    """One separable box blur over the last two axes, normalized by in-world cells only.

    Normalizing by the window's *in-world* cell count, rather than treating outside as zero, makes
    the map edge **reflecting**: presence that would have spread past the boundary stays inside
    instead. Zero-padding would make borders read as emptier than they are, and prey would find
    the world's edge spuriously safe — an artifact of the grid that selection would nonetheless
    act on. The reflecting edge errs the other way, so a cornered animal reads as slightly *more*
    concentrated, which is both the safer error and the physically sensible one.
    """
    weights = np.ones(field.shape[-2:], dtype=np.float32)
    return _box_axis(_box_axis(field, half_width, -1), half_width, -2) / _box_axis(
        _box_axis(weights, half_width, -1), half_width, -2
    )


def _box_axis(field: np.ndarray, half_width: int, axis: int) -> np.ndarray:
    """Running sum over a (2·half_width + 1) window along one axis, via cumulative sums.

    Padded with a leading zero so that `cumulative[hi] - cumulative[lo]` is a plain difference of
    prefix sums, and clipped at the ends so the window shortens at the boundary instead of
    wrapping — a wrap would carry scent from one edge of the world to the other.
    """
    length = field.shape[axis]
    cumulative = np.cumsum(field, axis=axis)
    zero = np.zeros_like(np.take(cumulative, [0], axis=axis))
    cumulative = np.concatenate([zero, cumulative], axis=axis)

    index = np.arange(length)
    hi = np.minimum(index + half_width + 1, length)
    lo = np.maximum(index - half_width, 0)
    return np.take(cumulative, hi, axis=axis) - np.take(cumulative, lo, axis=axis)
