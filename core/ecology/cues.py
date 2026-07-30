"""The cue field: what the world smells of, and where (#22).

Every creature carries a position in a fixed-dimensional **cue space** — its signature, what it
smells and looks like — and broadcasts it at a strength of its own (CLAUDE.md §2.5). This module
accumulates those broadcasts over the terrain grid and diffuses them, so that "what is around me"
is a field lookup rather than a search through neighbours.

**It knows nothing about species, danger, or genes.** It is handed positions, emission strengths
and signature vectors, and returns concentrations — exactly as `Plants.perceive` is handed a
caller-computed radius rather than reading a sight gene. That is what lets three different
questions share it: fear reads it with an *aversion* vector (#22), a predator will read it with an
attraction toward prey (#19), and mate-finding will read it with an attraction toward the
searcher's own signature (#20). All three are the same dot product with a different vector.

**Why cue channels rather than one plane per species.** A per-species field would need a plane per
species and a weight per species pair, and speciation creates species at runtime — so the cost
would grow exactly where CLAUDE.md §2.3 says it must not, and the weights could never be genes,
because a gene "fear of species 47" cannot exist in a fixed, versioned vocabulary. Cue channels are
fixed in number, so speciation costs this module nothing at all.

**Why a field rather than a neighbour query.** For scent it is the physically right model — scent
diffuses, and wind advection (#97) enters as a drift term here rather than as a per-pair test. It
is also the affordable one: a per-observer nearest-threat query measured 6.3 s/tick at 100,000
entities against a 1 s tick (#96).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.entities.store import EntityStore
from core.genetics.service import Genetics
from core.genetics.registry import GeneRegistry, Unit
from core.selection import Selection
from core.world.terrain import Terrain


@dataclass(frozen=True)
class CueFieldConfig:
    """Per-world tuning for the cue field (CLAUDE.md §2.1: a table, not scattered literals).

    diffusion_range: world units. The characteristic distance a cue carries before it is
        negligible — not a hard cutoff. Must be positive; at zero the field collapses to a raw
        per-cell tally, so a creature one cell away would be as undetectable as one across the map.
    """

    diffusion_range: float

    def __post_init__(self) -> None:
        if self.diffusion_range <= 0:
            raise ValueError(
                f"diffusion_range must be positive, got {self.diffusion_range}; see the config "
                "docstring — zero collapses the field to a per-cell tally"
            )


class CueField:
    """Diffused cue concentration per channel over the terrain grid.

    concentration: ``(n_channels, height, width)`` float32, signature-units per cell after
        diffusion. Aligned cell-for-cell with `terrain.heights`. Rebuilt from scratch each tick, so
        a stale field is impossible rather than merely unlikely.
    self_response: ``(height, width)`` float32. The share of its own deposit that a creature
        standing in each cell reads back — see `sample_excluding_self`. Static, so it is computed
        once here rather than per tick.

    Memory is ``n_channels × cells``, which is fixed: 3 MB for 8 channels on a 200×200 grid, and it
    does not move when the world speciates. float32 rather than the plant field's float64 because
    this is rebuilt every tick and carries no conservation invariant, so there is no pool for
    rounding to drift against.
    """

    concentration: np.ndarray
    self_response: np.ndarray

    def __init__(self, terrain: Terrain, n_channels: int, config: CueFieldConfig) -> None:
        if n_channels < 1:
            raise ValueError("n_channels must be at least 1")

        self.terrain = terrain
        self.n_channels = n_channels
        self.config = config
        self._half_width = max(1, int(round(config.diffusion_range / terrain.cell_size)))
        self.concentration = np.zeros((n_channels, *terrain.heights.shape), dtype=np.float32)
        self.self_response = _self_response(terrain.heights.shape, self._half_width)

    def rebuild(
        self, x: np.ndarray, y: np.ndarray, emission: np.ndarray, signature: np.ndarray
    ) -> None:
        """Deposit every broadcaster's signature at its position, then diffuse.

        x, y:      (n,) world units — where each broadcaster stands.
        emission:  (n,) how loudly it broadcasts on this modality. Non-negative.
        signature: (n, n_channels) its position in cue space.

        A creature contributes ``emission × signature`` to its own cell, so being loud and being
        distinctive are different things: the first makes it easier to detect at all, the second
        makes it easier to tell apart from something else.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        emission = np.asarray(emission, dtype=np.float32)
        signature = np.asarray(signature, dtype=np.float32)
        if not x.shape == y.shape == emission.shape:
            raise ValueError("x, y and emission must be the same length")
        if signature.shape != (x.shape[0], self.n_channels):
            raise ValueError(
                f"signature must be {(x.shape[0], self.n_channels)}, got {signature.shape}"
            )
        if np.any(emission < 0):
            raise ValueError("emission must be non-negative")

        shape = self.terrain.heights.shape
        deposit = np.zeros((self.n_channels, *shape), dtype=np.float32)
        if x.shape[0] > 0:
            grid_rows, grid_cols = self._cell_indices(x, y)
            cell = grid_rows * shape[1] + grid_cols
            # One scatter-add over a flattened (channel, row, col) index, so depositing a whole
            # population is a single vectorized pass and channels never become a loop (§2.3).
            channel_offset = np.arange(self.n_channels, dtype=np.int64)[:, None] * (
                shape[0] * shape[1]
            )
            np.add.at(
                deposit.reshape(-1),
                (channel_offset + cell[None, :]).ravel(),
                (emission[:, None] * signature).T.ravel(),
            )

        self.concentration = _diffuse(deposit, self._half_width)

    def sample(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """(n, n_channels) float32: every cue channel's concentration at each world position.

        All channels at once rather than one at a time, because every caller weights across them —
        fear against an aversion vector, a predator against its prey's signature — and asking per
        channel would put a Python loop where a dot product belongs.

        Sampled to the containing cell rather than interpolated, matching `Plants.biomass_at`: a
        diffused field is already smooth at cell scale, so interpolating would cost more and say
        nothing new.
        """
        grid_rows, grid_cols = self._cell_indices(x, y)
        return self.concentration[:, grid_rows, grid_cols].T

    def sample_excluding_self(
        self, x: np.ndarray, y: np.ndarray, emission: np.ndarray, signature: np.ndarray
    ) -> np.ndarray:
        """`sample`, minus what each sampler contributed to its own reading.

        **A creature does not perceive itself.** Without this, any lineage whose aversion vector
        overlapped its own signature — every cannibal (CLAUDE.md §2.5) — would read as permanently
        terrified while standing alone in an empty world, and the emergent cannibalism the cue
        encoding exists to allow would be indistinguishable from a bug.

        Exact, not approximate. Diffusion is linear, so a sampler's contribution to its own cell is
        its deposit times the blur operator's diagonal there; and because the blur is a separable
        *normalized* box, that diagonal factorizes per axis and is precomputed as `self_response`.
        The arguments must be the ones passed to `rebuild` for those same samplers.
        """
        emission = np.asarray(emission, dtype=np.float32)
        signature = np.asarray(signature, dtype=np.float32)
        grid_rows, grid_cols = self._cell_indices(x, y)
        own = self.self_response[grid_rows, grid_cols]
        return self.sample(x, y) - own[:, None] * emission[:, None] * signature

    def _cell_indices(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Grid (row, col) of the cell containing each world position.

        Raises ValueError outside the world, matching `Plants._cell_indices` — a position off the
        map is a bug in whatever moved the entity, and clamping it to an edge cell would pile
        phantom creatures against the border (§8.7).
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


@dataclass(frozen=True)
class ScentGenes:
    """Which genes carry the scent modality (CLAUDE.md §2.5's reserved block).

    Named per world rather than assumed, as `MetabolismConfig.insulation_gene` is, because the
    vocabulary is per-world.

    emission_gene: how loudly a creature broadcasts on the scent modality.
    signature_genes: its position in cue space, in channel order.
    """

    emission_gene: str
    signature_genes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.signature_genes:
            raise ValueError("signature_genes must name at least one gene")


class Scent:
    """The scent modality: the gene matrix on one side, the cue field on the other.

    This exists so the gene names appear **once**. `CueField.rebuild` takes plain arrays and
    `sample_excluding_self` needs a creature's own deposit back in order to subtract it, so without
    a binder every caller would independently decide which genes mean "emission" and "signature" —
    and a caller that disagreed with another would produce a self-exclusion that silently
    subtracted the wrong thing. Here the deposit and the exclusion read the same two fields.

    One deposit serves every reader. Fear weights the result by aversion (#22); a predator will
    weight it by attraction toward its prey's signature (#19); mate-finding will weight it toward
    the searcher's own (#20). None of them rebuilds the field.
    """

    def __init__(
        self,
        store: EntityStore,
        genetics: Genetics,
        field: CueField,
        genes: GeneRegistry,
        scent: ScentGenes,
    ) -> None:
        if len(scent.signature_genes) != field.n_channels:
            raise ValueError(
                f"signature_genes names {len(scent.signature_genes)} genes but the cue field has "
                f"{field.n_channels} channels; they index the same space and must match"
            )
        self.store = store
        self.genetics = genetics
        self.field = field
        # Raise KeyError naming the vocabulary version if any gene does not exist. Broadcast
        # strength and a position in cue space are both bare numbers.
        self._emission_index = genes.index_of(scent.emission_gene, unit=Unit.DIMENSIONLESS)
        self._signature_indices = np.array(
            [genes.index_of(name, unit=Unit.DIMENSIONLESS) for name in scent.signature_genes],
            dtype=np.int64,
        )

    def rebuild(self, population: Selection) -> None:
        """Deposit every living creature's scent and diffuse it. Once per tick, before any read."""
        mask = population.to_mask()
        emission, signature = self._broadcast(population)
        self.field.rebuild(self.store.x[mask], self.store.y[mask], emission, signature)

    def perceive(self, selection: Selection) -> np.ndarray:
        """(len(selection), n_channels) float32: cue concentration each creature can smell.

        Excludes the creature's own scent — see `CueField.sample_excluding_self` for why that is
        not optional.
        """
        mask = selection.to_mask()
        emission, signature = self._broadcast(selection)
        return self.field.sample_excluding_self(
            self.store.x[mask], self.store.y[mask], emission, signature
        )

    def _broadcast(self, selection: Selection) -> tuple[np.ndarray, np.ndarray]:
        """Each creature's (emission, signature) from its *expressed* phenotype.

        Expression, not genotype, for the same reason upkeep follows it (#17): a species that does
        not express a signature gene neither carries that scent nor pays for it.
        """
        expressed = self.genetics.expressed(selection)
        return expressed[:, self._emission_index], expressed[:, self._signature_indices]


# Repeated box convolution converges on a Gaussian (the central limit theorem applied to the
# kernel) while each box stays a pair of cumulative-sum differences — O(cells) regardless of blur
# width, where a direct kernel would be O(cells × width). The shape difference between three boxes
# and a true Gaussian is not observable by anything in this simulation; what matters ecologically
# is only that a cue falls off smoothly with distance.
_BLUR_PASSES = 3


def _diffuse(deposit: np.ndarray, half_width: int) -> np.ndarray:
    """Spread every channel's plane. All planes blur in the same pass, so channels never loop."""
    blurred = deposit.astype(np.float32)
    for _ in range(_BLUR_PASSES):
        blurred = _box_blur(blurred, half_width)
    return blurred


def _box_blur(field: np.ndarray, half_width: int) -> np.ndarray:
    """One separable box blur over the last two axes, normalized by in-world cells only.

    Normalizing by the window's *in-world* cell count, rather than treating outside as zero, makes
    the map edge **reflecting**: a cue that would have spread past the boundary stays inside
    instead. Zero-padding would make borders read as emptier than they are, and prey would find the
    world's edge spuriously safe — an artifact of the grid that selection would nonetheless act on.
    The reflecting edge errs the other way, so a cornered animal reads as slightly *more*
    detectable, which is both the safer error and the physically sensible one.
    """
    weights = np.ones(field.shape[-2:], dtype=np.float32)
    return _box_axis(_box_axis(field, half_width, -1), half_width, -2) / _box_axis(
        _box_axis(weights, half_width, -1), half_width, -2
    )


def _box_axis(field: np.ndarray, half_width: int, axis: int) -> np.ndarray:
    """Running sum over a (2·half_width + 1) window along one axis, via cumulative sums.

    Padded with a leading zero so `cumulative[hi] - cumulative[lo]` is a plain difference of prefix
    sums, and clipped at the ends so the window shortens at the boundary instead of wrapping — a
    wrap would carry scent from one edge of the world to the other.
    """
    length = field.shape[axis]
    cumulative = np.cumsum(field, axis=axis)
    zero = np.zeros_like(np.take(cumulative, [0], axis=axis))
    cumulative = np.concatenate([zero, cumulative], axis=axis)

    index = np.arange(length)
    hi = np.minimum(index + half_width + 1, length)
    lo = np.maximum(index - half_width, 0)
    return np.take(cumulative, hi, axis=axis) - np.take(cumulative, lo, axis=axis)


def _self_response(shape: tuple[int, int], half_width: int) -> np.ndarray:
    """(height, width) float32: the diagonal of the diffusion operator.

    How much of a unit deposit made in a cell is still readable *in that same cell* after blurring.
    `sample_excluding_self` subtracts exactly this share so that nothing perceives itself.

    Computed per axis and multiplied, which is exact rather than an approximation: `_box_blur` is a
    normalized box along each axis independently, so the 2-D operator is the outer product of two
    1-D operators and its diagonal is the outer product of their diagonals. Each 1-D operator is
    built by blurring an identity matrix — O(length²) once at construction, against a grid axis in
    the hundreds.
    """
    return np.outer(_axis_diagonal(shape[0], half_width), _axis_diagonal(shape[1], half_width))


def _axis_diagonal(length: int, half_width: int) -> np.ndarray:
    """(length,) float32: the 1-D normalized-box operator's diagonal, after `_BLUR_PASSES` passes.

    Column j of `operator` is the response to a unit impulse at j, so the diagonal is what an
    impulse reads back at its own position.
    """
    counts = _box_axis(np.ones(length, dtype=np.float32), half_width, -1)
    operator = np.eye(length, dtype=np.float32)
    for _ in range(_BLUR_PASSES):
        operator = _box_axis(operator, half_width, 0) / counts[:, None]
    return np.diagonal(operator).copy()
