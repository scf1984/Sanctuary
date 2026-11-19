"""Emergent speciation at a genetic distance threshold (CLAUDE.md §2.5, issue #16).

Isolation is the payoff of the whole design: two sub-populations that stop exchanging genes drift
apart under `core.genetics.inheritance`, their signature centroids separate (measured by
`core.genetics.distance`), and past a threshold they are no longer one species. This module is the
mechanic and nothing more -- it decides *that* a split happened and records it. Naming is the
player's (decided on #16, 2026-11-06) and is presentation, so `split()` returns an opaque id and
stores no name.

What a split costs: one appended row on the species registry's `(n_species, n_genes)` mask table,
and one vectorized write into the `species_id` column. Nothing entities-shaped is reallocated,
copied, or restructured -- no gene row moves, no entity row moves, capacity does not change
(CLAUDE.md §2.3). That is the whole reason species are an id plus a mask instead of a storage
layout, and it is asserted directly in the tests by array identity, not inferred.

Reproductive isolation degrades, it does not switch (issue #16's "no cliff-edge"). A couple is
fully fertile up to an *ordinary* within-species distance and falls linearly to zero at the
*threshold* (the owner's decision on #16). A split fires when the two groups' centroids sit
`√(threshold² − ordinary²)` apart, which is where a typical pair drawn across them reaches the
threshold: such a pair is the centroid gap plus an ordinary within-group spread, in quadrature. So
by the time a split is recorded, cross-group couples were already infertile, and the species-id
gate that takes over removes an already-vanished probability rather than a live one.

Fertility used to fall linearly from zero distance. Measured on the demo world, two animals of one
species sit 6–10 mutation steps apart, so a threshold keeping them 80% fertile is about 80 steps,
and no isolated pen ever drifted that far. With ordinary couples fully fertile, pens of 20, 40 and
100 reached a split in a median of 60, 100 and 200 generations (`speciation-signature.md`), and a
larger pen takes longer, which is real peripatric speciation rather than a tuning artefact.

**Distance reads only the scent-signature genes** (the owner's decision on #16). They are neutral,
so they drift like a clock, and mate-finding already reads them (#188), so a split records an
isolation that has already begun in behaviour. The cost, accepted deliberately: two populations
diverging only in diet, size or life history stay one species. Every signature gene is founded on
one scale, so there is nothing to commensurate (#193 resolved by scope).

**The threshold is counted in mutation steps, not signature units** (also the owner's, measured in
`docs/spikes/speciation-signature.md`). `mutability` is a gene and selection moved it sixfold up in
one demo world and to a third in another, and within-pool spread follows it (σ ≈ 1.8 × mutability at
mutation–drift balance, `speciation-drift.md`). An absolute threshold would read a lineage that
evolved high mutability as diverged with no isolation behind it, and make its own members infertile
with each other. Dividing by the animals' expressed mutability gives one dimensionless number that
means the same thing in every world. The species' *measured* spread was the other way to be
relative, and it fails: it includes the very substructure a split detects, so it grows as two demes
drift apart and cross-deme pairs never fall outside it.

**A 2-way clustering of each species' signatures proposes a split** (the owner's choice over sealed
fence regions, so that terrain can isolate as well as fences). It only nominates: `has_diverged` and
a per-world minimum group size decide, and a false nomination costs one centroid distance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.genetics.distance import between, centroid_between
from core.genetics.registry import GeneRegistry, Unit
from core.genetics.service import Genetics
from core.selection import Selection


class MixedSpeciesError(Exception):
    """A selection that had to belong to a single species contained more than one."""


class Lineage:
    """Which species each species split off from.

    Roots -- species registered directly rather than produced by `split()` -- have no parent and
    are not recorded here at all; `parent_of` returns None for them, which is what terminates an
    `ancestry` walk. Only the parent link is stored: it is what ancestry display needs, and this
    module has no clock of its own to timestamp anything with (CLAUDE.md §2.1 -- the tick counter
    lives in the world loop, and inventing a parameter no caller passes today would violate §8.2).
    """

    def __init__(self) -> None:
        self._parent_by_species: dict[int, int] = {}

    def record_split(self, species_id: int, parent_species_id: int) -> None:
        """Record that `species_id` split off from `parent_species_id`.

        Raises ValueError on re-recording a species: a species id is created by exactly one
        split and its parent never changes, so a second record is a caller bug, not an update
        (CLAUDE.md §8.7).
        """
        if species_id in self._parent_by_species:
            raise ValueError(
                f"species {species_id} already descends from "
                f"{self._parent_by_species[species_id]}"
            )
        self._parent_by_species[species_id] = parent_species_id

    def parent_of(self, species_id: int) -> int | None:
        """The species `species_id` split off from, or None if it is a root."""
        return self._parent_by_species.get(species_id)

    def ancestry(self, species_id: int) -> tuple[int, ...]:
        """The chain from the root ancestor down to `species_id`, inclusive.

        A root's ancestry is just itself. Because every recorded parent is a species that already
        existed when the split happened, and `record_split` refuses to re-parent, the parent
        links form a forest -- so this walk always terminates at a root.
        """
        chain = [species_id]
        parent = self.parent_of(species_id)
        while parent is not None:
            chain.append(parent)
            parent = self.parent_of(parent)
        return tuple(reversed(chain))

    def parents(self) -> dict[int, int]:
        """Every recorded split, as `{daughter: parent}`. A copy, for a snapshot to store (#31)."""
        return dict(self._parent_by_species)

    def restore(self, parents: dict[int, int]) -> None:
        """Replace the record with a saved one (#31). A split is state, not a derivation: which
        species came from which cannot be recomputed from the masks or the ids."""
        self._parent_by_species = {int(k): int(v) for k, v in parents.items()}


def mutation_step(genetics: Genetics, rows: np.ndarray) -> np.ndarray:
    """(len(rows),) float32: each animal's expressed `mutability`, the unit distance is counted in.

    Heritable rather than aged, for the reason `distance` reads the heritable phenotype (#239):
    this compares genomes, and age is not one.
    """
    column = np.array([genetics.expression.mutability_index], dtype=np.int64)
    return genetics.heritable_columns_at(np.asarray(rows, dtype=np.int64), column)[:, 0]


def species_mutation_step(genetics: Genetics, population: Selection) -> np.ndarray:
    """(n_species,) float64: each species' mean expressed `mutability` over `population`.

    This is the ruler a couple's distance is read against, and it is the *species'*, never the
    couple's own (#304). Founders' mutability varies threefold, so a per-couple step made the least
    mutable couples count the same distance as more steps: at founding on the demo world they were
    78% fertile against 100% for the most mutable, a fertility tax on low mutability nobody
    designed, which left 4 of 10 demo worlds stalled after the opening crash. One ruler per species
    makes every couple in it read distance alike, and is still relative to how mutable that
    lineage has evolved to be, which is why the threshold is counted in steps at all.

    A species with no member in `population` gets 0, which `interbreeding_probability` reads as
    "no step to count in". Nothing breeds in it, so nothing reads it.
    """
    rows = population.to_indices()
    species = genetics.species_ids_at(rows).astype(np.int64)
    n_species = genetics.species.n_species
    totals = np.bincount(
        species, weights=mutation_step(genetics, rows).astype(np.float64), minlength=n_species
    )
    counts = np.bincount(species, minlength=n_species)
    return np.divide(totals, counts, out=np.zeros(n_species), where=counts > 0)


def split_gap(ordinary: float, threshold: float) -> float:
    """The centroid gap, in mutation steps, at which a typical cross-group pair reaches `threshold`.

    A pair drawn across two groups is their centroid gap plus each animal's ordinary scatter about
    its own centroid, and the two add in quadrature, so a typical cross pair sits at
    `√(gap² + ordinary²)`. Setting that to the threshold is what makes "interbreeding has reached
    zero" and "a split fires" one event rather than two numbers that could disagree.
    """
    return float(np.sqrt(threshold**2 - ordinary**2))


def has_diverged(
    genetics: Genetics,
    a: Selection,
    b: Selection,
    ordinary: float,
    threshold: float,
    columns: np.ndarray,
) -> bool:
    """Whether sub-populations `a` and `b` have drifted far enough apart to be separate species.

    Compares *centroids*, not individuals: speciation is a property of two populations having
    moved apart, and a single outlier pair in either one says nothing about whether the gene pools
    have separated. `a` and `b` may differ in size; both must be non-empty (`centroid_between`
    raises otherwise, CLAUDE.md §8.7).

    ordinary, threshold: dimensionless, in mutation steps (see `SpeciationConfig`). The centroids
        must sit `split_gap(ordinary, threshold)` steps apart, a step being the two populations'
        mean expressed mutability -- the species' own ruler, since `a` and `b` are the two halves
        of one species, and the one `interbreeding_probability` reads (#304): the gap at which `interbreeding_probability` has already reached
        zero for a typical cross pair, so the two cannot disagree into "incompatible but not yet
        split" or "split but still fertile".
    columns: the vocabulary indices distance reads (the signature block in every world so far).
    """
    rows = np.concatenate([a.to_indices(), b.to_indices()])
    step = float(mutation_step(genetics, rows).mean())
    gap = split_gap(ordinary, threshold) * step
    return bool(centroid_between(genetics, a, b, columns) >= gap)


def split(genetics: Genetics, lineage: Lineage, diverged: Selection) -> int:
    """Make `diverged` a new species descended from its current one; returns the new species id.

    The daughter species starts with a copy of its parent's expression mask: a split changes who
    breeds with whom, not which genes anyone expresses. Divergence in expression comes afterwards,
    from the two populations' masks being edited independently -- and every gene stays stored and
    inherited in both, expressed or not, so a trait one branch stops expressing can resurface in a
    descendant (CLAUDE.md §2.3).

    `diverged` must be non-empty and entirely one species: splitting a mixed selection would give
    the daughter a single parent it did not entirely come from, silently corrupting the lineage
    record, so it raises (CLAUDE.md §8.7) rather than picking a parent.
    """
    if len(diverged) == 0:
        raise ValueError("cannot split an empty selection into a new species")

    species_ids = genetics.species_ids(diverged)
    parent_species_id = int(species_ids[0])
    if not (species_ids == parent_species_id).all():
        raise MixedSpeciesError(
            f"a split must come from one species; selection spans "
            f"{sorted(set(species_ids.tolist()))}"
        )

    new_species_id = genetics.species.derive(parent_species_id)
    genetics.speciate(diverged, new_species_id)
    lineage.record_split(new_species_id, parent_species_id)
    return new_species_id


def interbreeding_probability(
    genetics: Genetics,
    a: np.ndarray,
    b: np.ndarray,
    ordinary: float,
    threshold: float,
    columns: np.ndarray,
    species_step: np.ndarray,
) -> np.ndarray:
    """(len(a),) float32 in [0, 1]: how readily each paired (a[i], b[i]) couple can produce young.

    `a` and `b` are **row index arrays**, paired elementwise, exactly as `distance.between` pairs
    them and for the same reason: a `Selection` is a mask and cannot carry a pairing that crosses
    in row space (#20). Both consulted signals appear here:

    - **species id** -- a pair from two different species scores 0. Once a split has been recorded
      the isolation is a fact of the world, not something re-derived from where two individuals
      happen to sit each tick.
    - **distance** -- within one species, a couple up to `ordinary` mutation steps apart is fully
      fertile, and compatibility falls linearly to 0 at `threshold`, the step being the species'
      mean expressed mutability (`species_mutation_step`), never the couple's own (#304). This is the gradual degradation issue #16 asks for: a
      sub-population approaching a split breeds with the parent stock ever more rarely, so the
      hard species gate that follows has almost nothing left to remove. Ordinary couples pay
      nothing for it, because a falloff from zero distance would tax every birth in the world to
      price the few that cross a boundary (see the module docstring for the measurement).

    Linear between the two because nothing measured yet justifies a curve (CLAUDE.md §8.5).

    A species with no mutability at all has no step to count in. Two identical genomes are then
    compatible and any difference is infinitely many steps, which is the limit the formula
    approaches rather than a special case invented for it.

    ordinary, threshold: mutation steps, with `0 <= ordinary < threshold` (`SpeciationConfig`
        validates the pair).
    species_step: (n_species,) float64, indexed by species id: `species_mutation_step` over the
        population the couples were drawn from.
    """
    if not 0.0 <= ordinary < threshold:
        raise ValueError(
            f"need 0 <= ordinary < threshold, got ordinary {ordinary} and threshold {threshold}"
        )

    distance = between(genetics, a, b, columns).astype(np.float64)
    species_a = genetics.species_ids_at(a)
    same_species = species_a == genetics.species_ids_at(b)
    # A cross-species couple is refused below whatever it reads, so a's species is as good a ruler
    # as any for it.
    step = species_step[species_a]
    steps = np.divide(
        distance, step, out=np.where(distance > 0.0, np.inf, 0.0), where=step > 0.0
    )
    compatibility = np.clip((threshold - steps) / (threshold - ordinary), 0.0, 1.0)
    return np.where(same_species, compatibility, 0.0).astype(np.float32)


@dataclass(frozen=True)
class SpeciationConfig:
    """Per-world speciation rules, never constants in `core/` (§2.1).

    distance_genes: the genes genetic distance reads. The scent-signature block in every world so
        far, by the owner's decision on #16 (see the module docstring for what that forecloses).
    ordinary: dimensionless, in mutation steps. The distance up to which a couple is fully
        fertile: an ordinary pair within one interbreeding population. At mutation–drift balance a
        gene's spread is about 1.8 mutation steps (`speciation-drift.md`), so two animals sit about
        `1.8 × √(2 × len(distance_genes))` steps apart, 7.2 for the eight signature genes.
    threshold: dimensionless, in mutation steps. Where interbreeding reaches zero. A split fires at
        the centroid gap that puts a typical cross pair here (`split_gap`).
    min_founders: the fewest animals either side of a proposed split may hold. A species of one
        cannot breed and exists only to go extinct (#16's reopening). This stands in for something
        that should be derived: once `selfing_rate` exists (#99) a selfing lineage can be founded
        by one, and this should read it rather than carry its own number.
    every_n_ticks: how often the check runs (§2.1, step 12), about once a generation in the demo
        world. Isolation takes hundreds of generations to reach a split
        (`docs/spikes/speciation-signature.md`), so a split landing a few ticks late cannot matter,
        and the clustering is paid for once a generation rather than every tick.
    """

    distance_genes: tuple[str, ...]
    ordinary: float
    threshold: float
    min_founders: int
    every_n_ticks: int

    def __post_init__(self) -> None:
        if not self.distance_genes:
            raise ValueError("distance must read at least one gene")
        if not 0.0 <= self.ordinary < self.threshold:
            raise ValueError(
                f"need 0 <= ordinary < threshold, got ordinary {self.ordinary} and threshold "
                f"{self.threshold}; fertility must have somewhere to fall from and to"
            )
        if self.min_founders < 2:
            raise ValueError(
                f"min_founders must be at least 2, got {self.min_founders}; a species of one "
                "cannot breed (#16)"
            )
        if self.every_n_ticks < 1:
            raise ValueError(f"every_n_ticks must be at least 1, got {self.every_n_ticks}")


def propose(
    genetics: Genetics, members: Selection, columns: np.ndarray
) -> tuple[Selection, Selection]:
    """Split `members` into the two groups their signatures fall into: a 2-means clustering.

    Seeded at the member farthest from the centroid and the member farthest from that, so the two
    seeds start on opposite flanks of whatever axis the population is most spread along, and a run
    needs no random draw. It nominates and never decides: on a single blob it still returns two
    halves, and `has_diverged` is what refuses them.

    Twenty-five rounds is a cap, not a target: on two real groups the assignment stops changing
    within a few, and one still moving after twenty-five is a blob with no gap worth finding.
    """
    rows = members.to_indices()
    points = genetics.heritable_columns_at(rows, columns).astype(np.float64)
    centre = points.mean(axis=0)
    first = points[np.argmax(((points - centre) ** 2).sum(axis=1))]
    second = points[np.argmax(((points - first) ** 2).sum(axis=1))]
    nearer_second = np.zeros(len(rows), dtype=bool)
    for _ in range(25):
        updated = ((points - second) ** 2).sum(axis=1) < ((points - first) ** 2).sum(axis=1)
        settled = np.array_equal(updated, nearer_second)
        nearer_second = updated
        if settled or not updated.any() or updated.all():
            break
        first, second = points[~nearer_second].mean(axis=0), points[nearer_second].mean(axis=0)
    capacity = members.capacity
    return (
        Selection.from_indices(rows[~nearer_second], capacity),
        Selection.from_indices(rows[nearer_second], capacity),
    )


class Speciation:
    """The world's speciation step (§2.1, step 12), and the compatibility `Conception` reads.

    Owns the `Lineage`, since a split is the only thing that writes one. Owns no store column:
    species ids are `Genetics`'.
    """

    def __init__(
        self, genetics: Genetics, genes: GeneRegistry, config: SpeciationConfig, lineage: Lineage
    ) -> None:
        self.genetics = genetics
        self.config = config
        self.lineage = lineage
        # Signature genes are positions in cue space, so dimensionless (§2.6).
        self.columns = np.array(
            [genes.index_of(name, unit=Unit.DIMENSIONLESS) for name in config.distance_genes],
            dtype=np.int64,
        )

    def compatibility(self, a: np.ndarray, b: np.ndarray, population: Selection) -> np.ndarray:
        """`interbreeding_probability` at this world's threshold, over its distance genes, with
        each species' step read over `population`, the breeders the couples were drawn from."""
        return interbreeding_probability(
            self.genetics,
            a,
            b,
            self.config.ordinary,
            self.config.threshold,
            self.columns,
            species_mutation_step(self.genetics, population),
        )

    def speciate(self, population: Selection, tick: int) -> None:
        """`check`, as the tick system: a system returns nothing (§2.1's uniform signature), and
        what a split made is read from the species registry and the lineage, which hold it."""
        self.check(population, tick)

    def check(self, population: Selection, tick: int) -> tuple[int, ...]:
        """Split every species in `population` whose two signature clusters have diverged, on the
        ticks the period falls on. Returns the new species ids, in the order they were made.

        `population` is everything allocated, born and gestating: an unborn young belongs to the
        gene pool its parents came from, and leaving it out would have it born into the parent
        species carrying the daughter's genes.

        One species at a time, which §2.3 warns against for a per-tick system. This runs once a
        generation and a clustering is inherently per species, so the loop is over the handful of
        species a world holds, not over animals. Species are read before any split, so a daughter
        made this tick is not itself proposed until the next check.
        """
        if tick % self.config.every_n_ticks:
            return ()
        rows = population.to_indices()
        species_ids = self.genetics.species_ids_at(rows)
        capacity = population.capacity
        made = []
        for species_id in np.unique(species_ids).tolist():
            members = Selection.from_indices(rows[species_ids == species_id], capacity)
            if len(members) < 2 * self.config.min_founders:
                continue
            first, second = propose(self.genetics, members, self.columns)
            if min(len(first), len(second)) < self.config.min_founders:
                continue
            if not has_diverged(
                self.genetics,
                first,
                second,
                self.config.ordinary,
                self.config.threshold,
                self.columns,
            ):
                continue
            # The smaller group is the daughter: a split is a group leaving its parent stock, and
            # the stock is the larger side. Which keeps the old id is otherwise arbitrary.
            daughter = first if len(first) < len(second) else second
            made.append(split(self.genetics, self.lineage, daughter))
        return tuple(made)
