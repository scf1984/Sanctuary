import time

import numpy as np
import pytest

from core.entities.store import EntityStore
from core.genetics.inheritance import inherit_genes
from core.genetics.expression import GeneticsConfig
from core.genetics.service import Genetics
from core.genetics.speciation import (
    Lineage,
    MixedSpeciesError,
    has_diverged,
    interbreeding_probability,
    split,
)
from core.genetics.species import SpeciesRegistry
from core.selection import Selection
from core.services import ColumnRegistry

from tests.support.genes import gene_registry

GENE_NAMES = ("size", "speed", "sight", "camouflage", "clutch_size", "gestation", "mutability")

# Every gene declares how its stored value is read (#104). These are all quantities, so all fold
# across zero; `mutability` is in the vocabulary because inheritance's spread floor is a gene, and
# every world needs one even when — as here — nothing in these tests breeds.
GENETICS_CONFIG = GeneticsConfig(
    mutability_gene="mutability",
    drift_margin=2.0,
)
GENE_REGISTRY = gene_registry(GENE_NAMES)


def make_world(n_entities, expressed=GENE_NAMES, capacity=None):
    """A store of `n_entities` creatures, all one species expressing `expressed`.

    Returns (store, genetics, rows) where `rows` maps population index -> store row, so tests can
    build selections over specific creatures without reaching into the store's id mapping twice.
    """
    capacity = capacity if capacity is not None else n_entities
    store = EntityStore(initial_capacity=capacity, n_drives=1, n_genes=len(GENE_NAMES))
    vocabulary = GENE_REGISTRY
    species = SpeciesRegistry(vocabulary)
    genetics = Genetics(store, ColumnRegistry(), species, vocabulary, GENETICS_CONFIG)
    species_id = species.register(expressed)

    ids = store.allocate(
        n_entities, species_id=np.full(n_entities, species_id, dtype=np.int32)
    )
    rows = np.array([store._id_to_row[i] for i in ids.tolist()], dtype=np.int64)
    return store, genetics, rows


def selection_of(store, rows, indices):
    mask = np.zeros(store.capacity, dtype=np.bool_)
    mask[rows[np.asarray(indices, dtype=np.int64)]] = True
    return Selection.from_mask(mask)


class TestLineage:
    def test_a_root_species_has_no_parent_and_is_its_own_ancestry(self):
        lineage = Lineage()
        assert lineage.parent_of(0) is None
        assert lineage.ancestry(0) == (0,)

    def test_ancestry_walks_from_the_root_down(self):
        lineage = Lineage()
        lineage.record_split(1, 0)
        lineage.record_split(2, 1)
        lineage.record_split(3, 1)

        assert lineage.parent_of(2) == 1
        assert lineage.ancestry(2) == (0, 1, 2)
        # A sibling branch shares the ancestry prefix but not the other branch's tip.
        assert lineage.ancestry(3) == (0, 1, 3)

    def test_recording_a_species_twice_raises(self):
        lineage = Lineage()
        lineage.record_split(1, 0)
        with pytest.raises(ValueError):
            lineage.record_split(1, 0)


class TestSplit:
    def test_diverged_creatures_get_a_new_species_and_the_rest_keep_the_old_one(self):
        store, genetics, rows = make_world(6)
        everyone = selection_of(store, rows, range(6))
        parent_species_id = int(genetics.species_ids(everyone)[0])

        diverged = selection_of(store, rows, [0, 1, 2])
        stayed = selection_of(store, rows, [3, 4, 5])

        new_species_id = split(genetics, Lineage(), diverged)

        assert new_species_id != parent_species_id
        assert (genetics.species_ids(diverged) == new_species_id).all()
        assert (genetics.species_ids(stayed) == parent_species_id).all()

    def test_the_daughter_species_inherits_the_parents_expression_mask(self):
        store, genetics, rows = make_world(4, expressed=("size", "sight"))
        diverged = selection_of(store, rows, [0, 1])
        parent_species_id = int(genetics.species_ids(diverged)[0])

        new_species_id = split(genetics, Lineage(), diverged)

        np.testing.assert_array_equal(
            genetics.species.mask_of(new_species_id),
            genetics.species.mask_of(parent_species_id),
        )

    def test_masks_are_independent_after_a_split(self):
        # The copy is the point: two branches must be able to diverge in expression later without
        # one rewriting the other's phenotype.
        store, genetics, rows = make_world(4, expressed=("size", "sight"))
        diverged = selection_of(store, rows, [0, 1])
        parent_species_id = int(genetics.species_ids(diverged)[0])
        new_species_id = split(genetics, Lineage(), diverged)

        genetics.species.mask_of(new_species_id)[:] = False

        assert genetics.species.mask_of(parent_species_id).any()

    def test_split_records_the_lineage(self):
        store, genetics, rows = make_world(4)
        lineage = Lineage()
        diverged = selection_of(store, rows, [0, 1])
        parent_species_id = int(genetics.species_ids(diverged)[0])

        new_species_id = split(genetics, lineage, diverged)

        assert lineage.parent_of(new_species_id) == parent_species_id
        assert lineage.ancestry(new_species_id) == (parent_species_id, new_species_id)

    def test_repeated_splits_build_a_chain(self):
        store, genetics, rows = make_world(6)
        lineage = Lineage()
        first = split(genetics, lineage, selection_of(store, rows, [0, 1, 2, 3]))
        second = split(genetics, lineage, selection_of(store, rows, [0, 1]))

        assert lineage.ancestry(second) == (0, first, second)

    def test_splitting_an_empty_selection_raises(self):
        store, genetics, rows = make_world(4)
        with pytest.raises(ValueError):
            split(genetics, Lineage(), Selection.none(store.capacity))

    def test_splitting_across_two_species_raises(self):
        store, genetics, rows = make_world(6)
        split(genetics, Lineage(), selection_of(store, rows, [0, 1]))

        # Rows 0-1 are now a different species from rows 2-5; a split spanning both has no single
        # parent to descend from.
        with pytest.raises(MixedSpeciesError):
            split(genetics, Lineage(), selection_of(store, rows, [1, 2]))

    def test_genes_are_untouched_by_a_split_including_unexpressed_ones(self):
        # Speciation is a species-id write and nothing else (CLAUDE.md §2.3): a gene the new
        # species does not express is still carried, so it can resurface generations later.
        store, genetics, rows = make_world(4, expressed=("size", "speed"))
        rng = np.random.default_rng(0)
        genes_before = rng.uniform(0.5, 2.0, size=(4, len(GENE_NAMES))).astype(np.float32)
        store.genes[rows] = genes_before

        split(genetics, Lineage(), selection_of(store, rows, [0, 1]))

        np.testing.assert_array_equal(store.genes[rows], genes_before)


class TestSplitCostsNoReallocation:
    """CLAUDE.md §2.3: speciation is a species-id write plus a mask row. Nothing entities-shaped
    is reallocated, copied, or restructured -- asserted by array identity, not by inspection."""

    def test_no_entity_column_is_replaced_and_capacity_is_unchanged(self):
        store, genetics, rows = make_world(1000, capacity=1000)
        columns_before = {
            name: getattr(store, name)
            for name in ("x", "y", "z", "energy", "age", "health", "species_id", "genes", "alive")
        }
        capacity_before = store.capacity

        split(genetics, Lineage(), selection_of(store, rows, range(500)))

        assert store.capacity == capacity_before
        for name, array in columns_before.items():
            assert getattr(store, name) is array, f"column '{name}' was replaced by the split"

    def test_split_cost_does_not_scale_with_population(self):
        """A split writes only the diverged rows, so splitting a fixed-size group out of a 100x
        larger world must not cost 100x more. Asserts a ratio, not a wall-clock budget, so the
        measurement stays meaningful on whatever hardware runs it."""

        def time_split(population):
            store, genetics, rows = make_world(population)
            diverged = selection_of(store, rows, range(100))
            start = time.perf_counter()
            split(genetics, Lineage(), diverged)
            return time.perf_counter() - start

        small = min(time_split(1_000) for _ in range(5))
        large = min(time_split(100_000) for _ in range(5))

        # The write itself is O(100 rows) in both, but `Selection.to_mask()` indexing is a pass
        # over the full capacity-length boolean mask, so the large world is legitimately slower by
        # some margin. 20x leaves room for that while still failing loudly if a future change
        # makes speciation copy or rebuild anything population-sized.
        assert large < small * 20 + 0.005


class TestHasDiverged:
    def test_a_population_against_itself_has_not_diverged(self):
        store, genetics, rows = make_world(6)
        store.genes[rows] = np.random.default_rng(1).uniform(0.5, 2.0, (6, len(GENE_NAMES)))
        everyone = selection_of(store, rows, range(6))

        assert not has_diverged(genetics, everyone, everyone, threshold=0.01)

    def test_divergence_is_decided_at_the_threshold(self):
        store, genetics, rows = make_world(2)
        store.genes[rows] = np.array(
            [[0.0] * len(GENE_NAMES), [3.0, 4.0, *([0.0] * (len(GENE_NAMES) - 2))]], dtype=np.float32
        )
        a = selection_of(store, rows, [0])
        b = selection_of(store, rows, [1])

        assert has_diverged(genetics, a, b, threshold=4.9)  # centroid distance is 5.0
        assert not has_diverged(genetics, a, b, threshold=5.1)

    def test_an_outlier_pair_does_not_speciate_a_population(self):
        # Divergence compares centroids: one wildly different individual inside an otherwise
        # identical population must not read as two species.
        store, genetics, rows = make_world(20)
        store.genes[rows] = 1.0
        store.genes[rows[0]] = 50.0
        a = selection_of(store, rows, range(10))
        b = selection_of(store, rows, range(10, 20))

        assert has_diverged(genetics, a, b, threshold=1.0)  # the outlier does shift a's centroid
        assert not has_diverged(genetics, a, b, threshold=100.0)


class TestInterbreedingProbability:
    def two_creatures(self, gene_values):
        store, genetics, rows = make_world(2)
        store.genes[rows] = np.array(gene_values, dtype=np.float32)
        return store, genetics, rows

    def test_identical_creatures_of_one_species_are_fully_compatible(self):
        store, genetics, rows = self.two_creatures([[1.0] * len(GENE_NAMES), [1.0] * len(GENE_NAMES)])
        a, b = selection_of(store, rows, [0]), selection_of(store, rows, [1])

        assert interbreeding_probability(genetics, a, b, threshold=1.0)[0] == pytest.approx(1.0)

    def test_compatibility_falls_linearly_and_reaches_zero_at_the_threshold(self):
        store, genetics, rows = self.two_creatures([[0.0] * len(GENE_NAMES), [1.0, *([0.0] * (len(GENE_NAMES) - 1))]])
        a, b = selection_of(store, rows, [0]), selection_of(store, rows, [1])

        assert interbreeding_probability(genetics, a, b, threshold=4.0)[0] == pytest.approx(0.75)
        assert interbreeding_probability(genetics, a, b, threshold=2.0)[0] == pytest.approx(0.5)
        assert interbreeding_probability(genetics, a, b, threshold=1.0)[0] == pytest.approx(0.0)

    def test_beyond_the_threshold_compatibility_stays_at_zero(self):
        store, genetics, rows = self.two_creatures([[0.0] * len(GENE_NAMES), [10.0, *([0.0] * (len(GENE_NAMES) - 1))]])
        a, b = selection_of(store, rows, [0]), selection_of(store, rows, [1])

        assert interbreeding_probability(genetics, a, b, threshold=1.0)[0] == 0.0

    def test_the_split_removes_an_already_vanishing_probability(self):
        """The no-cliff-edge property (#16): a pair close enough to the threshold that their
        population is about to be split is already breeding at a negligible rate, so the hard
        species gate that follows the split is not where compatibility falls off."""
        threshold = 1.0
        store, genetics, rows = self.two_creatures([[0.0] * len(GENE_NAMES), [0.98, *([0.0] * (len(GENE_NAMES) - 1))]])
        a, b = selection_of(store, rows, [0]), selection_of(store, rows, [1])

        before_split = interbreeding_probability(genetics, a, b, threshold)[0]
        assert before_split < 0.05

        split(genetics, Lineage(), a)
        after_split = interbreeding_probability(genetics, a, b, threshold)[0]

        assert after_split == 0.0
        assert before_split - after_split < 0.05  # the discontinuity at the split is negligible

    def test_different_species_never_interbreed_however_similar(self):
        store, genetics, rows = self.two_creatures([[1.0] * len(GENE_NAMES), [1.0] * len(GENE_NAMES)])
        a, b = selection_of(store, rows, [0]), selection_of(store, rows, [1])
        split(genetics, Lineage(), a)

        # Phenotypically identical, so distance alone would say fully compatible -- the recorded
        # split is what makes them incompatible.
        assert interbreeding_probability(genetics, a, b, threshold=1.0)[0] == 0.0

    def test_is_evaluated_per_pair_over_a_whole_selection(self):
        store, genetics, rows = make_world(4)
        store.genes[rows] = np.array(
            [
                [0.0, *([0.0] * (len(GENE_NAMES) - 1))],
                [0.0, *([0.0] * (len(GENE_NAMES) - 1))],
                [0.0, *([0.0] * (len(GENE_NAMES) - 1))],  # pairs with row 0: distance 0
                [1.0, *([0.0] * (len(GENE_NAMES) - 1))],  # pairs with row 1: distance 1
            ],
            dtype=np.float32,
        )
        a = selection_of(store, rows, [0, 1])
        b = selection_of(store, rows, [2, 3])

        np.testing.assert_allclose(
            interbreeding_probability(genetics, a, b, threshold=2.0), [1.0, 0.5]
        )

    def test_a_non_positive_threshold_raises(self):
        store, genetics, rows = make_world(2)
        a, b = selection_of(store, rows, [0]), selection_of(store, rows, [1])
        with pytest.raises(ValueError):
            interbreeding_probability(genetics, a, b, threshold=0.0)

    @pytest.mark.parametrize("seed", range(10))
    def test_compatibility_never_increases_with_distance(self, seed):
        rng = np.random.default_rng(seed)
        store, genetics, rows = make_world(2 * 12)
        store.genes[rows] = 0.0
        # Row 12+i sits distance `spread[i]` from row i along one gene, with spread ascending --
        # so compatibility must come back descending.
        spread = np.sort(rng.uniform(0.0, 3.0, size=12))
        store.genes[rows[12:], 0] = spread

        probability = interbreeding_probability(
            genetics,
            selection_of(store, rows, range(12)),
            selection_of(store, rows, range(12, 24)),
            threshold=2.0,
        )
        assert (np.diff(probability) <= 1e-6).all()


# Statistical parameters for TestIsolationCausesSpeciation, measured rather than guessed --
# docs/spikes/speciation-drift.md, a 200-seed sweep, **re-measured for #104's inheritance rule**.
# After 50 generations at these settings two isolated sub-populations' centroids sat at or above 0.12
# in 200/200 runs (minimum 0.145, 5th percentile 0.272, median 0.503), while two arbitrary halves of
# one interbreeding pool crossed it in 1 run of 200 (median 0.025, 95th percentile 0.061). Re-run
# that spike if the gene count, population size, mutability or drift margin here changes.
#
# `MUTABILITY` is what makes the isolated arm keep moving at all. The spike's control arm carries no
# floor -- the rule this replaced -- and its within-pool spread collapses from 0.081 at generation 20
# to 0.0004 at generation 100, so the two halves stop diverging once they have frozen. At 0.02 the
# spread instead settles at ~0.036 and holds, which is a mutation-drift balance rather than either
# collapse or blow-up (#104).
POPULATION = 40
GENERATIONS = 50
MUTABILITY = 0.02
DRIFT_MARGIN = 2.0
SPECIATION_THRESHOLD = 0.12


def breed(genetics, population, rng):
    """Replace `population`'s genes with one generation of offspring bred within it.

    Parents are drawn at random with replacement from the population itself, which is what makes
    it a closed gene pool -- the "isolated by a fence or a mountain range" case (CLAUDE.md §2.5)
    reduced to its genetic content. Pairing is done here rather than through `Genetics.inherit`
    because that method pairs two selections positionally and a Selection is a set of rows, so it
    cannot express "row 3 mates with row 7, and row 3 again with row 12".

    The mutability floor is passed directly rather than read from the population's own mutability
    gene, so that this measurement holds one variable fixed: what is under test is whether isolation
    separates two pools, not whether a lineage's evolvability itself drifts.
    """
    genes = genetics.genes(population)
    n = genes.shape[0]
    parent_a = genes[rng.integers(0, n, size=n)]
    parent_b = genes[rng.integers(0, n, size=n)]
    floor = np.full(n, MUTABILITY, dtype=np.float32)
    genetics.set_genes(
        population, inherit_genes(parent_a, parent_b, floor, DRIFT_MARGIN, rng)
    )


def evolve(seed, isolated):
    """Run two sub-populations for GENERATIONS and report whether they ended up diverged.

    isolated=True breeds each half only within itself; isolated=False breeds the two halves as one
    pool. Everything else -- founders, population size, generation count, gene count -- is
    identical between the two arms, so the only difference is whether genes cross the boundary.
    """
    rng = np.random.default_rng(seed)
    store, genetics, rows = make_world(2 * POPULATION)
    store.genes[rows] = rng.uniform(0.5, 1.5, size=(2 * POPULATION, len(GENE_NAMES)))

    left = selection_of(store, rows, range(POPULATION))
    right = selection_of(store, rows, range(POPULATION, 2 * POPULATION))
    everyone = left | right

    for _ in range(GENERATIONS):
        if isolated:
            breed(genetics, left, rng)
            breed(genetics, right, rng)
        else:
            breed(genetics, everyone, rng)

    return has_diverged(genetics, left, right, SPECIATION_THRESHOLD), genetics, left, right


class TestIsolationCausesSpeciation:
    """The payoff test (#16's "done when"): isolation reliably produces a new species and mixing
    reliably does not. Asserted as a rate over many seeds, never as a single run's outcome --
    the simulation is deliberately non-deterministic (CLAUDE.md §2.2, §6)."""

    def test_isolated_populations_reliably_diverge(self):
        diverged = sum(evolve(seed, isolated=True)[0] for seed in range(40))
        assert diverged >= 36  # measured 200/200; 36/40 leaves room for seed-set variation

    def test_an_interbreeding_population_reliably_does_not_diverge(self):
        """A *rate*, not zero. #104's inheritance keeps a mixed pool's variance alive instead of
        letting it converge, so an arbitrary partition of one pool occasionally wanders across the
        threshold — measured at 1 run in 200, and 1 of these 40 seeds is that run. Asserting zero
        against a quantity measured at 0.5% is the knife edge §6 warns about, so the bound is a rate
        like its sibling above. A rare spurious split of a well-mixed population is real behaviour
        rather than a bug: divergence is a distance between centroids, and nothing forbids one.
        """
        diverged = sum(evolve(seed, isolated=False)[0] for seed in range(40))
        assert diverged <= 2

    def test_divergence_then_split_leaves_two_species_that_cannot_interbreed(self):
        """End to end: drift apart in isolation, split, and the two populations stop being able
        to produce young together."""
        did_diverge, genetics, left, right = evolve(seed=0, isolated=True)
        assert did_diverge

        lineage = Lineage()
        parent_species_id = int(genetics.species_ids(left)[0])
        new_species_id = split(genetics, lineage, left)

        assert lineage.ancestry(new_species_id) == (parent_species_id, new_species_id)
        assert (genetics.species_ids(right) == parent_species_id).all()
        cross_pairs = interbreeding_probability(genetics, left, right, SPECIATION_THRESHOLD)
        assert (cross_pairs == 0.0).all()
