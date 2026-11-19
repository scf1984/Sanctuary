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
    Speciation,
    SpeciationConfig,
    has_diverged,
    interbreeding_probability,
    propose,
    split,
    split_gap,
)
from core.genetics.species import SpeciesRegistry
from core.selection import Selection
from core.services import ColumnRegistry

from tests.support.genes import gene_registry
from tests.support.timing import wall_clock

# Every distance gene is dimensionless, as the signature block is: `Speciation` checks the unit, and
# `speed` (a length, in the shared test vocabulary) would be refused, as it should be.
GENE_NAMES = ("size", "stamina", "sight", "camouflage", "clutch_size", "gestation", "mutability")

# Every gene declares how its stored value is read (#104). These are all quantities, so all fold
# across zero; `mutability` is in the vocabulary because inheritance's spread floor is a gene, and
# every world needs one even when — as here — nothing in these tests breeds.
GENETICS_CONFIG = GeneticsConfig(
    mutability_gene="mutability",
    drift_margin=2.0,
    senescence=None,
)
GENE_REGISTRY = gene_registry(GENE_NAMES)

# Distance reads every gene but `mutability`, which is the unit it is counted in (#16). The
# helpers below hold that unit at 1 unless a test is about it, so a threshold here reads in the
# same units as the genes the test writes.
DISTANCE_GENES = GENE_NAMES[:-1]
COLUMNS = np.arange(len(DISTANCE_GENES), dtype=np.int64)
MUTABILITY_COLUMN = GENE_NAMES.index("mutability")


def unit_steps(genetics):
    """Set every allocated animal's mutability gene to 1: one mutation step is one gene unit."""
    store = genetics.store
    store.genes[store.alive, MUTABILITY_COLUMN] = 1.0


def diverged(genetics, a, b, threshold, ordinary=0.0):
    """`has_diverged` at unit steps. With no ordinary distance the split gap is the threshold
    itself, which is what the tests written before the ordinary distance existed assert."""
    unit_steps(genetics)
    return has_diverged(genetics, a, b, ordinary, threshold, COLUMNS)


def compatible(genetics, a, b, threshold, ordinary=0.0):
    """`interbreeding_probability` at unit steps; linear from zero distance unless `ordinary`."""
    unit_steps(genetics)
    return interbreeding_probability(genetics, a, b, ordinary, threshold, COLUMNS)


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
        store, genetics, rows = make_world(4, expressed=("size", "stamina"))
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

    @wall_clock
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

        assert not diverged(genetics, everyone, everyone, 0.01)

    def test_divergence_is_decided_at_the_threshold(self):
        store, genetics, rows = make_world(2)
        store.genes[rows] = np.array(
            [[0.0] * len(GENE_NAMES), [3.0, 4.0, *([0.0] * (len(GENE_NAMES) - 2))]], dtype=np.float32
        )
        a = selection_of(store, rows, [0])
        b = selection_of(store, rows, [1])

        assert diverged(genetics, a, b, 4.9)  # centroid distance is 5.0
        assert not diverged(genetics, a, b, 5.1)

    def test_an_outlier_pair_does_not_speciate_a_population(self):
        # Divergence compares centroids: one wildly different individual inside an otherwise
        # identical population must not read as two species.
        store, genetics, rows = make_world(20)
        store.genes[rows] = 1.0
        store.genes[rows[0]] = 50.0
        a = selection_of(store, rows, range(10))
        b = selection_of(store, rows, range(10, 20))

        assert diverged(genetics, a, b, 1.0)  # the outlier does shift a's centroid
        assert not diverged(genetics, a, b, 100.0)


class TestInterbreedingProbability:
    def two_creatures(self, gene_values):
        store, genetics, rows = make_world(2)
        store.genes[rows] = np.array(gene_values, dtype=np.float32)
        return store, genetics, rows

    def test_identical_creatures_of_one_species_are_fully_compatible(self):
        store, genetics, rows = self.two_creatures([[1.0] * len(GENE_NAMES), [1.0] * len(GENE_NAMES)])
        a, b = selection_of(store, rows, [0]), selection_of(store, rows, [1])

        assert compatible(genetics, a.to_indices(), b.to_indices(), 1.0)[0] == pytest.approx(1.0)

    def test_compatibility_falls_linearly_and_reaches_zero_at_the_threshold(self):
        store, genetics, rows = self.two_creatures([[0.0] * len(GENE_NAMES), [1.0, *([0.0] * (len(GENE_NAMES) - 1))]])
        a, b = selection_of(store, rows, [0]), selection_of(store, rows, [1])

        assert compatible(genetics, a.to_indices(), b.to_indices(), 4.0)[0] == pytest.approx(0.75)
        assert compatible(genetics, a.to_indices(), b.to_indices(), 2.0)[0] == pytest.approx(0.5)
        assert compatible(genetics, a.to_indices(), b.to_indices(), 1.0)[0] == pytest.approx(0.0)

    def test_beyond_the_threshold_compatibility_stays_at_zero(self):
        store, genetics, rows = self.two_creatures([[0.0] * len(GENE_NAMES), [10.0, *([0.0] * (len(GENE_NAMES) - 1))]])
        a, b = selection_of(store, rows, [0]), selection_of(store, rows, [1])

        assert compatible(genetics, a.to_indices(), b.to_indices(), 1.0)[0] == 0.0

    def test_the_split_removes_an_already_vanishing_probability(self):
        """The no-cliff-edge property (#16): a pair close enough to the threshold that their
        population is about to be split is already breeding at a negligible rate, so the hard
        species gate that follows the split is not where compatibility falls off."""
        threshold = 1.0
        store, genetics, rows = self.two_creatures([[0.0] * len(GENE_NAMES), [0.98, *([0.0] * (len(GENE_NAMES) - 1))]])
        a, b = selection_of(store, rows, [0]), selection_of(store, rows, [1])

        before_split = compatible(genetics, a.to_indices(), b.to_indices(), threshold)[0]
        assert before_split < 0.05

        split(genetics, Lineage(), a)
        after_split = compatible(genetics, a.to_indices(), b.to_indices(), threshold)[0]

        assert after_split == 0.0
        assert before_split - after_split < 0.05  # the discontinuity at the split is negligible

    def test_different_species_never_interbreed_however_similar(self):
        store, genetics, rows = self.two_creatures([[1.0] * len(GENE_NAMES), [1.0] * len(GENE_NAMES)])
        a, b = selection_of(store, rows, [0]), selection_of(store, rows, [1])
        split(genetics, Lineage(), a)

        # Phenotypically identical, so distance alone would say fully compatible -- the recorded
        # split is what makes them incompatible.
        assert compatible(genetics, a.to_indices(), b.to_indices(), 1.0)[0] == 0.0

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
            compatible(genetics, a.to_indices(), b.to_indices(), 2.0), [1.0, 0.5]
        )

    def test_a_non_positive_threshold_raises(self):
        store, genetics, rows = make_world(2)
        a, b = selection_of(store, rows, [0]), selection_of(store, rows, [1])
        with pytest.raises(ValueError):
            compatible(genetics, a.to_indices(), b.to_indices(), 0.0)

    @pytest.mark.parametrize("seed", range(10))
    def test_compatibility_never_increases_with_distance(self, seed):
        rng = np.random.default_rng(seed)
        store, genetics, rows = make_world(2 * 12)
        store.genes[rows] = 0.0
        # Row 12+i sits distance `spread[i]` from row i along one gene, with spread ascending --
        # so compatibility must come back descending.
        spread = np.sort(rng.uniform(0.0, 3.0, size=12))
        store.genes[rows[12:], 0] = spread

        probability = compatible(
            genetics,
            selection_of(store, rows, range(12)).to_indices(),
            selection_of(store, rows, range(12, 24)).to_indices(),
            2.0,
        )
        assert (np.diff(probability) <= 1e-6).all()


class TestTheThresholdIsCountedInMutationSteps:
    """#16, the owner's decision: distance is read against the animals' own mutability, so one
    threshold means the same in a world whose lineages evolved high mutability as in one whose
    lineages evolved low."""

    def two_at_distance(self, distance, mutability):
        store, genetics, rows = make_world(2)
        store.genes[rows] = 0.0
        store.genes[rows[1], 0] = distance
        store.genes[rows, MUTABILITY_COLUMN] = mutability
        return genetics, rows

    def test_the_same_distance_is_further_for_less_mutable_animals(self):
        steady, rows = self.two_at_distance(1.0, 0.5)
        volatile, _ = self.two_at_distance(1.0, 2.0)

        assert interbreeding_probability(steady, rows[:1], rows[1:], 0.0, 4.0, COLUMNS)[0] == (
            pytest.approx(0.5)
        )
        assert interbreeding_probability(volatile, rows[:1], rows[1:], 0.0, 4.0, COLUMNS)[0] == (
            pytest.approx(0.875)
        )

    def test_scaling_distance_and_mutability_together_changes_nothing(self):
        """The property that makes the threshold dimensionless."""
        small, rows = self.two_at_distance(0.3, 0.1)
        large, _ = self.two_at_distance(3.0, 1.0)

        assert interbreeding_probability(small, rows[:1], rows[1:], 0.0, 5.0, COLUMNS)[0] == (
            pytest.approx(interbreeding_probability(large, rows[:1], rows[1:], 0.0, 5.0, COLUMNS)[0])
        )

    def test_a_couple_mean_is_the_step(self):
        store, genetics, rows = make_world(2)
        store.genes[rows] = 0.0
        store.genes[rows[1], 0] = 1.0
        store.genes[rows[0], MUTABILITY_COLUMN] = 0.5
        store.genes[rows[1], MUTABILITY_COLUMN] = 1.5

        # Mean mutability 1, so distance 1 is one step of four.
        assert interbreeding_probability(genetics, rows[:1], rows[1:], 0.0, 4.0, COLUMNS)[0] == (
            pytest.approx(0.75)
        )

    def test_without_mutability_only_identical_genomes_are_compatible(self):
        same, rows = self.two_at_distance(0.0, 0.0)
        apart, _ = self.two_at_distance(0.1, 0.0)

        assert interbreeding_probability(same, rows[:1], rows[1:], 0.0, 4.0, COLUMNS)[0] == 1.0
        assert interbreeding_probability(apart, rows[:1], rows[1:], 0.0, 4.0, COLUMNS)[0] == 0.0

    def test_divergence_is_counted_in_the_same_steps(self):
        store, genetics, rows = make_world(2)
        store.genes[rows] = 0.0
        store.genes[rows[1], 0] = 1.0
        store.genes[rows, MUTABILITY_COLUMN] = 0.25
        a, b = selection_of(store, rows, [0]), selection_of(store, rows, [1])

        assert has_diverged(genetics, a, b, 0.0, 3.9, COLUMNS)  # 1.0 is four steps of 0.25
        assert not has_diverged(genetics, a, b, 0.0, 4.1, COLUMNS)

    def test_genes_outside_the_distance_columns_do_not_count(self):
        store, genetics, rows = make_world(2)
        store.genes[rows] = 0.0
        store.genes[rows, MUTABILITY_COLUMN] = 1.0
        store.genes[rows[1], 0] = 50.0  # a gene distance does not read
        only_second = np.array([1], dtype=np.int64)

        assert interbreeding_probability(genetics, rows[:1], rows[1:], 0.0, 2.0, only_second)[0] == 1.0


class TestOrdinaryCouplesAreFullyFertile:
    """#16, the owner's decision: fertility does not fall from zero distance. A couple up to an
    ordinary within-species distance pays nothing, and fertility falls linearly to zero at the
    threshold. Measured on the demo world, a falloff from zero taxed every birth to price the few
    that cross a boundary, and needed a threshold drift never reached."""

    def at(self, distance, ordinary, threshold):
        store, genetics, rows = make_world(2)
        store.genes[rows] = 0.0
        store.genes[rows[1], 0] = distance
        return compatible(genetics, rows[:1], rows[1:], threshold, ordinary)[0]

    def test_a_couple_inside_the_ordinary_distance_is_fully_fertile(self):
        assert self.at(0.0, ordinary=2.0, threshold=4.0) == 1.0
        assert self.at(1.9, ordinary=2.0, threshold=4.0) == 1.0

    def test_fertility_falls_linearly_between_ordinary_and_threshold(self):
        assert self.at(3.0, ordinary=2.0, threshold=4.0) == pytest.approx(0.5)
        assert self.at(3.5, ordinary=2.0, threshold=4.0) == pytest.approx(0.25)

    def test_a_couple_at_the_threshold_is_infertile(self):
        assert self.at(4.0, ordinary=2.0, threshold=4.0) == 0.0
        assert self.at(9.0, ordinary=2.0, threshold=4.0) == 0.0

    def test_the_ordinary_distance_must_sit_below_the_threshold(self):
        store, genetics, rows = make_world(2)
        with pytest.raises(ValueError, match="ordinary < threshold"):
            interbreeding_probability(genetics, rows[:1], rows[1:], 4.0, 4.0, COLUMNS)
        with pytest.raises(ValueError, match="ordinary < threshold"):
            SpeciationConfig(
                DISTANCE_GENES, ordinary=5.0, threshold=4.0, min_founders=2, every_n_ticks=1
            )

    def test_a_split_fires_where_a_typical_cross_pair_reaches_the_threshold(self):
        """The gap is the centroid distance at which a pair drawn across the two groups (that gap
        plus an ordinary scatter, in quadrature) sits exactly at the threshold: √(4² − 2²)."""
        assert split_gap(2.0, 4.0) == pytest.approx(np.sqrt(12.0))

        store, genetics, rows = make_world(2)
        store.genes[rows] = 0.0
        a, b = selection_of(store, rows, [0]), selection_of(store, rows, [1])
        store.genes[rows[1], 0] = np.sqrt(12.0) + 0.01
        assert diverged(genetics, a, b, 4.0, ordinary=2.0)
        store.genes[rows[1], 0] = np.sqrt(12.0) - 0.01
        assert not diverged(genetics, a, b, 4.0, ordinary=2.0)


def blobs(n_each, centres, spread, rng):
    """A one-species world of `len(centres)` groups of `n_each`, around `centres` along gene 0."""
    store, genetics, rows = make_world(n_each * len(centres))
    store.genes[rows] = rng.normal(0.0, spread, (len(rows), len(GENE_NAMES)))
    for group, centre in enumerate(centres):
        store.genes[rows[group * n_each:(group + 1) * n_each], 0] += centre
    store.genes[rows, MUTABILITY_COLUMN] = 1.0
    return store, genetics, rows


class TestPropose:
    def test_two_separated_groups_are_found(self):
        store, genetics, rows = blobs(30, (0.0, 10.0), 0.3, np.random.default_rng(0))
        everyone = selection_of(store, rows, range(60))

        first, second = propose(genetics, everyone, COLUMNS)

        groups = {frozenset(first.to_indices().tolist()), frozenset(second.to_indices().tolist())}
        assert groups == {frozenset(rows[:30].tolist()), frozenset(rows[30:].tolist())}

    def test_the_two_halves_cover_every_member_once(self):
        store, genetics, rows = blobs(25, (0.0,), 1.0, np.random.default_rng(1))
        everyone = selection_of(store, rows, range(25))

        first, second = propose(genetics, everyone, COLUMNS)

        assert len(first) + len(second) == 25
        assert len(first & second) == 0

    def test_identical_animals_cannot_be_halved(self):
        store, genetics, rows = make_world(6)
        store.genes[rows] = 1.0

        first, second = propose(genetics, selection_of(store, rows, range(6)), COLUMNS)

        assert min(len(first), len(second)) == 0


def speciation_for(genetics, threshold, min_founders=10, every_n_ticks=1, ordinary=0.0):
    config = SpeciationConfig(
        distance_genes=DISTANCE_GENES,
        ordinary=ordinary,
        threshold=threshold,
        min_founders=min_founders,
        every_n_ticks=every_n_ticks,
    )
    return Speciation(genetics, GENE_REGISTRY, config, Lineage())


class TestCheck:
    def test_two_diverged_groups_split_into_two_species(self):
        store, genetics, rows = blobs(30, (0.0, 10.0), 0.3, np.random.default_rng(0))
        speciation = speciation_for(genetics, threshold=5.0)

        made = speciation.check(selection_of(store, rows, range(60)), tick=0)

        assert len(made) == 1
        species = genetics.species_ids_at(rows)
        assert len(set(species[:30].tolist())) == 1 and len(set(species[30:].tolist())) == 1
        assert species[0] != species[30]
        assert speciation.lineage.parent_of(made[0]) == 0

    def test_the_smaller_group_is_the_daughter(self):
        store, genetics, rows = make_world(50)
        rng = np.random.default_rng(3)
        store.genes[rows] = rng.normal(0.0, 0.3, (50, len(GENE_NAMES)))
        store.genes[rows[:15], 0] += 10.0
        store.genes[rows, MUTABILITY_COLUMN] = 1.0

        (daughter,) = speciation_for(genetics, threshold=5.0).check(
            selection_of(store, rows, range(50)), tick=0
        )

        assert (genetics.species_ids_at(rows[:15]) == daughter).all()
        assert (genetics.species_ids_at(rows[15:]) == 0).all()

    def test_groups_closer_than_the_threshold_stay_one_species(self):
        store, genetics, rows = blobs(30, (0.0, 3.0), 0.3, np.random.default_rng(0))

        made = speciation_for(genetics, threshold=5.0).check(
            selection_of(store, rows, range(60)), tick=0
        )

        assert made == ()
        assert (genetics.species_ids_at(rows) == 0).all()

    def test_a_group_smaller_than_the_minimum_does_not_become_a_species(self):
        """#16's reopening: a species of one, or of a family, cannot breed and only goes extinct."""
        store, genetics, rows = make_world(40)
        rng = np.random.default_rng(4)
        store.genes[rows] = rng.normal(0.0, 0.3, (40, len(GENE_NAMES)))
        store.genes[rows[:5], 0] += 10.0
        store.genes[rows, MUTABILITY_COLUMN] = 1.0

        made = speciation_for(genetics, threshold=5.0, min_founders=10).check(
            selection_of(store, rows, range(40)), tick=0
        )

        assert made == ()

    def test_it_runs_only_on_the_ticks_its_period_falls_on(self):
        store, genetics, rows = blobs(30, (0.0, 10.0), 0.3, np.random.default_rng(0))
        speciation = speciation_for(genetics, threshold=5.0, every_n_ticks=100)
        everyone = selection_of(store, rows, range(60))

        assert speciation.check(everyone, tick=150) == ()
        assert len(speciation.check(everyone, tick=200)) == 1

    def test_a_daughter_made_this_check_is_not_split_again_until_the_next(self):
        store, genetics, rows = blobs(30, (0.0, 10.0, 20.0), 0.3, np.random.default_rng(5))
        speciation = speciation_for(genetics, threshold=5.0)
        everyone = selection_of(store, rows, range(90))

        first_round = speciation.check(everyone, tick=0)
        second_round = speciation.check(everyone, tick=1)

        assert len(first_round) == 1
        assert len(second_round) == 1
        assert genetics.species.n_species == 3


# The statistical harness for TestIsolationCausesSpeciation: two pools of POPULATION, bred for
# GENERATIONS either apart (isolated) or as one (mixed). It mirrors the demo world after #16:
#
# - founders sit at one point in cue space, spread about MUTABILITY x 1.8 per gene, which is the
#   mutation-drift balance (docs/spikes/speciation-drift.md), so the pool starts as one population;
# - every animal's mutability gene is MUTABILITY and held there, so a mutation step is a known
#   quantity and the test is about isolation, not about evolvability drifting;
# - distance is counted in those steps, with the ordinary distance derived exactly as the demo world
#   derives it (1.8 x sqrt(2 x genes)) and the threshold at twice it.
#
# Measured over 200 seeds per arm, splits by `Speciation.check`:
#
#   generations   isolated   mixed
#           100     80/200   4/200
#           200    159/200   1/200
#           400    192/200   3/200
#
# So 400: isolated 96% against mixed 1.5%, and the bounds below (36/40, 2/40) sit outside both with
# room for a different 40-seed draw. That is pens of 40 needing a few hundred generations, which is
# why a split is something the player comes back to rather than watches (#16, #292). Re-measure if
# any of these numbers changes.
POPULATION = 40
GENERATIONS = 400
MUTABILITY = 0.02
DRIFT_MARGIN = 2.0
ORDINARY_STEPS = 1.8 * (2 * len(DISTANCE_GENES)) ** 0.5
THRESHOLD_STEPS = 2.0 * ORDINARY_STEPS
CHECK_ISOLATED_AT_LEAST = 36
CHECK_MIXED_AT_MOST = 2


def breed(genetics, population, rng):
    """Replace `population`'s genes with one generation of offspring bred within it.

    Parents are drawn at random with replacement from the population itself, which is what makes
    it a closed gene pool -- the "isolated by a fence or a mountain range" case (CLAUDE.md §2.5)
    reduced to its genetic content. Pairing is done here rather than through `Genetics.inherit`
    because that method pairs two selections positionally and a Selection is a set of rows, so it
    cannot express "row 3 mates with row 7, and row 3 again with row 12".

    The mutability floor is passed directly and the mutability gene is reset after the draw, so
    this measurement holds one variable fixed: what is under test is whether isolation separates
    two pools, not whether a lineage's evolvability itself drifts.
    """
    genes = genetics.genes(population)
    n = genes.shape[0]
    parent_a = genes[rng.integers(0, n, size=n)]
    parent_b = genes[rng.integers(0, n, size=n)]
    floor = np.full(n, MUTABILITY, dtype=np.float32)
    offspring = inherit_genes(parent_a, parent_b, floor, DRIFT_MARGIN, rng)
    offspring[:, MUTABILITY_COLUMN] = MUTABILITY
    genetics.set_genes(population, offspring)


def evolve(seed, isolated):
    """Run two sub-populations for GENERATIONS. Returns (genetics, left, right).

    isolated=True breeds each half only within itself; isolated=False breeds the two halves as one
    pool. Everything else -- founders, population size, generation count, gene count -- is
    identical between the two arms, so the only difference is whether genes cross the boundary.
    """
    rng = np.random.default_rng(seed)
    store, genetics, rows = make_world(2 * POPULATION)
    store.genes[rows] = rng.normal(0.5, 1.8 * MUTABILITY, size=(2 * POPULATION, len(GENE_NAMES)))
    store.genes[rows, MUTABILITY_COLUMN] = MUTABILITY

    left = selection_of(store, rows, range(POPULATION))
    right = selection_of(store, rows, range(POPULATION, 2 * POPULATION))
    everyone = left | right

    for _ in range(GENERATIONS):
        if isolated:
            breed(genetics, left, rng)
            breed(genetics, right, rng)
        else:
            breed(genetics, everyone, rng)
    return genetics, left, right


def splits(seed, isolated):
    """Whether `Speciation.check`, handed the whole population as one species, finds a split. It
    is told nothing about where the boundary was."""
    genetics, left, right = evolve(seed, isolated)
    speciation = speciation_for(
        genetics, THRESHOLD_STEPS, min_founders=10, ordinary=ORDINARY_STEPS
    )
    return len(speciation.check(left | right, tick=0)) == 1, genetics, left, right


class TestIsolationCausesSpeciation:
    """The payoff test (#16's "done when"): isolation reliably produces a new species and mixing
    reliably does not, through the real proposer and gate. Asserted as a rate over many seeds,
    never as a single run's outcome -- the simulation is deliberately non-deterministic (§2.2, §6)."""

    def test_isolated_populations_reliably_speciate(self):
        assert sum(splits(seed, isolated=True)[0] for seed in range(40)) >= CHECK_ISOLATED_AT_LEAST

    def test_an_interbreeding_population_reliably_does_not(self):
        """A *rate* bound, not zero: nothing forbids a mixed pool's two 2-means halves from
        wandering apart, and asserting zero against a measured rate is the knife edge §6 warns
        about."""
        assert sum(splits(seed, isolated=False)[0] for seed in range(40)) <= CHECK_MIXED_AT_MOST

    def test_after_the_split_the_two_species_cannot_interbreed(self):
        """End to end: drift apart in isolation, split, and the two populations stop being able to
        produce young together. Before the split they had already all but stopped."""
        did_split, genetics, left, right = splits(seed=0, isolated=True)
        assert did_split

        cross = interbreeding_probability(
            genetics, left.to_indices(), right.to_indices(), ORDINARY_STEPS, THRESHOLD_STEPS, COLUMNS
        )
        assert (cross == 0.0).all()
        assert genetics.species.n_species == 2
