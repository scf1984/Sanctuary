"""Senescence: performance traits degrade with age, and nothing else does (CLAUDE.md §2.5, #239).

The contract is fully checkable in advance (§8.1): the decay curve is a pure function of age and
one gene, and where it applies is a statement about which phenotype columns move. What is *not*
asserted here is any lifespan. That is an outcome the degradation rate is tuned toward (§2.1), not
something the mechanism decides, and it is measured on the demo world rather than pinned.
"""

import numpy as np
import pytest

from core.entities.store import EntityStore
from core.genetics.expression import GeneticsConfig
from core.genetics.registry import ExpressionMode, GeneRegistry, GeneSpec, Unit
from core.genetics.senescence import SenescenceConfig, decay
from core.genetics.service import Genetics
from core.genetics.species import SpeciesRegistry
from core.invariants import senescence_decay_stays_in_unit_interval
from core.selection import Selection
from core.services import ColumnRegistry

from tests.support.genes import gene_registry

GENE_NAMES = ("size", "speed", "sight", "scent_acuity", "senescence_resistance", "mutability")
GENE_REGISTRY = gene_registry(GENE_NAMES)
SIZE, SPEED, SIGHT, SCENT, RESISTANCE = range(5)

SENESCENCE = SenescenceConfig(
    resistance_gene="senescence_resistance",
    base_decay=0.01,
    decaying_genes=("speed", "sight", "scent_acuity"),
)


def genetics_config(senescence=SENESCENCE):
    return GeneticsConfig(mutability_gene="mutability", drift_margin=2.0, senescence=senescence)


def make_world(senescence=SENESCENCE, registry=GENE_REGISTRY):
    store = EntityStore(initial_capacity=8, n_drives=1, n_genes=len(registry.vocabulary))
    species = SpeciesRegistry(registry)
    # One species expressing every gene: the mask is not what is under test.
    species.register(registry.vocabulary.names)
    genetics = Genetics(store, ColumnRegistry(), species, registry, genetics_config(senescence))
    return store, genetics


def twins(store, ages, resistance=0.0):
    """Identical genomes at the given ages: any phenotype difference between them is age alone."""
    genes = np.tile(np.array([1.0, 2.0, 3.0, 4.0, resistance, 0.1], dtype=np.float32), (len(ages), 1))
    ids = store.allocate(
        len(ages),
        genes=genes,
        age=np.asarray(ages, dtype=np.int64),
        species_id=np.zeros(len(ages), dtype=np.int32),
    )
    return np.array([store._id_to_row[i] for i in ids.tolist()], dtype=np.int64)


class TestTheCurve:
    def test_a_newborn_has_not_decayed_at_all(self):
        assert decay(np.array([0]), np.array([0.0]), 0.01)[0] == 1.0

    def test_decay_falls_monotonically_with_age_so_nothing_rejuvenates(self):
        ages = np.arange(0, 5_000, 7)
        values = decay(ages, np.zeros(len(ages)), 0.01)
        assert np.all(np.diff(values) < 0.0)

    def test_decay_stays_strictly_inside_zero_to_one_at_any_age_the_world_can_reach(self):
        # The exponential approaches zero without reaching it, so an old animal is negligibly slow
        # rather than motionless (§2.5). 50,000 ticks is far past any measured death age.
        ages = np.array([0, 1, 100, 10_000, 50_000])
        values = decay(ages, np.zeros(len(ages)), 0.01)
        assert np.all(values > 0.0)
        assert np.all(values <= 1.0)

    def test_more_resistance_decays_slower_with_diminishing_returns(self):
        age = np.full(4, 200)
        values = decay(age, np.array([0.0, 1.0, 2.0, 3.0]), 0.01)
        assert np.all(np.diff(values) > 0.0)
        # The `1 +` denominator: each extra unit of resistance buys less than the last, so the gene
        # cannot buy its way out of the mechanism into immortality.
        assert np.all(np.diff(np.diff(-np.log(values))) > 0.0)

    def test_resistance_is_read_as_a_magnitude(self):
        assert decay(np.array([300]), np.array([-2.0]), 0.01)[0] == decay(
            np.array([300]), np.array([2.0]), 0.01
        )[0]

    def test_an_unborn_animal_has_not_aged(self):
        # A gestating row carries a negative age (#20). It has lived no ticks, so it has not
        # decayed, and the curve must not read the negative age as rejuvenation above 1.
        assert decay(np.array([-40]), np.array([0.0]), 0.01)[0] == 1.0


class TestConfigValidation:
    @pytest.mark.parametrize("base_decay", [0.0, -0.01, float("nan"), float("inf")])
    def test_rejects_a_base_decay_that_is_not_positive_and_finite(self, base_decay):
        with pytest.raises(ValueError, match="base_decay"):
            SenescenceConfig("senescence_resistance", base_decay, ("speed",))

    def test_rejects_a_gene_listed_to_decay_twice(self):
        with pytest.raises(ValueError, match="twice"):
            SenescenceConfig("senescence_resistance", 0.01, ("speed", "speed"))

    def test_the_resistance_gene_must_be_read_as_a_magnitude(self):
        signed = GeneRegistry(
            tuple(
                GeneSpec(
                    name,
                    0.0,
                    ExpressionMode.SIGNED if name == "senescence_resistance" else ExpressionMode.MAGNITUDE,
                    Unit.LENGTH if name == "speed" else Unit.DIMENSIONLESS,
                    name,
                )
                for name in GENE_NAMES
            )
        )
        with pytest.raises(ValueError, match="magnitude"):
            make_world(registry=signed)

    def test_a_decaying_gene_must_be_a_quantity(self):
        with pytest.raises(ValueError, match="magnitude"):
            make_world(
                senescence=SenescenceConfig("senescence_resistance", 0.01, ("signature_0",)),
                registry=gene_registry(GENE_NAMES + ("signature_0",)),
            )

    def test_an_unknown_decaying_gene_is_refused_rather_than_skipped(self):
        with pytest.raises(KeyError, match="toxicity"):
            make_world(senescence=SenescenceConfig("senescence_resistance", 0.01, ("toxicity",)))


class TestWhereItApplies:
    def test_an_old_animal_expresses_less_of_every_performance_trait(self):
        store, genetics = make_world()
        young, old = genetics.expressed_at(twins(store, [0, 500]))
        for column in (SPEED, SIGHT, SCENT):
            assert old[column] < young[column]
        assert old[SPEED] == pytest.approx(young[SPEED] * np.exp(-0.01 * 500), rel=1e-6)

    def test_capacity_and_everything_unlisted_does_not_decay(self):
        store, genetics = make_world()
        young, old = genetics.expressed_at(twins(store, [0, 500]))
        for column in (SIZE, RESISTANCE):
            assert old[column] == young[column]

    def test_a_resistant_animal_has_kept_more_at_the_same_age(self):
        store, genetics = make_world()
        frail = genetics.expressed_at(twins(store, [300], resistance=0.0))[0]
        hardy = genetics.expressed_at(twins(store, [300], resistance=3.0))[0]
        assert hardy[SPEED] > frail[SPEED]

    def test_the_heritable_phenotype_does_not_age(self):
        # Genetic distance and the diversity metric compare genomes, not condition: two identical
        # genomes must read as identical however far apart their ages are.
        store, genetics = make_world()
        young, old = genetics.heritable_at(twins(store, [0, 5_000]))
        np.testing.assert_array_equal(young, old)

    def test_a_world_whose_fixture_opts_out_never_ages(self):
        store, genetics = make_world(senescence=None)
        young, old = genetics.expressed_at(twins(store, [0, 5_000]))
        np.testing.assert_array_equal(young, old)


class TestTheInvariant:
    def test_holds_over_a_population_of_every_age(self):
        store, genetics = make_world()
        twins(store, [0, 1, 300, 5_000, -20])
        assert senescence_decay_stays_in_unit_interval(genetics)(store) is None

    def test_trips_on_a_formula_that_reaches_zero(self, monkeypatch):
        # The invariant guards against a future change to the formula (#21's follow-up), so the
        # test is a deliberately broken formula, not a broken config the validation would refuse.
        store, genetics = make_world()
        rows = twins(store, [0, 300])
        monkeypatch.setattr(
            "core.genetics.service.decay", lambda age, resistance, base: np.where(age > 0, 0.0, 1.0)
        )
        violation = senescence_decay_stays_in_unit_interval(genetics)(store)
        assert violation is not None
        np.testing.assert_array_equal(violation.rows, rows[1:])

    def test_trips_on_a_formula_that_rejuvenates(self, monkeypatch):
        store, genetics = make_world()
        twins(store, [10])
        monkeypatch.setattr(
            "core.genetics.service.decay", lambda age, resistance, base: np.full(len(age), 1.5)
        )
        assert senescence_decay_stays_in_unit_interval(genetics)(store) is not None

    def test_reads_only_the_living(self, monkeypatch):
        # A freed row keeps its stale age until the row is reused, and is nobody's phenotype.
        store, genetics = make_world()
        rows = twins(store, [10, 20])
        store.release(store.row_ids()[rows[:1]])
        monkeypatch.setattr(
            "core.genetics.service.decay",
            lambda age, resistance, base: np.where(age == 10, 0.0, 1.0),
        )
        assert senescence_decay_stays_in_unit_interval(genetics)(store) is None


class TestReadingSomeColumns:
    """`expressed_columns` is `expressed` with the unread columns never built (#258).

    A drive that reads one gene used to pay for the whole (n, n_genes) block, seventeen times per
    `choose()`. The column read is only worth having if it is the *same* phenotype, so the contract
    is exact equality with the full block's columns, through every mode, a partial species mask,
    and senescence at every age — not closeness.
    """

    NAMES = (
        "size", "speed", "sight", "scent_acuity", "senescence_resistance", "mutability",
        "choice_temperature", "diet_animal_derived", "signature_0", "aversion0_0",
    )

    def population(self):
        registry = gene_registry(self.NAMES)
        store = EntityStore(initial_capacity=64, n_drives=1, n_genes=len(self.NAMES))
        species = SpeciesRegistry(registry)
        species.register(registry.vocabulary.names)
        # A second species that does not express sight, the resistance or a cue gene, so masking
        # has to be applied to the subset rather than assumed away.
        unexpressed = ("sight", "senescence_resistance", "signature_0")
        species.register(tuple(n for n in self.NAMES if n not in unexpressed))
        genetics = Genetics(store, ColumnRegistry(), species, registry, genetics_config())
        rng = np.random.default_rng(3)
        n = 50
        ids = store.allocate(
            n,
            genes=rng.normal(0.0, 2.0, size=(n, len(self.NAMES))).astype(np.float32),
            age=rng.integers(-5, 400, size=n),
            species_id=rng.integers(0, 2, size=n).astype(np.int32),
        )
        rows = np.array([store._id_to_row[i] for i in ids.tolist()], dtype=np.int64)
        return genetics, Selection.from_indices(rows, capacity=store.capacity)

    @pytest.mark.parametrize(
        "columns",
        [[1], [2, 4], [6, 7, 8, 9], [9, 0, 3], list(range(10))],
        ids=["one decaying", "decaying and resistance", "every other mode", "out of order", "all"],
    )
    def test_equals_the_same_columns_of_the_full_block_exactly(self, columns):
        genetics, selection = self.population()
        full = genetics.expressed(selection)
        some = genetics.expressed_columns(selection, np.array(columns))
        assert some.dtype == np.float32
        np.testing.assert_array_equal(some, full[:, columns])
