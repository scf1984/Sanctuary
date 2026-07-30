import numpy as np
import pytest

from core.ecology.metabolism import Metabolism, MetabolismConfig
from core.genetics.expression import ExpressionMode
from core.genetics.vocabulary import GeneVocabulary


# `aversion0_0` is here to make the fixture representative rather than convenient: a real
# vocabulary mixes quantities with cue-space directions, and the rule this module enforces is
# about the boundary between them (#136).
GENE_NAMES = ("size", "speed", "sight", "insulation", "aversion0_0")

EXPRESSION_MODES = {
    "size": ExpressionMode.MAGNITUDE,
    "speed": ExpressionMode.MAGNITUDE,
    "sight": ExpressionMode.MAGNITUDE,
    "insulation": ExpressionMode.MAGNITUDE,
    "aversion0_0": ExpressionMode.SIGNED,
}


def make_metabolism(expression_modes=EXPRESSION_MODES, **overrides):
    defaults = dict(
        gene_costs={
            "size": 2.0,
            "speed": 3.0,
            "sight": 0.0,
            "insulation": 1.0,
            "aversion0_0": 0.0,
        },
        basal_rate=1.0,
        thermoregulation_rate=0.5,
        neutral_temperature=20.0,
        insulation_gene="insulation",
    )
    defaults.update(overrides)
    return Metabolism(
        GeneVocabulary(GENE_NAMES), MetabolismConfig(**defaults), expression_modes
    )


def expressed(**genes):
    """One (1, n_genes) expressed-phenotype row, named by gene rather than column index."""
    row = np.zeros((1, len(GENE_NAMES)), dtype=np.float32)
    for name, value in genes.items():
        row[0, GENE_NAMES.index(name)] = value
    return row


def upkeep_of(metabolism, temperature=20.0, **genes):
    return float(
        metabolism.upkeep(expressed(**genes), np.array([temperature], dtype=np.float32))[0]
    )


class TestConfigValidation:
    """The config is the whole "no free lunch" contract (CLAUDE.md §2.5), so every way of
    accidentally granting one has to fail at construction rather than at some later population
    explosion nobody traces back to here.
    """

    def test_a_gene_with_no_declared_cost_is_rejected(self):
        with pytest.raises(ValueError, match="sight"):
            make_metabolism(gene_costs={"size": 2.0, "speed": 3.0, "insulation": 1.0})

    def test_a_cost_for_a_gene_outside_the_vocabulary_is_rejected(self):
        with pytest.raises(ValueError, match="venom"):
            make_metabolism(
                gene_costs={
                    "size": 2.0,
                    "speed": 3.0,
                    "sight": 0.0,
                    "insulation": 1.0,
                    "venom": 4.0,
                }
            )

    def test_a_negative_gene_cost_is_rejected(self):
        with pytest.raises(ValueError, match="speed"):
            make_metabolism(
                gene_costs={
                    "size": 2.0,
                    "speed": -3.0,
                    "sight": 0.0,
                    "insulation": 1.0,
                    "aversion0_0": 0.0,
                }
            )

    def test_a_negative_basal_rate_is_rejected(self):
        with pytest.raises(ValueError, match="basal_rate"):
            make_metabolism(basal_rate=-1.0)

    def test_a_negative_thermoregulation_rate_is_rejected(self):
        with pytest.raises(ValueError, match="thermoregulation_rate"):
            make_metabolism(thermoregulation_rate=-0.5)

    def test_an_insulation_gene_that_costs_nothing_is_rejected(self):
        # Insulation only ever reduces upkeep, so a free one is unbounded free benefit -- the
        # exact runaway this issue exists to prevent.
        with pytest.raises(ValueError, match="insulation"):
            make_metabolism(
                gene_costs={
                    "size": 2.0,
                    "speed": 3.0,
                    "sight": 0.0,
                    "insulation": 0.0,
                    "aversion0_0": 0.0,
                }
            )

    def test_an_insulation_gene_outside_the_vocabulary_is_rejected(self):
        with pytest.raises(KeyError):
            make_metabolism(insulation_gene="blubber")


class TestOnlyMagnitudeGenesMayCost:
    """A cost is only bounded below by zero if the phenotype it multiplies is (#136).

    Storage is signed (#104), and what makes an expressed value non-negative is the gene's
    expression mode, not inheritance. So a `SIGNED` gene — a cue-space direction, founded across
    zero by design — multiplied by a positive cost contributes a **negative** term to upkeep.

    That is not a crash waiting to happen so much as a subsidy: upkeep is a sum, so a negative
    term merely discounts the total, and the animal is charged less for holding its aversion one
    way round than the other. Selection acts on the discount, which is the free lunch §2.5's hard
    budget exists to forbid, running in reverse. Only when the discount exceeds everything else
    does the total go negative, and `Ecology.spend` rejects that — mid-tick, naming a module that
    did nothing wrong. Both failures are the same misconfiguration, and it is knowable at
    construction.
    """

    def test_a_signed_gene_carrying_a_cost_is_rejected(self):
        with pytest.raises(ValueError, match="aversion0_0"):
            make_metabolism(
                gene_costs={
                    "size": 2.0,
                    "speed": 3.0,
                    "sight": 0.0,
                    "insulation": 1.0,
                    "aversion0_0": 0.01,
                }
            )

    def test_a_signed_gene_costing_nothing_is_accepted(self):
        # §2.5 gives every cue gene a cost of zero, so a config following the document builds.
        assert make_metabolism() is not None

    def test_a_costed_gene_with_no_declared_mode_is_rejected(self):
        # Not redundant with ExpressionTable's own completeness check: a `Metabolism` can be
        # built without one ever existing, and skipping the check for want of a mode is exactly
        # the silent default §8.7 forbids.
        modes = {name: mode for name, mode in EXPRESSION_MODES.items() if name != "speed"}
        with pytest.raises(ValueError, match="speed"):
            make_metabolism(expression_modes=modes)

    def test_the_rule_is_about_the_mode_and_not_the_gene_name(self):
        # A quantity read as a magnitude may cost whatever it likes, whatever it is called: it is
        # the mode that guarantees a non-negative phenotype, so the same name flips from rejected
        # to accepted on the mode alone.
        metabolism = make_metabolism(
            expression_modes={**EXPRESSION_MODES, "aversion0_0": ExpressionMode.MAGNITUDE},
            gene_costs={
                "size": 2.0,
                "speed": 3.0,
                "sight": 0.0,
                "insulation": 1.0,
                "aversion0_0": 4.0,
            },
        )
        assert upkeep_of(metabolism, aversion0_0=1.0) == pytest.approx(5.0)


class TestTraitUpkeep:
    def test_upkeep_is_basal_when_nothing_is_expressed_at_neutral_temperature(self):
        metabolism = make_metabolism()
        assert upkeep_of(metabolism) == pytest.approx(1.0)

    def test_each_expressed_gene_charges_its_declared_cost_per_unit(self):
        metabolism = make_metabolism()
        # basal 1.0 + size 2.0*2 + speed 3.0*1
        assert upkeep_of(metabolism, size=2.0, speed=1.0) == pytest.approx(8.0)

    def test_raising_a_trait_raises_upkeep(self):
        metabolism = make_metabolism()
        cheap = upkeep_of(metabolism, speed=1.0)
        costly = upkeep_of(metabolism, speed=4.0)
        assert costly > cheap

    def test_a_zero_cost_gene_charges_nothing(self):
        metabolism = make_metabolism()
        assert upkeep_of(metabolism, sight=10.0) == upkeep_of(metabolism)

    def test_upkeep_is_one_value_per_row(self):
        metabolism = make_metabolism()
        rows = np.zeros((5, len(GENE_NAMES)), dtype=np.float32)
        temperature = np.full(5, 20.0, dtype=np.float32)

        upkeep = metabolism.upkeep(rows, temperature)

        assert upkeep.shape == (5,)
        assert upkeep.dtype == np.float32


class TestThermoregulation:
    def test_costs_nothing_at_the_neutral_temperature(self):
        metabolism = make_metabolism()
        assert upkeep_of(metabolism, temperature=20.0) == pytest.approx(1.0)

    def test_cost_rises_with_deviation_in_either_direction(self):
        metabolism = make_metabolism()
        neutral = upkeep_of(metabolism, temperature=20.0)
        cold = upkeep_of(metabolism, temperature=0.0)
        hot = upkeep_of(metabolism, temperature=40.0)

        assert cold > neutral
        assert hot > neutral
        # Deviation, not signed difference: equal distance either side costs the same.
        assert cold == pytest.approx(hot)

    def test_insulation_reduces_thermoregulation_cost(self):
        metabolism = make_metabolism()
        bare = upkeep_of(metabolism, temperature=0.0, insulation=0.0)
        insulated = upkeep_of(metabolism, temperature=0.0, insulation=3.0)

        # Insulation charges its own upkeep too, so compare the thermal share directly rather
        # than the totals: 3 units of insulation cost 3.0 J/tick and must save more than that.
        assert insulated < bare
        assert insulated - 3.0 < bare

    def test_insulation_is_pure_cost_at_the_neutral_temperature(self):
        metabolism = make_metabolism()
        bare = upkeep_of(metabolism, temperature=20.0, insulation=0.0)
        insulated = upkeep_of(metabolism, temperature=20.0, insulation=3.0)

        # This crossover is what makes climate select genetically (CLAUDE.md §2.5): the same
        # gene that pays for itself in the cold is dead weight where there is nothing to insulate
        # against.
        assert insulated == pytest.approx(bare + 3.0)


class TestNoEnergyIsCreated:
    def test_upkeep_is_never_negative_across_a_wide_sweep_of_phenotypes(self):
        metabolism = make_metabolism()
        rng = np.random.default_rng(7)
        rows = rng.uniform(0.0, 20.0, size=(500, len(GENE_NAMES))).astype(np.float32)
        temperature = rng.uniform(-40.0, 60.0, size=500).astype(np.float32)

        assert (metabolism.upkeep(rows, temperature) >= 0.0).all()

    def test_upkeep_is_at_least_the_basal_rate(self):
        metabolism = make_metabolism()
        rng = np.random.default_rng(11)
        rows = rng.uniform(0.0, 20.0, size=(200, len(GENE_NAMES))).astype(np.float32)
        temperature = rng.uniform(-40.0, 60.0, size=200).astype(np.float32)

        assert (metabolism.upkeep(rows, temperature) >= 1.0).all()

    def test_upkeep_holds_when_signed_phenotypes_are_negative(self):
        """The sweeps above draw phenotypes from `[0, 20]`, which is why neither of them could
        ever have caught #136: a real expressed phenotype is *not* non-negative any more. Cue
        genes are read `SIGNED` and arrive as stored, so this draws across zero — and what keeps
        the total non-negative is that the constructor refused to cost those columns.
        """
        metabolism = make_metabolism()
        rng = np.random.default_rng(13)
        rows = rng.uniform(0.0, 20.0, size=(500, len(GENE_NAMES))).astype(np.float32)
        rows[:, GENE_NAMES.index("aversion0_0")] = rng.uniform(-20.0, 20.0, size=500)
        temperature = rng.uniform(-40.0, 60.0, size=500).astype(np.float32)

        assert (metabolism.upkeep(rows, temperature) >= 0.0).all()
