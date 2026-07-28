import numpy as np
import pytest

from core.ecology.metabolism import Metabolism, MetabolismConfig
from core.genetics.vocabulary import GeneVocabulary


GENE_NAMES = ("size", "speed", "sight", "insulation")


def make_metabolism(**overrides):
    defaults = dict(
        gene_costs={"size": 2.0, "speed": 3.0, "sight": 0.0, "insulation": 1.0},
        basal_rate=1.0,
        thermoregulation_rate=0.5,
        neutral_temperature=20.0,
        insulation_gene="insulation",
    )
    defaults.update(overrides)
    return Metabolism(GeneVocabulary(GENE_NAMES), MetabolismConfig(**defaults))


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
                gene_costs={"size": 2.0, "speed": -3.0, "sight": 0.0, "insulation": 1.0}
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
                gene_costs={"size": 2.0, "speed": 3.0, "sight": 0.0, "insulation": 0.0}
            )

    def test_an_insulation_gene_outside_the_vocabulary_is_rejected(self):
        with pytest.raises(KeyError):
            make_metabolism(insulation_gene="blubber")


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
