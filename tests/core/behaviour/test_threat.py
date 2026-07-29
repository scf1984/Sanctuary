import numpy as np
import pytest

from core.behaviour.threat import Threat
from core.genetics.species import SpeciesRegistry
from core.genetics.vocabulary import GeneVocabulary

GENE_NAMES = ("size", "scent")


def registry(n_species):
    species = SpeciesRegistry(GeneVocabulary(GENE_NAMES))
    for _ in range(n_species):
        species.register(GENE_NAMES)
    return species


class TestConstruction:
    def test_rejects_a_matrix_that_does_not_match_the_registry(self):
        with pytest.raises(ValueError):
            Threat(registry(3), np.zeros((2, 2), dtype=np.float32))

    def test_rejects_negative_weights(self):
        """A negative weight is attraction wearing fear's name; seeking something out is a
        different drive, not fear with the sign flipped.
        """
        with pytest.raises(ValueError):
            Threat(registry(2), np.array([[0.0, -1.0], [0.0, 0.0]], dtype=np.float32))


class TestAsymmetry:
    def test_the_matrix_need_not_be_symmetric(self):
        """Prey fear predators far more than predators fear prey. That asymmetry is the thing an
        ecology is built on, so nothing may quietly symmetrize it.
        """
        prey, predator = 0, 1
        threat = Threat(registry(2), np.array([[0.0, 0.9], [0.1, 0.0]], dtype=np.float32))

        assert threat.weights[prey, predator] == pytest.approx(0.9)
        assert threat.weights[predator, prey] == pytest.approx(0.1)


class TestCannibalism:
    def test_the_diagonal_is_an_ordinary_entry(self):
        """Fear of your own kind needs no special case anywhere — it is W[s, s], authored like
        every other pair, and a species that does not eat its young simply has zero there.
        """
        cannibal, peaceable = 0, 1
        threat = Threat(registry(2), np.array([[0.4, 0.0], [0.0, 0.0]], dtype=np.float32))

        rows = threat.rows_for(np.array([cannibal, peaceable]))

        assert rows[0, cannibal] == pytest.approx(0.4)
        assert rows[1, peaceable] == pytest.approx(0.0)


class TestDerive:
    def test_a_daughter_inherits_what_its_parent_feared(self):
        """At the moment of a split the daughter is ecologically identical to its parent. Drift
        is what separates them afterwards (CLAUDE.md §2.5).
        """
        parent, wolf = 0, 1
        threat = Threat(registry(2), np.array([[0.0, 0.8], [0.0, 0.0]], dtype=np.float32))

        daughter = threat.derive(parent)

        assert threat.weights[daughter, wolf] == pytest.approx(0.8)

    def test_a_daughter_inherits_how_others_feared_its_parent(self):
        """The column, not just the row: a wolf's prey must fear the daughter exactly as it
        feared the parent, or a split would silently make a predator harmless.
        """
        prey, predator = 0, 1
        threat = Threat(registry(2), np.array([[0.0, 0.9], [0.0, 0.0]], dtype=np.float32))

        daughter = threat.derive(predator)

        assert threat.weights[prey, daughter] == pytest.approx(0.9)

    def test_a_daughter_regards_its_parent_as_it_regarded_itself(self):
        """Two populations that were one species a tick ago are strangers to nobody yet."""
        cannibal = 0
        threat = Threat(registry(1), np.array([[0.5]], dtype=np.float32))

        daughter = threat.derive(cannibal)

        assert threat.weights[daughter, daughter] == pytest.approx(0.5)

    def test_derive_keeps_the_registry_and_the_matrix_in_step(self):
        """The registry is the source of species ids. If these two ever disagreed about how many
        species exist, a species id would index past the end of the matrix.
        """
        species = registry(2)
        threat = Threat(species, np.zeros((2, 2), dtype=np.float32))

        daughter = threat.derive(0)

        assert daughter == 2
        assert species.n_species == 3
        assert threat.weights.shape == (3, 3)

    def test_existing_weights_survive_a_split(self):
        species = registry(2)
        original = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        threat = Threat(species, original.copy())

        threat.derive(0)

        assert threat.weights[:2, :2] == pytest.approx(original)


class TestRowsFor:
    def test_a_mixed_species_population_is_gathered_in_one_pass(self):
        threat = Threat(registry(3), np.arange(9, dtype=np.float32).reshape(3, 3))

        rows = threat.rows_for(np.array([2, 0, 2]))

        assert rows == pytest.approx(
            np.array([[6.0, 7.0, 8.0], [0.0, 1.0, 2.0], [6.0, 7.0, 8.0]])
        )
