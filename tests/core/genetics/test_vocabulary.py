import pytest

from core.genetics.vocabulary import DuplicateGeneError, GeneVocabulary


class TestConstruction:
    def test_rejects_empty_vocabulary(self):
        with pytest.raises(ValueError):
            GeneVocabulary(())

    def test_rejects_duplicate_gene_names(self):
        with pytest.raises(DuplicateGeneError):
            GeneVocabulary(("size", "speed", "size"))

    def test_starts_at_version_one_by_default(self):
        vocabulary = GeneVocabulary(("size", "speed"))
        assert vocabulary.version == 1

    def test_len_is_gene_count(self):
        vocabulary = GeneVocabulary(("size", "speed", "sight"))
        assert len(vocabulary) == 3


class TestIndexOf:
    def test_index_matches_declaration_order(self):
        vocabulary = GeneVocabulary(("size", "speed", "sight"))
        assert vocabulary.index_of("size") == 0
        assert vocabulary.index_of("speed") == 1
        assert vocabulary.index_of("sight") == 2

    def test_unknown_gene_raises(self):
        vocabulary = GeneVocabulary(("size",))
        with pytest.raises(KeyError):
            vocabulary.index_of("speed")


class TestWiden:
    def test_widen_requires_at_least_one_new_gene(self):
        vocabulary = GeneVocabulary(("size",))
        with pytest.raises(ValueError):
            vocabulary.widen()

    def test_widened_vocabulary_keeps_existing_gene_indices(self):
        v1 = GeneVocabulary(("size", "speed"))
        v2 = v1.widen("sight")
        assert v2.index_of("size") == 0
        assert v2.index_of("speed") == 1
        assert v2.index_of("sight") == 2
        assert len(v2) == 3

    def test_widen_increments_version(self):
        v1 = GeneVocabulary(("size",), version=1)
        v2 = v1.widen("speed")
        assert v2.version == 2

    def test_widen_does_not_mutate_the_source_vocabulary(self):
        v1 = GeneVocabulary(("size",))
        v1.widen("speed")
        assert len(v1) == 1
        assert v1.version == 1

    def test_widen_rejects_a_name_already_in_the_vocabulary(self):
        v1 = GeneVocabulary(("size", "speed"))
        with pytest.raises(DuplicateGeneError):
            v1.widen("speed")
