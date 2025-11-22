"""Tests for Vocab class."""

import pytest
import jax.numpy as jnp
from pathlib import Path
import tempfile

from mo_net.samples.word2vec.vocab import Vocab


class TestVocabCreation:
    """Test vocabulary creation and building."""

    def test_from_sentences_basic(self):
        """Test basic vocabulary creation from sentences."""
        sentences = [["the", "quick", "brown", "fox"], ["the", "lazy", "dog"]]
        vocab, _ = Vocab.from_sentences(sentences, max_size=100)

        # Check that all words are in vocab
        for word in ["the", "quick", "brown", "fox", "lazy", "dog"]:
            assert word in vocab.token_to_id

        # Check vocab size (unique words + UNK)
        assert len(vocab) == 7  # 6 unique words + UNK

    def test_from_sentences_with_max_size(self):
        """Test vocabulary creation with maximum size threshold."""
        sentences = [["the"] * 10, ["rare", "word"], ["another", "rare"]]
        vocab, _ = Vocab.from_sentences(sentences, max_size=1)

        # Only "the" (most common) should be included
        assert "the" in vocab.token_to_id

        # Others should not be in vocab
        assert "rare" not in vocab.token_to_id
        assert "word" not in vocab.token_to_id

    def test_from_sentences_with_forced_words(self):
        """Test that forced words are always included regardless of count."""
        sentences = [["common"] * 10, ["rare"]]
        vocab, _ = Vocab.from_sentences(
            sentences, max_size=1, forced_words=["rare", "nonexistent"]
        )

        # "rare" appears only once but should be included (forced)
        assert "rare" in vocab.token_to_id

        # "nonexistent" doesn't appear but should be included (forced)
        assert "nonexistent" in vocab.token_to_id

        # "common" appears 10 times, should be included
        assert "common" in vocab.token_to_id

    def test_unknown_token_always_present(self):
        """Test that unknown token is always in vocabulary."""
        sentences = [["a", "b", "c"]]
        vocab, _ = Vocab.from_sentences(sentences, max_size=100)

        # Unknown token should be accessible
        assert vocab.unknown_token_id == len(vocab.vocab)

    def test_word_counts(self):
        """Test that word counts are tracked correctly."""
        sentences = [["the"] * 5, ["quick", "quick", "brown"]]
        vocab, _ = Vocab.from_sentences(sentences, max_size=100)

        # Access word counts through negative sampling distribution
        # Words with higher counts should have higher probabilities
        dist = vocab.get_negative_sampling_distribution()

        the_idx = vocab["the"]
        quick_idx = vocab["quick"]
        brown_idx = vocab["brown"]

        # "the" appears 5 times, "quick" 2 times, "brown" 1 time
        assert dist[the_idx] > dist[quick_idx]
        assert dist[quick_idx] > dist[brown_idx]


class TestVocabTokenization:
    """Test tokenization and word-to-ID conversions."""

    def test_getitem_known_words(self):
        """Test conversion of known words to IDs using __getitem__."""
        sentences = [["apple", "banana", "cherry"]]
        vocab, _ = Vocab.from_sentences(sentences, max_size=100)

        # Each word should have a unique ID
        apple_id = vocab["apple"]
        banana_id = vocab["banana"]
        cherry_id = vocab["cherry"]

        assert apple_id != banana_id
        assert banana_id != cherry_id
        assert apple_id != cherry_id

        # IDs should be valid indices
        assert 0 <= apple_id < len(vocab)
        assert 0 <= banana_id < len(vocab)
        assert 0 <= cherry_id < len(vocab)

    def test_getitem_unknown_word(self):
        """Test that unknown words map to unknown token ID."""
        sentences = [["known", "word"]]
        vocab, _ = Vocab.from_sentences(sentences, max_size=100)

        unk_id = vocab["unknown"]
        expected_unk_id = vocab.unknown_token_id

        assert unk_id == expected_unk_id

    def test_id_to_token_roundtrip(self):
        """Test that token -> ID -> token roundtrip works."""
        sentences = [["hello", "world"]]
        vocab, _ = Vocab.from_sentences(sentences, max_size=100)

        for word in ["hello", "world"]:
            word_id = vocab[word]
            recovered_word = vocab.id_to_token[word_id]
            assert word == recovered_word

    def test_sentence_to_ids(self):
        """Test converting a sentence to IDs."""
        sentences = [["the", "cat", "sat"]]
        vocab, _ = Vocab.from_sentences(sentences, max_size=100)

        # Convert known sentence
        ids = [vocab[word] for word in ["the", "cat", "sat"]]

        assert len(ids) == 3
        assert all(0 <= id < len(vocab) for id in ids)
        assert len(set(ids)) == 3  # All different

    def test_sentence_to_ids_with_unknown(self):
        """Test converting sentence with unknown words."""
        sentences = [["known"]]
        vocab, _ = Vocab.from_sentences(sentences, max_size=100)

        sentence = ["known", "unknown1", "unknown2"]
        ids = [vocab[word] for word in sentence]

        unk_id = vocab.unknown_token_id

        # First word is known, others map to UNK
        assert ids[0] != unk_id
        assert ids[1] == unk_id
        assert ids[2] == unk_id


class TestVocabSerialization:
    """Test vocabulary serialization and deserialization."""

    def test_serialize_deserialize_roundtrip(self):
        """Test that serialize/deserialize preserves vocabulary."""
        sentences = [
            ["the", "quick", "brown", "fox"],
            ["jumps", "over", "the", "lazy", "dog"],
        ]
        original_vocab, _ = Vocab.from_sentences(sentences, max_size=100)

        # Serialize to file
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "vocab.msgpack"

            with open(file_path, "wb") as f:
                f.write(original_vocab.serialize())

            # Deserialize from file
            restored_vocab = Vocab.deserialize(file_path)

            # Check that vocabularies are equivalent
            assert len(original_vocab) == len(restored_vocab)
            assert original_vocab.unknown_token_id == restored_vocab.unknown_token_id

            # Check all words map to same IDs
            for word in [
                "the",
                "quick",
                "brown",
                "fox",
                "jumps",
                "over",
                "lazy",
                "dog",
            ]:
                assert original_vocab[word] == restored_vocab[word]

    def test_from_bytes_roundtrip(self):
        """Test from_bytes serialization method."""
        sentences = [["hello", "world"]]
        original_vocab, _ = Vocab.from_sentences(sentences, max_size=100)

        # Serialize and deserialize
        serialized = original_vocab.serialize()
        restored_vocab = Vocab.from_bytes(serialized)

        # Verify
        assert len(original_vocab) == len(restored_vocab)
        assert original_vocab["hello"] == restored_vocab["hello"]
        assert original_vocab["world"] == restored_vocab["world"]

    def test_serialization_preserves_word_counts(self):
        """Test that serialization preserves word frequency information."""
        sentences = [["frequent"] * 10, ["rare"]]
        original_vocab, _ = Vocab.from_sentences(sentences, max_size=100)

        # Get original distribution
        original_dist = original_vocab.get_negative_sampling_distribution()

        # Serialize and deserialize
        serialized = original_vocab.serialize()
        restored_vocab = Vocab.from_bytes(serialized)

        # Get restored distribution
        restored_dist = restored_vocab.get_negative_sampling_distribution()

        # Distributions should be identical
        assert jnp.allclose(original_dist, restored_dist)


class TestVocabEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_sentences(self):
        """Test vocabulary creation with empty sentences."""
        sentences = []
        vocab, _ = Vocab.from_sentences(sentences, max_size=100)

        # Should at least have UNK token
        assert len(vocab) >= 1

    def test_sentences_with_empty_strings(self):
        """Test handling of empty strings in sentences."""
        sentences = [["word1", "", "word2"]]
        vocab, _ = Vocab.from_sentences(sentences, max_size=100)

        # Empty string might be treated as valid token or filtered
        # At minimum, should not crash
        assert len(vocab) >= 1

    def test_very_large_vocabulary(self):
        """Test vocabulary with many unique words."""
        # Generate many unique words
        num_words = 1000
        sentences = [[f"word_{i}" for i in range(num_words)]]
        vocab, _ = Vocab.from_sentences(sentences, max_size=num_words)

        # Should handle large vocabulary
        assert len(vocab) >= num_words

    def test_unicode_words(self):
        """Test vocabulary with unicode characters."""
        sentences = [["hello", "世界", "مرحبا", "🌍"]]
        vocab, _ = Vocab.from_sentences(sentences, max_size=100)

        # Should handle unicode properly
        assert "世界" in vocab.token_to_id
        assert "مرحبا" in vocab.token_to_id
        assert "🌍" in vocab.token_to_id

    def test_case_sensitivity(self):
        """Test that vocabulary is case-sensitive."""
        sentences = [["Word", "word", "WORD"]]
        vocab, _ = Vocab.from_sentences(sentences, max_size=100)

        # All three should be different words
        assert vocab["Word"] != vocab["word"]
        assert vocab["word"] != vocab["WORD"]


class TestVocabContains:
    """Test membership checking in vocabulary."""

    def test_contains_known_words(self):
        """Test that known words are in token_to_id dict."""
        sentences = [["apple", "banana"]]
        vocab, _ = Vocab.from_sentences(sentences, max_size=100)

        assert "apple" in vocab.token_to_id
        assert "banana" in vocab.token_to_id

    def test_contains_unknown_words(self):
        """Test that unknown words are not in token_to_id dict."""
        sentences = [["apple"]]
        vocab, _ = Vocab.from_sentences(sentences, max_size=100)

        assert "banana" not in vocab.token_to_id
        assert "unknown" not in vocab.token_to_id


class TestVocabLength:
    """Test the __len__ method."""

    def test_len_matches_vocab_size(self):
        """Test that len() returns correct vocabulary size."""
        sentences = [["a", "b", "c"]]
        vocab, _ = Vocab.from_sentences(sentences, max_size=100)

        # Should be 3 words + UNK token
        assert len(vocab) == 4

    def test_len_with_duplicates(self):
        """Test that len() counts unique words only."""
        sentences = [["word", "word", "word"]]
        vocab, _ = Vocab.from_sentences(sentences, max_size=100)

        # Should be 1 word + UNK token
        assert len(vocab) == 2


class TestFromVocab:
    """Test creating vocab from existing word list."""

    def test_from_vocab_basic(self):
        """Test creating vocab from collection of words."""
        words = ["apple", "banana", "cherry"]
        vocab = Vocab.from_vocab(words)

        assert len(vocab) == 4  # 3 words + UNK
        assert "apple" in vocab.token_to_id
        assert "banana" in vocab.token_to_id
        assert "cherry" in vocab.token_to_id
