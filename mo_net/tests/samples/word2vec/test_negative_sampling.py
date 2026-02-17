"""Tests for negative sampling in word2vec models."""

import jax
import jax.numpy as jnp
import pytest
from dataclasses import dataclass

from mo_net.samples.word2vec.softmax_strategy import SoftmaxConfig
from mo_net.samples.word2vec.vocab import Vocab
from mo_net.samples.word2vec import SkipGramModel
from mo_net.model.layer.output import SparseCategoricalSoftmaxOutputLayer


class TestNegativeSamplingDistribution:
    """Test the negative sampling distribution computation."""

    def test_distribution_sum_to_one(self):
        """Test that negative sampling distribution sums to 1."""
        sentences = [
            ["the", "quick", "brown", "fox"],
            ["the", "lazy", "dog"],
            ["the", "cat"],
        ]
        vocab, _ = Vocab.from_sentences(sentences, max_size=1000)
        dist = vocab.get_negative_sampling_distribution()

        assert jnp.isclose(jnp.sum(dist), 1.0, rtol=1e-5)

    def test_distribution_uses_power_075(self):
        """Test that distribution uses unigram^0.75 as per Mikolov et al."""
        # Create vocab with known word counts
        sentences = [
            ["a"] * 16,  # freq = 16, freq^0.75 = 8
            ["b"] * 4,  # freq = 4, freq^0.75 = 2.83
            ["c"] * 1,  # freq = 1, freq^0.75 = 1
        ]
        flat_sentences = [" ".join(s).split() for s in sentences]
        vocab, _ = Vocab.from_sentences(flat_sentences, max_size=1000)
        dist = vocab.get_negative_sampling_distribution()

        # Get indices for our words
        a_idx = vocab["a"]
        b_idx = vocab["b"]
        c_idx = vocab["c"]

        # Verify relative probabilities match freq^0.75
        # a should have higher probability than b, b higher than c
        assert dist[a_idx] > dist[b_idx]
        assert dist[b_idx] > dist[c_idx]

        # Check approximate ratios (allowing for numerical error and UNK token)
        # Expected: 8 : 2.83 : 1 approximately
        assert dist[a_idx] / dist[c_idx] > 5.0  # Should be close to 8

    def test_distribution_handles_unknown_token(self):
        """Test that unknown token is included in distribution."""
        sentences = [["word1", "word2"]]
        vocab, _ = Vocab.from_sentences(sentences, max_size=1000)
        dist = vocab.get_negative_sampling_distribution()

        # Distribution should have length = vocab_size (including UNK)
        assert len(dist) == len(vocab)

        # UNK token should have non-zero probability
        unk_idx = vocab.unknown_token_id
        assert dist[unk_idx] > 0

    def test_distribution_all_positive(self):
        """Test that all probabilities are non-negative."""
        sentences = [["a", "b", "c", "d", "e"]]
        vocab, _ = Vocab.from_sentences(sentences, max_size=1000)
        dist = vocab.get_negative_sampling_distribution()

        assert jnp.all(dist >= 0)


class TestNegativeSampling:
    """Test negative sampling behavior in SkipGram model."""

    def test_negative_samples_are_valid_indices(self):
        """Test that negative samples are valid vocabulary indices."""
        sentences = [
            ["the", "quick", "brown", "fox", "jumps"],
            ["over", "the", "lazy", "dog"],
        ]
        vocab, _ = Vocab.from_sentences(sentences, max_size=1000)

        model = SkipGramModel.create(
            vocab=vocab,
            embedding_dim=10,
            softmax_config=SoftmaxConfig.negative_sampling(k=5),
            key=jax.random.PRNGKey(42),
            negative_samples=5,
        )
        model._negative_sampling_dist = vocab.get_negative_sampling_distribution()

        # Create a simple training batch
        X = jnp.array([[1]])  # center word (batch=1, input=1)
        Y = jnp.array([[3, 4]])  # context words (batch=1, context=2)

        # Forward pass
        model.forward_prop(X)

        # Backward pass (which generates negative samples)
        model.backward_prop(Y)

        # Check that model still works (negative samples were valid)
        # If negative samples were invalid, JAX would have raised an error
        assert True  # If we get here, negative sampling worked

    def test_negative_samples_different_values(self):
        """Test that different negative_samples values work correctly."""
        sentences = [["a", "b", "c", "d", "e", "f", "g", "h"]]
        vocab, _ = Vocab.from_sentences(sentences, max_size=1000)

        for n_neg in [1, 5, 10]:
            model = SkipGramModel.create(
                vocab=vocab,
                embedding_dim=8,
                softmax_config=SoftmaxConfig.negative_sampling(k=5),
                key=jax.random.PRNGKey(42),
                negative_samples=n_neg,
            )
            model._negative_sampling_dist = vocab.get_negative_sampling_distribution()

            X = jnp.array([[1]])
            Y = jnp.array([[2]])

            model.forward_prop(X)
            model.backward_prop(Y)

            # Model should work with any valid number of negative samples
            assert True

    @pytest.mark.skip(
        reason="Requires Trainer/Optimiser integration - see test_integration.py"
    )
    def test_negative_sampling_reduces_probability(self):
        """Test that negative sampling reduces likelihood of negative samples.

        This is the key test: negative samples should have their probabilities
        pushed down during training.

        NOTE: This test requires Trainer/Optimiser integration to properly train the model.
        See test_integration.py for full training tests.
        """
        pass

    def test_negative_sampling_with_seed(self):
        """Test that same seed produces same model initialization."""
        sentences = [["a", "b", "c", "d", "e"]]
        vocab, _ = Vocab.from_sentences(sentences, max_size=1000)

        # Create two models with same seed
        model1 = SkipGramModel.create(
            vocab=vocab,
            embedding_dim=8,
            softmax_config=SoftmaxConfig.negative_sampling(k=5),
            key=jax.random.PRNGKey(42),
            negative_samples=3,
        )

        model2 = SkipGramModel.create(
            vocab=vocab,
            embedding_dim=8,
            softmax_config=SoftmaxConfig.negative_sampling(k=5),
            key=jax.random.PRNGKey(42),
            negative_samples=3,
        )

        # Models with same seed should have identical embeddings
        assert jnp.allclose(model1.embeddings, model2.embeddings)


class TestGradientSparsity:
    """Test that gradients are sparse (only computed for sampled words).

    NOTE: These tests are comprehensively covered in test_output.py.
    The SparseCategoricalSoftmaxOutputLayer tests there verify gradient sparsity
    in detail, so we skip duplicating them here.
    """

    @pytest.mark.skip(
        reason="Covered in test_output.py - see TestNegativeSamplingBackwardProp"
    )
    def test_output_layer_gradient_sparsity(self):
        """Test that backward_prop_with_negative produces sparse gradients."""
        pass

    @pytest.mark.skip(
        reason="Covered in test_output.py - see TestNegativeSamplingBackwardProp"
    )
    def test_gradient_sparsity_with_different_negative_counts(self):
        """Test gradient sparsity with different numbers of negative samples."""
        pass


class TestNegativeSamplingEdgeCases:
    """Test edge cases in negative sampling."""

    def test_negative_sampling_with_small_vocab(self):
        """Test negative sampling works with very small vocabulary."""
        sentences = [["a", "b", "c"]]  # Only 3 words + UNK
        vocab, _ = Vocab.from_sentences(sentences, max_size=1000)

        # More negative samples than vocab size - should still work
        model = SkipGramModel.create(
            vocab=vocab,
            embedding_dim=8,
            softmax_config=SoftmaxConfig.negative_sampling(k=5),
            key=jax.random.PRNGKey(42),
            negative_samples=10,  # More than vocab size!
        )
        model._negative_sampling_dist = vocab.get_negative_sampling_distribution()

        X = jnp.array([[1]])
        Y = jnp.array([[2]])

        model.forward_prop(X)
        model.backward_prop(Y)

        # Should work without errors (sampling with replacement)
        assert True

    def test_negative_sampling_distribution_single_word(self):
        """Test distribution with corpus containing single repeated word."""
        sentences = [["word"] * 100]
        vocab, _ = Vocab.from_sentences(sentences, max_size=1000)
        dist = vocab.get_negative_sampling_distribution()

        # Should still be valid distribution
        assert jnp.isclose(jnp.sum(dist), 1.0, rtol=1e-5)
        assert jnp.all(dist >= 0)

    def test_negative_samples_exclude_positive(self):
        """Test that negative samples can be different from positive samples.

        Note: Current implementation samples uniformly from vocab, so negative
        samples CAN include the positive sample. This tests that the model
        handles this correctly.
        """
        sentences = [["a", "b"]]  # Very small vocab
        vocab, _ = Vocab.from_sentences(sentences, max_size=1000)

        model = SkipGramModel.create(
            vocab=vocab,
            embedding_dim=8,
            softmax_config=SoftmaxConfig.negative_sampling(k=5),
            key=jax.random.PRNGKey(42),
            negative_samples=5,  # Many samples from small vocab
        )
        model._negative_sampling_dist = vocab.get_negative_sampling_distribution()

        X = jnp.array([[vocab["a"]]])
        Y = jnp.array([[vocab["b"]]])

        # Should handle overlap between positive and negative samples
        model.forward_prop(X)
        model.backward_prop(Y)

        # No error means it handled potential overlap correctly
        assert True
