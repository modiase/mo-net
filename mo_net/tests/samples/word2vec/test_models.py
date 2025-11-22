"""Tests for CBOW and SkipGram models."""

import pytest
import jax
import jax.numpy as jnp
import tempfile
from pathlib import Path

from mo_net.samples.word2vec.softmax_strategy import SoftmaxConfig
from mo_net.samples.word2vec.vocab import Vocab
from mo_net.samples.word2vec import CBOWModel, SkipGramModel


class TestCBOWModelCreation:
    """Test CBOW model creation and initialization."""

    def test_create_basic(self):
        """Test basic CBOW model creation."""
        sentences = [["the", "quick", "brown", "fox"]]
        vocab, _ = Vocab.from_sentences(sentences, max_size=1000)

        model = CBOWModel.create(
            vocab=vocab,
            embedding_dim=10,
            context_size=2,
            softmax_config=SoftmaxConfig.negative_sampling(k=5),
            key=jax.random.PRNGKey(42),
        )

        assert model is not None
        assert model.embeddings.shape == (len(vocab), 10)

    def test_create_with_different_dimensions(self):
        """Test CBOW model with different embedding dimensions."""
        sentences = [["a", "b", "c"]]
        vocab, _ = Vocab.from_sentences(sentences, max_size=1000)

        for dim in [8, 16, 32, 64]:
            model = CBOWModel.create(
                vocab=vocab,
                embedding_dim=dim,
                context_size=1,
                softmax_config=SoftmaxConfig.negative_sampling(k=5),
                key=jax.random.PRNGKey(42),
            )

            assert model.embeddings.shape == (len(vocab), dim)

    def test_create_with_different_context_sizes(self):
        """Test CBOW model with different context window sizes."""
        sentences = [["a", "b", "c", "d", "e"]]
        vocab, _ = Vocab.from_sentences(sentences, max_size=1000)

        for context_size in [1, 2, 3, 5]:
            model = CBOWModel.create(
                vocab=vocab,
                embedding_dim=10,
                context_size=context_size,
                softmax_config=SoftmaxConfig.negative_sampling(k=5),
                key=jax.random.PRNGKey(42),
            )

            # Model should be created successfully
            assert model is not None


class TestCBOWForwardPass:
    """Test CBOW model forward propagation."""

    def test_forward_shape(self):
        """Test that forward pass produces correct output shape."""
        sentences = [["the", "quick", "brown", "fox", "jumps"]]
        vocab, _ = Vocab.from_sentences(sentences, max_size=1000)

        model = CBOWModel.create(
            vocab=vocab,
            embedding_dim=16,
            context_size=2,
            softmax_config=SoftmaxConfig.negative_sampling(k=5),
            key=jax.random.PRNGKey(0),
        )

        # Input: (batch_size, context_size * 2) - indices of context words
        batch_size = 3
        X = jnp.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 1]])

        output = model.forward_prop(X)

        # Output: (batch_size, vocab_size) - logits for target word
        assert output.shape == (batch_size, len(vocab))

    def test_forward_single_sample(self):
        """Test forward pass with single sample."""
        sentences = [["a", "b", "c"]]
        vocab, _ = Vocab.from_sentences(sentences, max_size=1000)

        model = CBOWModel.create(
            vocab=vocab,
            embedding_dim=8,
            context_size=1,
            softmax_config=SoftmaxConfig.negative_sampling(k=5),
            key=jax.random.PRNGKey(0),
        )

        # Single sample: 2 context words (left and right)
        X = jnp.array([[0, 1]])  # shape: (1, 2)

        output = model.forward_prop(X)

        assert output.shape == (1, len(vocab))

    def test_forward_output_range(self):
        """Test that forward pass output is reasonable (not NaN or Inf)."""
        sentences = [["word1", "word2", "word3", "word4"]]
        vocab, _ = Vocab.from_sentences(sentences, max_size=1000)

        model = CBOWModel.create(
            vocab=vocab,
            embedding_dim=10,
            context_size=1,
            softmax_config=SoftmaxConfig.negative_sampling(k=5),
            key=jax.random.PRNGKey(0),
        )

        X = jnp.array([[0, 1]])

        output = model.forward_prop(X)

        # Check for NaN and Inf
        assert not jnp.any(jnp.isnan(output))
        assert not jnp.any(jnp.isinf(output))


class TestCBOWBackwardPass:
    """Test CBOW model backward propagation."""

    def test_backward_runs_without_error(self):
        """Test that backward pass runs without errors."""
        sentences = [["a", "b", "c", "d"]]
        vocab, _ = Vocab.from_sentences(sentences, max_size=1000)

        model = CBOWModel.create(
            vocab=vocab,
            embedding_dim=8,
            context_size=1,
            softmax_config=SoftmaxConfig.negative_sampling(k=5),
            key=jax.random.PRNGKey(0),
        )

        X = jnp.array([[0, 1]])
        Y = jnp.array([2])  # target word

        model.forward_prop(X)
        gradient = model.backward_prop(Y)

        # Should run without errors and return a gradient
        assert gradient is not None
        # Gradient shape varies based on model architecture

    def test_backward_with_batch(self):
        """Test backward pass with batch of samples."""
        sentences = [["a", "b", "c", "d", "e", "f"]]
        vocab, _ = Vocab.from_sentences(sentences, max_size=1000)

        model = CBOWModel.create(
            vocab=vocab,
            embedding_dim=10,
            context_size=1,
            softmax_config=SoftmaxConfig.negative_sampling(k=5),
            key=jax.random.PRNGKey(0),
        )

        batch_size = 4
        X = jnp.array([[0, 1], [1, 2], [2, 3], [3, 4]])
        Y = jnp.array([2, 3, 4, 5])

        model.forward_prop(X)
        gradient = model.backward_prop(Y)

        # Should handle batch processing without errors
        assert gradient is not None


class TestSkipGramModelCreation:
    """Test SkipGram model creation and initialization."""

    def test_create_basic(self):
        """Test basic SkipGram model creation."""
        sentences = [["the", "quick", "brown", "fox"]]
        vocab, _ = Vocab.from_sentences(sentences, max_size=1000)

        model = SkipGramModel.create(
            vocab=vocab,
            embedding_dim=10,
            softmax_config=SoftmaxConfig.negative_sampling(k=5),
            key=jax.random.PRNGKey(42),
            negative_samples=5,
        )

        assert model is not None
        assert model.embeddings.shape == (len(vocab), 10)

    def test_create_with_negative_sampling(self):
        """Test SkipGram model with different negative sampling values."""
        sentences = [["a", "b", "c", "d"]]
        vocab, _ = Vocab.from_sentences(sentences, max_size=1000)

        for n_neg in [0, 1, 5, 10]:
            model = SkipGramModel.create(
                vocab=vocab,
                embedding_dim=8,
                softmax_config=SoftmaxConfig.negative_sampling(k=5),
                key=jax.random.PRNGKey(42),
                negative_samples=n_neg,
            )

            assert model is not None

    def test_create_with_seed(self):
        """Test that models with same seed produce same initialization."""
        sentences = [["a", "b", "c"]]
        vocab, _ = Vocab.from_sentences(sentences, max_size=1000)

        model1 = SkipGramModel.create(
            vocab=vocab,
            embedding_dim=10,
            softmax_config=SoftmaxConfig.negative_sampling(k=5),
            key=jax.random.PRNGKey(42),
            negative_samples=3,
        )

        model2 = SkipGramModel.create(
            vocab=vocab,
            embedding_dim=10,
            softmax_config=SoftmaxConfig.negative_sampling(k=5),
            key=jax.random.PRNGKey(42),
            negative_samples=3,
        )

        # Initial embeddings should be identical
        assert jnp.allclose(model1.embeddings, model2.embeddings)


class TestSkipGramForwardPass:
    """Test SkipGram model forward propagation."""

    def test_forward_shape(self):
        """Test that forward pass produces correct output shape."""
        sentences = [["the", "quick", "brown", "fox", "jumps"]]
        vocab, _ = Vocab.from_sentences(sentences, max_size=1000)

        model = SkipGramModel.create(
            vocab=vocab,
            embedding_dim=16,
            softmax_config=SoftmaxConfig.negative_sampling(k=5),
            key=jax.random.PRNGKey(0),
            negative_samples=5,
        )

        # Input: (batch_size, 1) - index of center word
        batch_size = 3
        X = jnp.array([[1], [2], [3]])

        output = model.forward_prop(X)

        # Output: (batch_size, vocab_size) - logits for context words
        assert output.shape == (batch_size, len(vocab))

    def test_forward_single_sample(self):
        """Test forward pass with single sample."""
        sentences = [["a", "b", "c"]]
        vocab, _ = Vocab.from_sentences(sentences, max_size=1000)

        model = SkipGramModel.create(
            vocab=vocab,
            embedding_dim=8,
            softmax_config=SoftmaxConfig.negative_sampling(k=5),
            key=jax.random.PRNGKey(0),
            negative_samples=2,
        )

        # Single center word
        X = jnp.array([[1]])  # shape: (1, 1)

        output = model.forward_prop(X)

        # Should predict vocabulary distribution
        assert output.shape == (1, len(vocab))

    def test_forward_output_range(self):
        """Test that forward pass output is reasonable (not NaN or Inf)."""
        sentences = [["word1", "word2", "word3", "word4"]]
        vocab, _ = Vocab.from_sentences(sentences, max_size=1000)

        model = SkipGramModel.create(
            vocab=vocab,
            embedding_dim=10,
            softmax_config=SoftmaxConfig.negative_sampling(k=5),
            key=jax.random.PRNGKey(0),
            negative_samples=3,
        )

        X = jnp.array([[1]])

        output = model.forward_prop(X)

        # Check for NaN and Inf
        assert not jnp.any(jnp.isnan(output))
        assert not jnp.any(jnp.isinf(output))


class TestSkipGramBackwardPass:
    """Test SkipGram model backward propagation."""

    def test_backward_runs_without_error(self):
        """Test that backward pass runs without errors."""
        sentences = [["a", "b", "c", "d"]]
        vocab, _ = Vocab.from_sentences(sentences, max_size=1000)

        # Set negative sampling distribution
        model = SkipGramModel.create(
            vocab=vocab,
            embedding_dim=8,
            softmax_config=SoftmaxConfig.negative_sampling(k=5),
            key=jax.random.PRNGKey(0),
            negative_samples=2,
        )
        model._negative_sampling_dist = vocab.get_negative_sampling_distribution()

        X = jnp.array([[1]])  # center word
        Y = jnp.array([[0, 2]])  # context words (batch_size, context_size)

        model.forward_prop(X)
        gradient = model.backward_prop(Y)

        # Should run without errors
        assert gradient is not None

    def test_backward_with_batch(self):
        """Test backward pass with batch."""
        sentences = [["a", "b", "c", "d", "e"]]
        vocab, _ = Vocab.from_sentences(sentences, max_size=1000)

        model = SkipGramModel.create(
            vocab=vocab,
            embedding_dim=10,
            softmax_config=SoftmaxConfig.negative_sampling(k=5),
            key=jax.random.PRNGKey(0),
            negative_samples=3,
        )
        model._negative_sampling_dist = vocab.get_negative_sampling_distribution()

        X = jnp.array([[1], [2]])
        Y = jnp.array([[0, 2], [1, 3]])  # 2D: (batch_size, context_size)

        model.forward_prop(X)
        gradient = model.backward_prop(Y)

        # Should handle batch processing
        assert gradient is not None


class TestModelSerialization:
    """Test model serialization and deserialization."""

    def test_cbow_save_load(self):
        """Test CBOW model save and load."""
        sentences = [["a", "b", "c", "d"]]
        vocab, _ = Vocab.from_sentences(sentences, max_size=1000)

        model = CBOWModel.create(
            vocab=vocab,
            embedding_dim=10,
            context_size=1,
            softmax_config=SoftmaxConfig.negative_sampling(k=5),
            key=jax.random.PRNGKey(42),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model"
            model.dump(model_path)

            # Load model
            loaded_model = CBOWModel.load(model_path)

            # Embeddings should be identical
            assert jnp.allclose(model.embeddings, loaded_model.embeddings)

    def test_skipgram_save_load(self):
        """Test SkipGram model save and load."""
        sentences = [["a", "b", "c", "d"]]
        vocab, _ = Vocab.from_sentences(sentences, max_size=1000)

        model = SkipGramModel.create(
            vocab=vocab,
            embedding_dim=10,
            softmax_config=SoftmaxConfig.negative_sampling(k=5),
            key=jax.random.PRNGKey(42),
            negative_samples=3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model"
            model.dump(model_path)

            # Load model
            loaded_model = SkipGramModel.load(model_path)

            # Embeddings should be identical
            assert jnp.allclose(model.embeddings, loaded_model.embeddings)

    def test_save_load_preserves_functionality(self):
        """Test that save/load preserves model functionality."""
        sentences = [["a", "b", "c", "d", "e"]]
        vocab, _ = Vocab.from_sentences(sentences, max_size=1000)

        original_model = SkipGramModel.create(
            vocab=vocab,
            embedding_dim=12,
            softmax_config=SoftmaxConfig.negative_sampling(k=5),
            key=jax.random.PRNGKey(42),
            negative_samples=2,
        )

        # Get output from original model
        X = jnp.array([[2]])
        output_original = original_model.forward_prop(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model"
            original_model.dump(model_path)

            loaded_model = SkipGramModel.load(model_path)

            # Get output from loaded model
            output_loaded = loaded_model.forward_prop(X)

            # Outputs should be identical
            assert jnp.allclose(output_original, output_loaded, rtol=1e-5)


class TestModelEmbeddings:
    """Test model embedding access and properties."""

    def test_cbow_embeddings_access(self):
        """Test accessing embeddings from CBOW model."""
        sentences = [["a", "b", "c"]]
        vocab, _ = Vocab.from_sentences(sentences, max_size=1000)

        model = CBOWModel.create(
            vocab=vocab,
            embedding_dim=10,
            context_size=1,
            softmax_config=SoftmaxConfig.negative_sampling(k=5),
            key=jax.random.PRNGKey(0),
        )

        embeddings = model.embeddings

        # Should have correct shape
        assert embeddings.shape == (len(vocab), 10)

        # Each word should have an embedding
        for i in range(len(vocab)):
            assert embeddings[i].shape == (10,)

    def test_skipgram_embeddings_access(self):
        """Test accessing embeddings from SkipGram model."""
        sentences = [["a", "b", "c"]]
        vocab, _ = Vocab.from_sentences(sentences, max_size=1000)

        model = SkipGramModel.create(
            vocab=vocab,
            embedding_dim=16,
            softmax_config=SoftmaxConfig.negative_sampling(k=5),
            key=jax.random.PRNGKey(0),
            negative_samples=2,
        )

        embeddings = model.embeddings

        # Should have correct shape
        assert embeddings.shape == (len(vocab), 16)


class TestModelEdgeCases:
    """Test edge cases and error handling."""

    def test_model_with_small_vocab(self):
        """Test models work with very small vocabulary."""
        sentences = [["a", "b"]]  # Only 2 words + UNK
        vocab, _ = Vocab.from_sentences(sentences, max_size=1000)

        # CBOW
        cbow = CBOWModel.create(
            vocab=vocab,
            embedding_dim=4,
            context_size=1,
            softmax_config=SoftmaxConfig.negative_sampling(k=5),
            key=jax.random.PRNGKey(0),
        )
        assert cbow is not None

        # SkipGram
        skipgram = SkipGramModel.create(
            vocab=vocab,
            embedding_dim=4,
            softmax_config=SoftmaxConfig.negative_sampling(k=5),
            key=jax.random.PRNGKey(0),
            negative_samples=2,
        )
        assert skipgram is not None

    def test_model_with_large_context_size(self):
        """Test CBOW models with large context windows."""
        sentences = [["a", "b", "c", "d", "e", "f", "g", "h"]]
        vocab, _ = Vocab.from_sentences(sentences, max_size=1000)

        model = CBOWModel.create(
            vocab=vocab,
            embedding_dim=8,
            context_size=5,  # Large context
            softmax_config=SoftmaxConfig.negative_sampling(k=5),
            key=jax.random.PRNGKey(0),
        )

        assert model is not None

    def test_model_with_small_embedding_dim(self):
        """Test models with very small embedding dimensions."""
        sentences = [["a", "b", "c"]]
        vocab, _ = Vocab.from_sentences(sentences, max_size=1000)

        model = SkipGramModel.create(
            vocab=vocab,
            embedding_dim=2,  # Very small
            softmax_config=SoftmaxConfig.negative_sampling(k=5),
            key=jax.random.PRNGKey(0),
            negative_samples=1,
        )

        assert model is not None
        assert model.embeddings.shape[1] == 2


class TestHierarchicalSoftmaxModels:
    """Test model creation with hierarchical softmax."""

    def test_cbow_with_hierarchical_softmax(self):
        """Test CBOW model with hierarchical softmax."""
        sentences = [["the", "quick", "brown", "fox", "jumps"]]
        vocab, _ = Vocab.from_sentences(sentences, max_size=1000)

        model = CBOWModel.create(
            vocab=vocab,
            embedding_dim=16,
            context_size=2,
            softmax_config=SoftmaxConfig.hierarchical_softmax(),
            key=jax.random.PRNGKey(42),
        )

        assert model is not None
        assert model.embeddings.shape == (len(vocab), 16)

        # Verify output layer is hierarchical
        from mo_net.model.layer.output import HierarchicalSoftmaxOutputLayer

        assert isinstance(model.output.output_layer, HierarchicalSoftmaxOutputLayer)

        # Verify tree is built
        assert model.output.output_layer.vocab_size == len(vocab)

    def test_skipgram_with_hierarchical_softmax(self):
        """Test SkipGram model with hierarchical softmax."""
        sentences = [["a", "b", "c", "d", "e"]]
        vocab, _ = Vocab.from_sentences(sentences, max_size=1000)

        model = SkipGramModel.create(
            vocab=vocab,
            embedding_dim=12,
            softmax_config=SoftmaxConfig.hierarchical_softmax(),
            key=jax.random.PRNGKey(42),
            negative_samples=5,  # Not used for hierarchical
        )

        assert model is not None
        assert model.embeddings.shape == (len(vocab), 12)

        # Verify output layer is hierarchical
        from mo_net.model.layer.output import HierarchicalSoftmaxOutputLayer

        assert isinstance(model.output.output_layer, HierarchicalSoftmaxOutputLayer)

    def test_hierarchical_requires_vocab(self):
        """Test that hierarchical softmax requires vocab parameter."""
        with pytest.raises(ValueError, match="vocab is required"):
            CBOWModel.create(
                vocab_size=100,  # Only vocab_size, no vocab
                embedding_dim=10,
                context_size=2,
                softmax_config=SoftmaxConfig.hierarchical_softmax(),
                key=jax.random.PRNGKey(42),
            )

    def test_hierarchical_forward_backward(self):
        """Test forward and backward passes work with hierarchical softmax."""
        sentences = [["a", "b", "c", "d"]]
        vocab, _ = Vocab.from_sentences(sentences, max_size=1000)

        model = CBOWModel.create(
            vocab=vocab,
            embedding_dim=8,
            context_size=1,
            softmax_config=SoftmaxConfig.hierarchical_softmax(),
            key=jax.random.PRNGKey(0),
        )

        # Forward pass
        X = jnp.array([[0, 1]])
        output = model.forward_prop(X)

        assert output.shape == (1, len(vocab))
        assert not jnp.any(jnp.isnan(output))

        # Backward pass
        Y = jnp.array([2])
        gradient = model.backward_prop(Y)

        assert gradient is not None

    def test_hierarchical_no_linear_layer(self):
        """Test that hierarchical softmax models don't have Linear layer."""
        sentences = [["a", "b", "c"]]
        vocab, _ = Vocab.from_sentences(sentences, max_size=1000)

        model = CBOWModel.create(
            vocab=vocab,
            embedding_dim=8,
            context_size=1,
            softmax_config=SoftmaxConfig.hierarchical_softmax(),
            key=jax.random.PRNGKey(42),
        )

        # Output module should have no layers (tree replaces Linear layer)
        assert len(model.output.layers) == 0
