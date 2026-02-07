"""Integration tests for word2vec training with BasicTrainer."""

import pytest
import jax
import jax.numpy as jnp
from pathlib import Path

from mo_net.samples.word2vec.softmax_strategy import SoftmaxConfig
from mo_net.samples.word2vec.vocab import Vocab
from mo_net.samples.word2vec import CBOWModel, SkipGramModel
from mo_net.train import TrainingParameters
from mo_net.train.backends.log import SqliteBackend
from mo_net.train.run import TrainingRun
from mo_net.train.trainer.trainer import BasicTrainer, get_optimiser, TrainingSuccessful
from mo_net.functions import sparse_cross_entropy
from mo_net.protos import NormalisationType


class TestCBOWIntegration:
    """Test CBOW model with BasicTrainer."""

    def test_cbow_basic_training(self):
        """Test that CBOW can train for a few steps without errors."""
        # Create a small corpus
        sentences = [
            ["the", "quick", "brown", "fox"],
            ["the", "lazy", "dog"],
            ["the", "cat", "sat"],
            ["quick", "brown", "fox", "jumps"],
        ] * 20  # Repeat for more training data

        vocab, tokenized = Vocab.from_sentences(sentences, max_size=1000)

        # Create model
        model = CBOWModel.create(
            vocab=vocab,
            embedding_dim=16,
            context_size=1,
            softmax_config=SoftmaxConfig.negative_sampling(k=5),
            key=jax.random.PRNGKey(42),
        )

        # Prepare training data - context windows
        from mo_net.samples.word2vec.__main__ import all_windows

        windows = list(all_windows(tokenized, window_size=3))
        X_train = jnp.array([[w[0], w[2]] for w in windows])  # [left, right] context
        Y_train = jnp.array([w[1] for w in windows])  # center word

        # Split train/val
        train_size = int(0.8 * len(X_train))
        X_train_split = X_train[:train_size]
        Y_train_split = Y_train[:train_size]
        X_val = X_train[train_size:]
        Y_val = Y_train[train_size:]

        # Training parameters - just 2 epochs
        training_parameters = TrainingParameters(
            batch_size=8,
            num_epochs=2,
            quiet=True,
            train_set_size=len(X_train_split),
            no_monitoring=True,
            dropout_keep_probs=(),
            history_max_len=100,
            learning_rate_limits=(0.001, 0.001),
            log_level="INFO",
            max_restarts=0,
            monotonic=False,
            normalisation_type=NormalisationType.NONE,
            regulariser_lambda=0.0,
            seed=42,
            trace_logging=False,
            warmup_epochs=0,
            workers=0,
        )

        # Create run and optimiser
        run = TrainingRun(seed=42, name="test_cbow", backend=SqliteBackend())
        optimiser = get_optimiser("adam", model, training_parameters)

        # Create trainer
        trainer = BasicTrainer(
            X_train=X_train_split,
            Y_train=Y_train_split,
            X_val=X_val,
            Y_val=Y_val,
            model=model,
            optimiser=optimiser,
            run=run,
            training_parameters=training_parameters,
            loss_fn=sparse_cross_entropy,
            key=jax.random.PRNGKey(42),
            disable_shutdown=True,
        )

        # Train
        result = trainer.train()

        # Should complete successfully
        assert isinstance(result, TrainingSuccessful)

        # Model should have embeddings
        assert model.embeddings.shape == (len(vocab), 16)

        # Clean up checkpoint
        result.model_checkpoint_path.unlink(missing_ok=True)


class TestSkipGramIntegration:
    """Test SkipGram model with BasicTrainer."""

    def test_skipgram_basic_training(self):
        """Test that SkipGram can train for a few steps without errors."""
        # Create a small corpus
        sentences = [
            ["dog", "barks", "loudly"],
            ["cat", "meows", "softly"],
            ["bird", "chirps", "happily"],
        ] * 30  # Repeat for more training data

        vocab, tokenized = Vocab.from_sentences(sentences, max_size=1000)

        # Create model
        model = SkipGramModel.create(
            vocab=vocab,
            embedding_dim=16,
            softmax_config=SoftmaxConfig.negative_sampling(k=5),
            key=jax.random.PRNGKey(42),
            negative_samples=3,
        )
        model._negative_sampling_dist = vocab.get_negative_sampling_distribution()

        # Prepare training data - center word predicts context
        from mo_net.samples.word2vec.__main__ import all_windows

        windows = list(all_windows(tokenized, window_size=3))
        X_train = jnp.array(
            [[w[1]] for w in windows]
        )  # center word (reshape for skipgram)
        Y_train = jnp.array([[w[0], w[2]] for w in windows])  # [left, right] context

        # Split train/val
        train_size = int(0.8 * len(X_train))
        X_train_split = X_train[:train_size]
        Y_train_split = Y_train[:train_size]
        X_val = X_train[train_size:]
        Y_val = Y_train[train_size:]

        # Training parameters - just 2 epochs
        training_parameters = TrainingParameters(
            batch_size=8,
            num_epochs=2,
            quiet=True,
            train_set_size=len(X_train_split),
            no_monitoring=True,
            dropout_keep_probs=(),
            history_max_len=100,
            learning_rate_limits=(0.001, 0.001),
            log_level="INFO",
            max_restarts=0,
            monotonic=False,
            normalisation_type=NormalisationType.NONE,
            regulariser_lambda=0.0,
            seed=42,
            trace_logging=False,
            warmup_epochs=0,
            workers=0,
        )

        # Create run and optimiser
        run = TrainingRun(seed=42, name="test_skipgram", backend=SqliteBackend())
        optimiser = get_optimiser("adam", model, training_parameters)

        # Create trainer
        trainer = BasicTrainer(
            X_train=X_train_split,
            Y_train=Y_train_split,
            X_val=X_val,
            Y_val=Y_val,
            model=model,
            optimiser=optimiser,
            run=run,
            training_parameters=training_parameters,
            loss_fn=sparse_cross_entropy,
            key=jax.random.PRNGKey(42),
            disable_shutdown=True,
        )

        # Train
        result = trainer.train()

        # Should complete successfully
        assert isinstance(result, TrainingSuccessful)

        # Model should have embeddings
        assert model.embeddings.shape == (len(vocab), 16)

        # Clean up checkpoint
        result.model_checkpoint_path.unlink(missing_ok=True)


class TestModelSaveLoad:
    """Test that models can be saved and loaded during training."""

    def test_skipgram_save_load_cycle(self):
        """Test saving and loading a trained SkipGram model."""
        import tempfile

        sentences = [["a", "b", "c", "d", "e"]] * 5
        vocab, _ = Vocab.from_sentences(sentences, max_size=1000)

        model = SkipGramModel.create(
            vocab=vocab,
            embedding_dim=10,
            softmax_config=SoftmaxConfig.negative_sampling(k=5),
            key=jax.random.PRNGKey(42),
            negative_samples=2,
        )

        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"
            model.dump(model_path)

            # Load
            loaded_model = SkipGramModel.load(model_path)

            # Embeddings should match
            assert jnp.allclose(model.embeddings, loaded_model.embeddings)
