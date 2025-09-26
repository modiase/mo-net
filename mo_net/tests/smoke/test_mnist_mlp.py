"""
Smoke test for MNIST MLP training.
Tests basic functionality with minimal dimensions for speed.
"""

import tempfile
from pathlib import Path

import jax
import pytest

from mo_net.data import SplitConfig, load_data
from mo_net.functions import cross_entropy, get_activation_fn
from mo_net.model.model import Model
from mo_net.protos import NormalisationType
from mo_net.resources import MNIST_TRAIN_URL
from mo_net.train import TrainingParameters
from mo_net.train.backends.log import NullBackend
from mo_net.train.run import TrainingRun
from mo_net.train.trainer.trainer import BasicTrainer, TrainingSuccessful, get_optimiser


@pytest.mark.smoke
def test_mnist_mlp_training():
    """Test MLP training with minimal dimensions for speed."""
    with tempfile.TemporaryDirectory() as temp_dir:
        X_train, Y_train, _, __ = load_data(
            MNIST_TRAIN_URL,
            split=SplitConfig.of(0.1, 0),
            one_hot=True,
        )

        X_train = X_train.reshape(-1, 28 * 28)

        model = Model.mlp_of(
            key=jax.random.PRNGKey(42),
            module_dimensions=[(784,), (32,), (10,)],
            activation_fn=get_activation_fn("relu"),
            normalisation_type=NormalisationType.NONE,
            tracing_enabled=False,
        )

        training_parameters = TrainingParameters(
            batch_size=32,
            dropout_keep_probs=(),
            history_max_len=10,
            learning_rate_limits=(1e-3, 1e-3),
            log_level="ERROR",
            max_restarts=0,
            monotonic=False,
            no_monitoring=True,
            normalisation_type=NormalisationType.NONE,
            num_epochs=1,
            quiet=True,
            regulariser_lambda=1e-4,
            seed=42,
            trace_logging=False,
            train_set_size=len(X_train),
            warmup_epochs=0,
            workers=0,
        )

        train_size = int(0.8 * len(X_train))

        run = TrainingRun(seed=42, name="smoke_test_mlp_42", backend=NullBackend())

        trainer = BasicTrainer(
            X_train=X_train[:train_size],
            X_val=X_train[train_size:],
            Y_train=Y_train[:train_size],
            Y_val=Y_train[train_size:],
            key=jax.random.PRNGKey(42),
            transform_fn=None,
            loss_fn=cross_entropy,
            model=model,
            optimiser=get_optimiser("adam", model, training_parameters),
            run=run,
            training_parameters=training_parameters,
            output_path=Path(temp_dir) / "model.pkl",
        )

        result = trainer.train()
        assert isinstance(result, TrainingSuccessful)
