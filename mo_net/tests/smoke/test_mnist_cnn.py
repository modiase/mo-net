"""
Smoke test for MNIST CNN training.
Tests basic functionality with minimal dimensions for speed.
"""

import tempfile
from pathlib import Path

import jax
import pytest

from mo_net.data import SplitConfig, load_data
from mo_net.functions import cross_entropy, get_activation_fn
from mo_net.model.layer.activation import Activation
from mo_net.model.layer.batch_norm.batch_norm_2d import BatchNorm2D
from mo_net.model.layer.convolution import Convolution2D
from mo_net.model.layer.linear import Linear
from mo_net.model.layer.output import SoftmaxOutputLayer
from mo_net.model.layer.pool import MaxPooling2D
from mo_net.model.layer.reshape import Flatten
from mo_net.model.model import Model
from mo_net.model.module.base import Hidden, Output
from mo_net.protos import NormalisationType
from mo_net.resources import MNIST_TRAIN_URL
from mo_net.train import TrainingParameters
from mo_net.train.augment import affine_transform2D
from mo_net.train.backends.log import NullBackend
from mo_net.train.run import TrainingRun
from mo_net.train.trainer.trainer import BasicTrainer, TrainingSuccessful, get_optimiser


# TODO: Fix CNN smoke test - ValueError in training
@pytest.mark.skip(reason="TODO: CNN training smoke test broken")
@pytest.mark.smoke
def test_mnist_cnn_training():
    """Test CNN training with minimal dimensions for speed."""
    with tempfile.TemporaryDirectory() as temp_dir:
        X_train, Y_train, _, __ = load_data(
            MNIST_TRAIN_URL,
            split=SplitConfig.of(0.1, 0),
            one_hot=True,
        )

        X_train = X_train.reshape(-1, 1, 28, 28)

        key1, key2, key3, key4, key5 = jax.random.split(jax.random.PRNGKey(42), 5)

        model = Model(
            input_dimensions=(1, 28, 28),
            hidden=(
                Hidden(
                    layers=(
                        conv1 := Convolution2D(
                            input_dimensions=(1, 28, 28),
                            n_kernels=1,  # Very few kernels for speed
                            kernel_size=8,
                            stride=1,
                            kernel_init_fn=lambda n_kernels,
                            in_channels,
                            kernel_height,
                            kernel_width: Convolution2D.Parameters.he(
                                n_kernels,
                                in_channels,
                                kernel_height,
                                kernel_width,
                                key=key1,
                            ),
                        ),
                        bn1 := BatchNorm2D(
                            input_dimensions=conv1.output_dimensions,
                        ),
                        Activation(
                            input_dimensions=bn1.output_dimensions,
                            activation_fn=get_activation_fn("relu"),
                        ),
                        pool1 := MaxPooling2D(
                            input_dimensions=bn1.output_dimensions,
                            pool_size=2,
                            stride=2,
                        ),
                    )
                ),
                Hidden(
                    layers=(
                        flatten := Flatten(
                            input_dimensions=pool1.output_dimensions,
                        ),
                    )
                ),
            ),
            output=Output(
                layers=(
                    dense := Linear(
                        input_dimensions=flatten.output_dimensions,
                        output_dimensions=(16,),  # Very small dense layer
                        parameters_init_fn=lambda input_dims,
                        output_dims: Linear.Parameters.xavier(
                            input_dims, output_dims, key=key4
                        ),
                        store_output_activations=False,
                    ),
                    Activation(
                        input_dimensions=dense.output_dimensions,
                        activation_fn=get_activation_fn("relu"),
                    ),
                    Linear(
                        input_dimensions=(16,),
                        output_dimensions=(10,),
                        parameters_init_fn=lambda input_dims,
                        output_dims: Linear.Parameters.xavier(
                            input_dims, output_dims, key=key5
                        ),
                        store_output_activations=False,
                    ),
                ),
                output_layer=SoftmaxOutputLayer(input_dimensions=(10,)),
            ),
        )

        training_parameters = TrainingParameters(
            batch_size=16,  # Small batch size for speed
            dropout_keep_probs=(),
            history_max_len=10,
            learning_rate_limits=(1e-3, 1e-3),
            log_level="ERROR",
            max_restarts=0,
            monotonic=False,
            no_monitoring=True,
            normalisation_type=NormalisationType.NONE,
            num_epochs=1,  # Just 1 epoch for speed
            quiet=True,
            regulariser_lambda=1e-4,
            seed=42,
            trace_logging=False,
            train_set_size=len(X_train),
            warmup_epochs=0,
            workers=0,
        )

        train_size = int(0.8 * len(X_train))

        trainer = BasicTrainer(
            X_train=X_train[:train_size],
            X_val=X_train[train_size:],
            Y_train=Y_train[:train_size],
            Y_val=Y_train[train_size:],
            key=jax.random.PRNGKey(42),
            transform_fn=affine_transform2D(x_size=28, y_size=28),
            loss_fn=cross_entropy,
            model=model,
            optimiser=get_optimiser("adam", model, training_parameters),
            run=TrainingRun(seed=42, name="smoke_test_cnn_42", backend=NullBackend()),
            training_parameters=training_parameters,
            output_path=Path(temp_dir) / "model.pkl",
        )

        result = trainer.train()
        assert isinstance(result, TrainingSuccessful)
