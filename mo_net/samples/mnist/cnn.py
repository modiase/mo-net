from __future__ import annotations

import functools
import time
from pathlib import Path
from typing import Callable, ParamSpec, TypeVar, assert_never

import click
import jax
from loguru import logger

from mo_net import print_device_info
from mo_net.data import DATA_DIR
from mo_net.functions import get_activation_fn, sparse_cross_entropy
from mo_net.log import LogLevel, setup_logging
from mo_net.model.layer.activation import Activation
from mo_net.model.layer.batch_norm.batch_norm_2d import BatchNorm2D
from mo_net.model.layer.convolution import Convolution2D
from mo_net.model.layer.linear import Linear
from mo_net.model.layer.output import SparseCategoricalSoftmaxOutputLayer
from mo_net.model.layer.pool import MaxPooling2D
from mo_net.model.layer.reshape import Flatten
from mo_net.model.model import Model
from mo_net.model.module.base import Hidden, Output
from mo_net.protos import NormalisationType
from mo_net.resources import MNIST_TRAIN_URL
from mo_net.train import TrainingParameters
from mo_net.train.augment import affine_transform2D
from mo_net.train.backends.log import SqliteBackend
from mo_net.train.run import TrainingRun
from mo_net.train.trainer.trainer import (
    BasicTrainer,
    TrainingFailed,
    TrainingSuccessful,
    get_optimiser,
)

P = ParamSpec("P")
R = TypeVar("R")

N_DIGITS: int = 10
MNIST_IMAGE_SIZE: int = 28


class CNNModel(Model):
    @classmethod
    def get_name(cls) -> str:
        return "cnn"

    @classmethod
    def get_description(cls) -> str:
        return "Convolutional Neural Network for MNIST"

    @classmethod
    def create(
        cls,
        *,
        key: jax.Array,
        tracing_enabled: bool = False,
    ) -> CNNModel:
        """
        Create a CNN model for MNIST with 3 convolution modules followed by a dense layer.

        Architecture:
        - Input: (1, 28, 28) - MNIST grayscale images
        - Conv1: 32 kernels, 3x3, ReLU, BatchNorm, MaxPool 2x2
        - Conv2: 64 kernels, 3x3, ReLU, BatchNorm, MaxPool 2x2
        - Conv3: 128 kernels, 3x3, ReLU, BatchNorm, MaxPool 2x2
        - Flatten
        - Dense: 512 units, ReLU
        - Output: 10 units (softmax)
        """
        key1, key2, key3, key4, key5 = jax.random.split(key, 5)
        return cls(
            input_dimensions=(1, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE),
            hidden=(
                Hidden(
                    layers=(
                        conv1 := Convolution2D(
                            input_dimensions=(1, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE),
                            n_kernels=32,
                            kernel_size=3,
                            stride=1,
                            kernel_init_fn=functools.partial(
                                Convolution2D.Parameters.he, key=key1
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
                        conv2 := Convolution2D(
                            input_dimensions=pool1.output_dimensions,
                            n_kernels=64,
                            kernel_size=3,
                            stride=1,
                            kernel_init_fn=functools.partial(
                                Convolution2D.Parameters.he, key=key2
                            ),
                        ),
                        bn2 := BatchNorm2D(
                            input_dimensions=conv2.output_dimensions,
                        ),
                        Activation(
                            input_dimensions=bn2.output_dimensions,
                            activation_fn=get_activation_fn("relu"),
                        ),
                        pool2 := MaxPooling2D(
                            input_dimensions=bn2.output_dimensions,
                            pool_size=2,
                            stride=2,
                        ),
                    )
                ),
                Hidden(
                    layers=(
                        conv3 := Convolution2D(
                            input_dimensions=pool2.output_dimensions,
                            n_kernels=128,
                            kernel_size=3,
                            stride=1,
                            kernel_init_fn=functools.partial(
                                Convolution2D.Parameters.he, key=key3
                            ),
                        ),
                        bn3 := BatchNorm2D(
                            input_dimensions=conv3.output_dimensions,
                        ),
                        Activation(
                            input_dimensions=bn3.output_dimensions,
                            activation_fn=get_activation_fn("relu"),
                        ),
                        pool3 := MaxPooling2D(
                            input_dimensions=bn3.output_dimensions,
                            pool_size=2,
                            stride=2,
                        ),
                    )
                ),
                Hidden(
                    layers=(
                        flatten := Flatten(
                            input_dimensions=pool3.output_dimensions,
                        ),
                    )
                ),
            ),
            output=Output(
                layers=(
                    dense := Linear(
                        input_dimensions=flatten.output_dimensions,
                        output_dimensions=(512,),
                        parameters_init_fn=functools.partial(
                            Linear.Parameters.xavier,
                            key=key4,
                        ),
                        store_output_activations=tracing_enabled,
                    ),
                    Activation(
                        input_dimensions=dense.output_dimensions,
                        activation_fn=get_activation_fn("relu"),
                    ),
                    Linear(
                        input_dimensions=(512,),
                        output_dimensions=(N_DIGITS,),
                        parameters_init_fn=functools.partial(
                            Linear.Parameters.xavier,
                            key=key5,
                        ),
                        store_output_activations=tracing_enabled,
                    ),
                ),
                output_layer=SparseCategoricalSoftmaxOutputLayer(
                    input_dimensions=(N_DIGITS,)
                ),
            ),
        )


def training_options(f: Callable[P, R]) -> Callable[P, R]:
    @click.option(
        "--batch-size",
        type=int,
        help="Batch size",
        default=128,
    )
    @click.option(
        "--model-path",
        type=Path,
        help="Path to the trained model",
        default=None,
    )
    @click.option(
        "--num-epochs",
        type=int,
        help="Number of epochs",
        default=50,
    )
    @click.option(
        "--learning-rate",
        type=float,
        help="Learning rate",
        default=1e-3,
    )
    @click.option(
        "--warmup-epochs",
        type=int,
        help="Warmup epochs",
        default=5,
    )
    @click.option(
        "--model-output-path",
        type=Path,
        help="Path to save the trained model",
        default=None,
    )
    @click.option(
        "--log-level",
        type=LogLevel,
        help="Log level",
        default=LogLevel.INFO,
    )
    @click.option(
        "--lambda",
        "lambda_",
        type=float,
        help="Weight decay regulariser lambda",
        default=1e-4,
    )
    @click.option(
        "--train-split",
        type=float,
        help="Training split ratio",
        default=0.8,
    )
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper


@click.group()
def cli():
    """CNN (Convolutional Neural Network) model CLI for MNIST"""
    pass


@cli.command("train", help="Train a CNN model on MNIST")
@training_options
def train(
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    lambda_: float,
    warmup_epochs: int,
    model_path: Path | None,
    model_output_path: Path | None,
    log_level: LogLevel,
    train_split: float,
):
    setup_logging(log_level)

    print_device_info()

    from mo_net.data import SplitConfig, load_data

    X_train, Y_train, _, __ = load_data(
        MNIST_TRAIN_URL,
        split=SplitConfig.of(train_split, 0),
        one_hot=False,
    )

    X_train = X_train.reshape(-1, 1, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE)

    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Number of classes: {N_DIGITS}")

    seed = time.time_ns() // 1000
    logger.info(f"Seed: {seed}")
    key = jax.random.PRNGKey(seed)

    if model_path is None:
        model = CNNModel.create(key=key, tracing_enabled=False)
    else:
        model = CNNModel.load(model_path, training=True)

    training_parameters = TrainingParameters(
        batch_size=batch_size,
        dropout_keep_probs=(),
        history_max_len=100,
        learning_rate_limits=(learning_rate, learning_rate),
        log_level=log_level,
        max_restarts=0,
        monotonic=False,
        no_monitoring=True,
        normalisation_type=NormalisationType.NONE,
        num_epochs=num_epochs,
        quiet=False,
        regulariser_lambda=lambda_,
        seed=seed,
        trace_logging=False,
        train_set_size=len(X_train),
        warmup_epochs=warmup_epochs,
        workers=0,
    )

    train_size = int(train_split * len(X_train))
    X_train_split = X_train[:train_size]
    Y_train_split = Y_train[:train_size]
    X_val = X_train[train_size:]
    Y_val = Y_train[train_size:]

    run = TrainingRun(seed=seed, name=f"cnn_run_{seed}", backend=SqliteBackend())
    optimiser = get_optimiser("adam", model, training_parameters)

    trainer = BasicTrainer(
        X_train=X_train_split,
        X_val=X_val,
        Y_train=Y_train_split,
        Y_val=Y_val,
        key=jax.random.PRNGKey(seed),
        transform_fn=affine_transform2D(
            x_size=MNIST_IMAGE_SIZE, y_size=MNIST_IMAGE_SIZE
        ),
        loss_fn=sparse_cross_entropy,
        model=model,
        optimiser=optimiser,
        run=run,
        training_parameters=training_parameters,
    )

    logger.info(f"Starting CNN training with {len(X_train_split)} training samples")
    result = trainer.train()

    match result:
        case TrainingSuccessful():
            if model_output_path is None:
                model_output_path = DATA_DIR / "output" / f"{run.name}.pkl"
            result.model_checkpoint_path.rename(model_output_path)
            logger.info(f"Training completed. Model saved to: {model_output_path}")
        case TrainingFailed():
            logger.error(f"Training failed: {result.message}")
        case never:
            assert_never(never)


if __name__ == "__main__":
    cli()
