from __future__ import annotations

import functools
import time
from pathlib import Path
from typing import Callable, ParamSpec, TypeVar, assert_never

import click
import jax
import jax.numpy as jnp
from loguru import logger

from mo_net import print_device_info
from mo_net.data import DATA_DIR
from mo_net.functions import get_activation_fn, sparse_cross_entropy
from mo_net.log import LogLevel, setup_logging
from mo_net.model.layer.activation import Activation
from mo_net.model.layer.linear import Linear
from mo_net.model.layer.output import SparseCategoricalSoftmaxOutputLayer
from mo_net.model.model import Model
from mo_net.model.module.base import Hidden, Output
from mo_net.protos import NormalisationType
from mo_net.resources import MNIST_TRAIN_URL
from mo_net.train import TrainingParameters
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
MNIST_INPUT_SIZE: int = MNIST_IMAGE_SIZE * MNIST_IMAGE_SIZE  # 784


class MLPModel(Model):
    @classmethod
    def get_name(cls) -> str:
        return "mlp"

    @classmethod
    def get_description(cls) -> str:
        return "Multi-Layer Perceptron for MNIST"

    @classmethod
    def create(
        cls,
        *,
        key: jax.Array,
        layer_sizes: list[int],
        activation: str = "relu",
        tracing_enabled: bool = False,
    ) -> MLPModel:
        if not layer_sizes:
            raise ValueError("layer_sizes cannot be empty")

        activation_fn = get_activation_fn(activation)
        hidden_modules = []
        current_size = MNIST_INPUT_SIZE

        for layer_size in layer_sizes:
            key, subkey = jax.random.split(key)
            hidden_modules.append(
                Hidden(
                    layers=(
                        Linear(
                            input_dimensions=(current_size,),
                            output_dimensions=(layer_size,),
                            parameters_init_fn=functools.partial(
                                Linear.Parameters.xavier, key=subkey
                            ),
                            store_output_activations=tracing_enabled,
                        ),
                        Activation(
                            input_dimensions=(layer_size,),
                            activation_fn=activation_fn,
                        ),
                    )
                )
            )
            current_size = layer_size

        key, subkey = jax.random.split(key)
        return cls(
            input_dimensions=(MNIST_INPUT_SIZE,),
            hidden=tuple(hidden_modules),
            output=Output(
                layers=(
                    Linear(
                        input_dimensions=(current_size,),
                        output_dimensions=(N_DIGITS,),
                        parameters_init_fn=functools.partial(
                            Linear.Parameters.xavier, key=subkey
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
    @click.option("--batch-size", type=int, help="Batch size", default=32)
    @click.option(
        "--model-path", type=Path, help="Path to the trained model", default=None
    )
    @click.option("--num-epochs", type=int, help="Number of epochs", default=50)
    @click.option("--learning-rate", type=float, help="Learning rate", default=1e-3)
    @click.option("--warmup-epochs", type=int, help="Warmup epochs", default=5)
    @click.option(
        "--model-output-path",
        type=Path,
        help="Path to save the trained model",
        default=None,
    )
    @click.option("--log-level", type=LogLevel, help="Log level", default=LogLevel.INFO)
    @click.option(
        "--lambda",
        "lambda_",
        type=float,
        help="Weight decay regulariser lambda",
        default=1e-4,
    )
    @click.option("--train-split", type=float, help="Training split ratio", default=0.8)
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper


@click.group()
def cli():
    """MLP (Multi-Layer Perceptron) model CLI for MNIST"""
    pass


@cli.command("train", help="Train an MLP model on MNIST")
@click.option(
    "-i",
    "--layer-sizes",
    type=str,
    help="Comma-separated list of hidden layer sizes (e.g., '128,64,32')",
    default="128,64",
    required=True,
)
@click.option(
    "--activation",
    type=click.Choice(["relu", "tanh", "sigmoid"]),
    help="Activation function",
    default="relu",
)
@training_options
def train(
    layer_sizes: str,
    activation: str,
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

    try:
        layer_sizes_list = [int(x.strip()) for x in layer_sizes.split(",")]
        if not layer_sizes_list or any(size <= 0 for size in layer_sizes_list):
            raise ValueError("Layer sizes must be positive integers")
    except ValueError as e:
        raise click.BadParameter(f"Invalid layer sizes '{layer_sizes}': {e}")

    X_train, Y_train, _, __ = load_data(
        MNIST_TRAIN_URL,
        split=SplitConfig.of(train_split, 0),
        one_hot=False,
    )
    X_train = X_train.reshape(-1, MNIST_INPUT_SIZE)

    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Number of classes: {N_DIGITS}")
    logger.info(f"Layer sizes: {layer_sizes_list}")
    logger.info(f"Activation function: {activation}")

    seed = time.time_ns() // 1000
    logger.info(f"Seed: {seed}")
    key = jax.random.PRNGKey(seed)

    model = (
        MLPModel.create(
            key=key,
            layer_sizes=layer_sizes_list,
            activation=activation,
            tracing_enabled=False,
        )
        if model_path is None
        else MLPModel.load(model_path, training=True)
    )

    logger.info(f"Model created: {model.print()}")
    logger.info(f"Model parameter count: {model.parameter_count}")

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
    run = TrainingRun(seed=seed, name=f"mlp_run_{seed}", backend=SqliteBackend())

    trainer = BasicTrainer(
        X_train=X_train[:train_size],
        X_val=X_train[train_size:],
        Y_train=Y_train[:train_size],
        Y_val=Y_train[train_size:],
        key=jax.random.PRNGKey(seed),
        transform_fn=None,
        loss_fn=sparse_cross_entropy,
        model=model,
        optimiser=get_optimiser("adam", model, training_parameters),
        run=run,
        training_parameters=training_parameters,
    )

    logger.info(f"Starting MLP training with {train_size} training samples")
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


@cli.command("predict", help="Make predictions with a trained MLP model")
@click.option(
    "--model-path",
    type=Path,
    help="Path to the trained model",
    required=True,
)
@click.option(
    "--data-path",
    type=Path,
    help="Path to test data (CSV file)",
    default=None,
)
def predict(
    model_path: Path,
    data_path: Path | None,
):
    """Make predictions with a trained MLP model."""
    setup_logging(LogLevel.INFO)

    logger.info(f"Loading model from {model_path}")
    model = MLPModel.load(model_path, training=False)

    if data_path is None:
        from mo_net.data import SplitConfig, load_data
        from mo_net.resources import MNIST_TEST_URL

        X_test, Y_test, _, __ = load_data(
            MNIST_TEST_URL,
            split=SplitConfig.of(0.9, 0),
            one_hot=False,
        )
        X_test = X_test.reshape(-1, MNIST_INPUT_SIZE)
    else:
        import pandas as pd

        X_test = jnp.array(pd.read_csv(data_path).values.reshape(-1, MNIST_INPUT_SIZE))
        Y_test = None  # No labels available for custom data

    logger.info(f"Making predictions on {len(X_test)} samples")
    predictions = model.predict(X_test)

    if Y_test is not None:
        accuracy = (predictions == Y_test).mean()
        logger.info(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    else:
        logger.info("No ground truth labels available for custom data")
        logger.info(f"Predictions: {predictions[:10]}...")


if __name__ == "__main__":
    cli()
