import functools
import pickle
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Callable, Final, ParamSpec, TypeVar

import click
import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from more_itertools import sample

from mnist_numpy.data import DEFAULT_DATA_PATH, OUTPUT_PATH, RUN_PATH, load_data
from mnist_numpy.model import (
    ModelBase,
    MultilayerPerceptron,
)
from mnist_numpy.optimizer import (
    AdalmOptimizer,
    AdamOptimizer,
    NoOptimizer,
    OptimizerBase,
)
from mnist_numpy.train import (
    ModelTrainer,
    TrainingParameters,
)

P = ParamSpec("P")
R = TypeVar("R")

DEFAULT_BATCH_SIZE: Final[int] = 100
DEFAULT_LEARNING_RATE: Final[float] = 0.1
DEFAULT_LEARNING_RATE_LIMITS: Final[str] = "0.000001, 1"
DEFAULT_MOMENTUM_PARAMETER: Final[float] = 0.9
DEFAULT_NUM_EPOCHS: Final[int] = 10000
DEFAULT_RESCALE_FACTOR_PER_EPOCH: Final[float] = 1.5


def training_options(f: Callable[P, R]) -> Callable[P, R]:
    @click.option(
        "-a",
        "--learning-rate",
        help="Set the learning rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )
    @click.option(
        "-l",
        "--training-log-path",
        type=Path,
        help="Set the path to the training log file",
        default=None,
    )
    @click.option(
        "-b",
        "--batch-size",
        type=int,
        help="Set the batch size",
        default=None,
    )
    @click.option(
        "-d",
        "--data-path",
        type=Path,
        help="Set the path to the data file",
        default=DEFAULT_DATA_PATH,
    )
    @click.option(
        "-m",
        "--momentum-parameter",
        type=float,
        help="Set the momentum parameter",
        default=DEFAULT_MOMENTUM_PARAMETER,
    )
    @click.option(
        "-r",
        "--learning-rate-rescale-factor-per-epoch",
        type=float,
        help="Set the learning rate rescale factor per epoch",
        default=DEFAULT_RESCALE_FACTOR_PER_EPOCH,
    )
    @click.option(
        "-s",
        "--learning-rate-limits",
        type=lambda x: tuple(float(y) for y in x.split(",")),
        help="Set the learning rate limits",
        default=DEFAULT_LEARNING_RATE_LIMITS,
    )
    @click.option(
        "-o",
        "--optimizer-type",
        type=click.Choice(["adam", "adalm", "no"]),
        help="The type of optimizer to use",
        default="adam",
    )
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper


@click.group()
def cli(): ...


@cli.command(help="Train the model")
@training_options
@click.option(
    "-n",
    "--num-epochs",
    help="Set number of epochs",
    type=int,
    default=DEFAULT_NUM_EPOCHS,
)
@click.option(
    "-t",
    "--model-type",
    type=click.Choice([MultilayerPerceptron.Serialized._tag]),
    help="The type of model to train",
    default=MultilayerPerceptron.Serialized._tag,
)
@click.option(
    "-i",
    "--dims",
    type=int,
    multiple=True,
    default=(10, 10),
)
def train(
    *,
    batch_size: int | None,
    data_path: Path,
    dims: Sequence[int],
    learning_rate: float,
    learning_rate_rescale_factor_per_epoch: float,
    learning_rate_limits: tuple[float, float],
    model_type: str,
    momentum_parameter: float,
    num_epochs: int,
    optimizer_type: str,
    training_log_path: Path | None,
) -> None:
    X_train, Y_train, X_test, Y_test = load_data(data_path)

    seed = int(time.time())
    np.random.seed(seed)
    logger.info(f"Training model with {seed=}.")

    model: ModelBase
    match model_type:
        case MultilayerPerceptron.Serialized._tag:
            model = MultilayerPerceptron.initialize(X_train.shape[1], *dims, 10)
        case _:
            raise ValueError(f"Invalid model type: {model_type}")

    if training_log_path is None:
        model_path = OUTPUT_PATH / f"{seed}_{model.get_name()}_model.pkl"
        training_log_path = RUN_PATH / (f"{model_path.stem}_training_log.csv")
    else:
        if (re.search(r"_training_log\.csv$", training_log_path.name)) is None:
            training_log_path = training_log_path.with_name(
                f"{training_log_path.stem}_training_log.csv"
            )
        model_path = OUTPUT_PATH / training_log_path.name.replace(
            "training_log.csv", ".pkl"
        )

    training_parameters = TrainingParameters(
        batch_size=(batch_size if batch_size is not None else X_train.shape[0]),
        learning_rate=learning_rate,
        learning_rate_limits=learning_rate_limits,
        learning_rate_rescale_factor_per_epoch=learning_rate_rescale_factor_per_epoch,
        momentum_parameter=momentum_parameter,
        num_epochs=num_epochs,
        total_epochs=num_epochs,
    )

    optimizer: OptimizerBase
    match optimizer_type:
        case "adam":
            optimizer = AdamOptimizer(
                model=model,
                config=AdamOptimizer.Config(learning_rate=learning_rate),
            )
        case "adalm":
            optimizer = AdalmOptimizer(
                model=model,
                config=AdalmOptimizer.Config(
                    num_epochs=num_epochs,
                    train_set_size=(train_set_size := X_train.shape[0]),
                    batch_size=(
                        batch_size if batch_size is not None else train_set_size
                    ),
                    learning_rate=learning_rate,
                    learning_rate_limits=learning_rate_limits,
                    learning_rate_rescale_factor_per_epoch=learning_rate_rescale_factor_per_epoch,
                    momentum_parameter=momentum_parameter,
                ),
            )
        case "no":
            optimizer = NoOptimizer(
                config=NoOptimizer.Config(learning_rate=learning_rate),
            )
        case _:
            raise ValueError(f"Invalid optimizer: {optimizer}")

    ModelTrainer.train(
        model=model,
        X_test=X_test,
        X_train=X_train,
        Y_test=Y_test,
        Y_train=Y_train,
        training_parameters=training_parameters,
        optimizer=optimizer,
        training_log_path=training_log_path,
    ).rename(model_path)
    logger.info(f"Saved output to {model_path}.")


@cli.command(help="Resume training the model")
@click.option(
    "-m",
    "--model-path",
    help="Set the path to the model file",
    type=Path,
    required=True,
)
@click.option(
    "-p",
    "--training-parameters-path",
    help="Set the path to the training parameter file",
    type=Path,
    required=True,
)
@click.option(
    "-d",
    "--data-path",
    type=Path,
    help="Set the path to the data file",
    default=DEFAULT_DATA_PATH,
)
def resume(
    *,
    data_path: Path,
    model_path: Path,
    training_parameters_path: Path,
):
    X_train, Y_train, X_test, Y_test = load_data(data_path)

    training_log_path = training_parameters_path.with_name(
        training_parameters_path.name.replace(
            "training_parameters.json", "training_log.csv"
        )
    )
    output_path = OUTPUT_PATH / training_parameters_path.name.replace(
        "training_parameters.json", ".pkl"
    )

    model = MultilayerPerceptron.load(pickle.load(open(model_path, "rb")))
    training_parameters = TrainingParameters.model_validate_json(
        training_parameters_path.read_text()
    )
    ModelTrainer.train(
        # TODO: Dispatch on model type
        model=model,
        X_test=X_test,
        X_train=X_train,
        Y_test=Y_test,
        Y_train=Y_train,
        training_parameters=training_parameters,
        optimizer=AdalmOptimizer(
            model=model,
            config=AdalmOptimizer.Config(
                num_epochs=training_parameters.num_epochs,
                train_set_size=X_train.shape[0],
                batch_size=training_parameters.batch_size,
                learning_rate=training_parameters.learning_rate,
                learning_rate_limits=training_parameters.learning_rate_limits,
                learning_rate_rescale_factor_per_epoch=training_parameters.learning_rate_rescale_factor_per_epoch,
                momentum_parameter=training_parameters.momentum_parameter,
            ),
        ),
        training_log_path=training_log_path,
    ).rename(output_path)
    logger.info(f"Saved output to {output_path}.")


@cli.command(help="Run inference using the model")
@click.option(
    "-m",
    "--model-path",
    help="Set the path to the model file",
    type=Path,
    required=True,
)
@click.option(
    "-d",
    "--data-path",
    type=Path,
    help="Set the path to the data file",
    default=DEFAULT_DATA_PATH,
)
def infer(*, model_path: Path, data_path: Path):
    X_train, Y_train, X_test, Y_test = load_data(data_path)

    # TODO: Dispatch on model type
    model = MultilayerPerceptron.load(pickle.load(open(model_path, "rb")))

    Y_pred = model.predict(X_train)
    Y_true = np.argmax(Y_train, axis=1)
    logger.info(f"Training Set Accuracy: {np.sum(Y_pred == Y_true) / len(Y_pred)}")

    Y_pred = model.predict(X_test)
    Y_true = np.argmax(Y_test, axis=1)
    logger.info(f"Test Set Accuracy: {np.sum(Y_pred == Y_true) / len(Y_pred)}")

    plt.figure(figsize=(15, 8))
    plt.suptitle("Mislabelled Samples", fontsize=16)

    sample_indices = sample(np.where(Y_true != Y_pred)[0], 25)
    for idx, i in enumerate(sample_indices):
        plt.subplot(8, 5, idx + 1)
        plt.imshow(X_test[i].reshape(28, 28), cmap="gray")
        plt.title(f"Pred: {Y_pred[i]}, True: {Y_true[i]}")
        plt.axis("off")
    plt.subplot(8, 1, (6, 8))
    unique_labels = list(range(10))
    counts_pred = [np.sum(Y_pred == label) for label in unique_labels]
    counts_true = [np.sum(Y_true == label) for label in unique_labels]
    counts_correct = [
        np.sum((Y_true == Y_pred) & (Y_true == label)) for label in unique_labels
    ]

    bar_width = 0.25
    x = np.arange(len(unique_labels))

    plt.bar(x - bar_width, counts_pred, bar_width, label="Predicted")
    plt.bar(x, counts_true, bar_width, label="True")
    plt.bar(x + bar_width, counts_correct, bar_width, label="Correct")

    plt.xticks(x, [str(label) for label in unique_labels])
    plt.xlabel("Digit")
    plt.ylabel("Count")
    plt.title("Predicted vs. True Label Distribution (Sample)")
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.tight_layout()
    plt.show()


@cli.command(help="Run explainability analysis")
@click.option(
    "-m",
    "--model-path",
    help="Set the path to the model file",
    required=True,
    type=Path,
)
@click.option(
    "-d",
    "--data-path",
    type=Path,
    help="Set the path to the data file",
    default=DEFAULT_DATA_PATH,
)
def explain(*, model_path: Path, data_path: Path):
    X_train = load_data(data_path)[0]
    # TODO: Dispatch on model type
    model = MultilayerPerceptron.load(pickle.load(open(model_path, "rb")))
    W = model._W[0].reshape(28, 28, 10)
    avg = np.average(X_train, axis=0).reshape(28, 28)

    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(np.multiply(W[:, :, i], avg), cmap="gray")
        plt.title(str(i))
        plt.axis("off")
    plt.show()


@cli.command(help="Sample input data", name="sample")
@click.option(
    "-d",
    "--data-path",
    type=Path,
    help="Set the path to the data file",
    default=DEFAULT_DATA_PATH,
)
def sample_data(*, data_path: Path):
    X_train = load_data(data_path)[0]
    sample_indices = sample(range(len(X_train)), 25)
    for idx, i in enumerate(sample_indices):
        plt.subplot(5, 5, idx + 1)
        plt.imshow(X_train[i].reshape(28, 28), cmap="gray")
        plt.axis("off")
    plt.show()
