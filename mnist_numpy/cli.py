import functools
import os
import re
import sys
import time
from collections.abc import Sequence
from itertools import chain
from pathlib import Path
from typing import Callable, Final, ParamSpec, TypeVar

import click
import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from more_itertools import sample

from mnist_numpy.data import (
    DATA_DIR,
    DEFAULT_DATA_PATH,
    OUTPUT_PATH,
    RUN_PATH,
    load_data,
)
from mnist_numpy.functions import LeakyReLU, ReLU, Tanh, get_activation_fn
from mnist_numpy.model import MultiLayerPerceptron
from mnist_numpy.model.scheduler import CosineScheduler, WarmupScheduler
from mnist_numpy.optimizer import (
    AdamOptimizer,
    NoOptimizer,
    OptimizerBase,
)
from mnist_numpy.regulariser.dropout import DropoutRegulariser
from mnist_numpy.regulariser.ridge import L2Regulariser
from mnist_numpy.trainer import (
    ModelTrainer,
    TrainingParameters,
)
from mnist_numpy.trainer.exceptions import AbortTraining

P = ParamSpec("P")
R = TypeVar("R")

DEFAULT_BATCH_SIZE: Final[int] = 100
DEFAULT_LEARNING_RATE: Final[float] = 0.001
DEFAULT_LEARNING_RATE_LIMITS: Final[str] = "0.0000001, 0.01"
DEFAULT_NUM_EPOCHS: Final[int] = 1000
N_DIGITS: Final[int] = 10


def training_options(f: Callable[P, R]) -> Callable[P, R]:
    # TODO: Remove optimizer specific options
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
        default=DEFAULT_BATCH_SIZE,
    )
    @click.option(
        "-d",
        "--data-path",
        type=Path,
        help="Set the path to the data file",
        default=DEFAULT_DATA_PATH,
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
    type=click.Choice([MultiLayerPerceptron.get_name()]),
    help="The type of model to train",
    default=MultiLayerPerceptron.get_name(),
)
@click.option(
    "-i",
    "--dims",
    type=int,
    multiple=True,
    default=(10, 10),
)
@click.option(
    "-m",
    "--model-path",
    help="Set the path to the model file",
    type=Path,
)
@click.option(
    "-k",
    "--dropout-keep-prob",
    type=float,
    help="Set the dropout keep probability",
    multiple=True,
    default=(),
)
@click.option(
    "-f",
    "--activation-fn",
    type=click.Choice([ReLU.name, Tanh.name, LeakyReLU.name]),
    help="Set the activation function",
    default=ReLU.name,
)
@click.option(
    "-t",
    "--trace-logging",
    type=bool,
    is_flag=True,
    help="Set the trace logging",
    default=False,
)
@click.option(
    "-l",
    "--regulariser-lambda",
    type=float,
    help="Set the regulariser lambda",
    default=0.0,
)
@click.option(
    "-w",
    "--warmup-epochs",
    type=int,
    help="Set the number of warmup epochs",
    default=100,
)
@click.option(
    "-r",
    "--max-restarts",
    type=int,
    help="Set the maximum number of restarts",
    default=0,
)
def train(
    *,
    activation_fn: str,
    batch_size: int | None,
    data_path: Path,
    dims: Sequence[int],
    dropout_keep_prob: tuple[float, ...],
    learning_rate: float,
    learning_rate_limits: tuple[float, float],
    model_path: Path | None,
    max_restarts: int,
    model_type: str,
    num_epochs: int,
    optimizer_type: str,
    regulariser_lambda: float,
    training_log_path: Path | None,
    trace_logging: bool,
    warmup_epochs: int,
) -> None:
    X_train, Y_train, X_test, Y_test = load_data(data_path)

    seed = int(os.getenv("MNIST_SEED", time.time()))
    np.random.seed(seed)
    logger.info(f"Training model with {seed=}.")

    model: MultiLayerPerceptron
    regularisers = tuple(
        (
            chain(
                (L2Regulariser(lambda_=regulariser_lambda, batch_size=batch_size),)
                if regulariser_lambda > 0
                else (),
                (
                    (DropoutRegulariser(keep_probs=dropout_keep_prob),)
                    if dropout_keep_prob
                    else ()
                ),
            )
        )
    )
    if model_path is None:
        if model_type == MultiLayerPerceptron.get_name():
            model = MultiLayerPerceptron.of(
                layer_neuron_counts=(X_train.shape[1], *dims, N_DIGITS),
                activation_fn=get_activation_fn(activation_fn),
            )
        else:
            raise ValueError(f"Invalid model type: {model_type}")
    else:
        model = MultiLayerPerceptron.load(open(model_path, "rb"))

    for regulariser in regularisers:
        regulariser(model)

    if training_log_path is None:
        model_path = OUTPUT_PATH / f"{int(time.time())}_{model.get_name()}_model.pkl"
        training_log_path = RUN_PATH / (f"{model_path.stem}_training_log.csv")
    else:
        if (re.search(r"_training_log\.csv$", training_log_path.name)) is None:
            training_log_path = training_log_path.with_name(
                f"{training_log_path.stem}_training_log.csv"
            )
        model_path = OUTPUT_PATH / training_log_path.name.replace(
            "training_log.csv", ".pkl"
        )
    train_set_size = X_train.shape[0]
    batch_size = batch_size if batch_size is not None else train_set_size
    training_parameters = TrainingParameters(
        batch_size=batch_size,
        dropout_keep_prob=dropout_keep_prob,
        learning_rate_limits=learning_rate_limits,
        low_gradient_abort_threshold=1e-6,
        high_gradient_abort_threshold=1e3,
        num_epochs=num_epochs,
        regulariser_lambda=regulariser_lambda,
        total_epochs=num_epochs,
        trace_logging=trace_logging,
        train_set_size=train_set_size,
        warmup_epochs=warmup_epochs,
    )

    optimizer: OptimizerBase
    match optimizer_type:
        case "adam":
            optimizer = AdamOptimizer(
                model=model,
                config=AdamOptimizer.Config(
                    learning_rate=learning_rate,
                    scheduler=WarmupScheduler.of(
                        training_parameters=training_parameters,
                        next_scheduler=CosineScheduler.of(
                            training_parameters=training_parameters,
                        ),
                    ),
                ),
            )
        case "no":
            optimizer = NoOptimizer(
                config=NoOptimizer.Config(learning_rate=learning_rate),
            )
        case _:
            raise ValueError(f"Invalid optimizer: {optimizer}")

    def save_model(model_checkpoint_path: Path) -> None:
        model_checkpoint_path.rename(model_path)
        logger.info(f"Saved output to {model_path}.")

    restarts = 0
    if training_parameters.trace_logging:
        # Disable restarts when tracing
        max_restarts = 0
    while restarts <= max_restarts:
        try:
            training_result = ModelTrainer.train(
                model=model,
                X_test=X_test,
                X_train=X_train,
                Y_test=Y_test,
                Y_train=Y_train,
                training_parameters=training_parameters,
                optimizer=optimizer,
                training_log_path=training_log_path,
            )
        except AbortTraining as e:
            logger.exception(e)
            if e.training_progress is not None and e.training_progress >= 0.1:
                break
            model.reinitialise()
            restarts += 1
    save_model(training_result.model_checkpoint_path)


@cli.command(help="Run inference using the model")
@click.option(
    "-m",
    "--model-path",
    help="Set the path to the model file",
    type=Path,
)
@click.option(
    "-d",
    "--data-path",
    type=Path,
    help="Set the path to the data file",
    default=DEFAULT_DATA_PATH,
)
def infer(*, model_path: Path | None, data_path: Path):
    X_train, Y_train, X_test, Y_test = load_data(data_path)

    if model_path is None:
        output_dir = DATA_DIR / "output"
        model_path = max(output_dir.glob("*.pkl"), key=lambda p: p.stat().st_mtime)
        logger.info(f"Using latest model file: {model_path}")
    if not model_path.exists():
        logger.error(f"File not found: {model_path}")
        sys.exit(1)

    model = MultiLayerPerceptron.load(open(model_path, "rb"))

    Y_pred = model.predict(X_train)
    Y_true = np.argmax(Y_train, axis=1)
    logger.info(f"Training Set Accuracy: {np.sum(Y_pred == Y_true) / len(Y_pred)}")

    Y_pred = model.predict(X_test)
    Y_true = np.argmax(Y_test, axis=1)
    logger.info(f"Test Set Accuracy: {np.sum(Y_pred == Y_true) / len(Y_pred)}")

    plt.figure(figsize=(15, 8))
    plt.suptitle("Mislabelled Examples (Sample)", fontsize=16)

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
