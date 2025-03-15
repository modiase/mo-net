from collections.abc import Sequence
import re
import time
from pathlib import Path

import click
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt
from more_itertools import sample
from mnist_numpy.data import DATA_DIR, load_data
from mnist_numpy.model import (
    LinearRegressionModel,
    ModelBase,
    MultilayerPerceptron,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_ITERATIONS,
    load_model,
)


@click.group()
def cli(): ...


@cli.command(help="Train the model")
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
    "-n",
    "--num-iterations",
    help="Set number of iterations",
    type=int,
    default=DEFAULT_NUM_ITERATIONS,
)
@click.option(
    "-t",
    "--model-type",
    type=click.Choice(
        [
            LinearRegressionModel.Serialized._tag,
            MultilayerPerceptron.Serialized._tag,
        ]
    ),
    help="The type of model to train",
)
@click.option(
    "-d",
    "--dims",
    type=int,
    multiple=True,
    default=(10, 10),
)
def train(
    *,
    training_log_path: Path | None,
    num_iterations: int,
    learning_rate: float,
    model_type: str,
    dims: Sequence[int],
) -> None:
    X_train, Y_train, X_test, Y_test = load_data()

    seed = int(time.time())
    np.random.seed(seed)
    logger.info(f"Training model with {seed=}.")

    model: ModelBase
    match model_type:
        case LinearRegressionModel.Serialized._tag:
            model = LinearRegressionModel.initialize(X_train.shape[1], Y_train.shape[1])
        case MultilayerPerceptron.Serialized._tag:
            model = MultilayerPerceptron.initialize(X_train.shape[1], *dims)
        case _:
            raise ValueError(f"Invalid model type: {model_type}")

    if training_log_path is None:
        model_path = (
            DATA_DIR
            / f"{seed}_{model_type}_model_{num_iterations=}_{learning_rate=}.pkl"
        )
        training_log_path = model_path.with_name(f"{model_path.stem}_training_log.csv")
    else:
        model_path = training_log_path.with_name(
            f"{training_log_path.stem.replace('_training_log', '')}.pkl"
        )

    model.train(
        learning_rate=learning_rate,
        num_iterations=num_iterations,
        total_iterations=num_iterations,
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        Y_test=Y_test,
        training_log_path=training_log_path,
    )

    model.dump(open(model_path, "wb"))


@cli.command(help="Resume training the model")
@click.option(
    "-m",
    "--model-path",
    help="Set the path to the model file",
    type=Path,
    required=True,
)
@click.option(
    "-l",
    "--training-log-path",
    type=Path,
    help="Set the path to the training log file",
)
@click.option(
    "-a",
    "--learning-rate",
    type=float,
    help="Set the learning rate",
    default=DEFAULT_LEARNING_RATE,
)
@click.option(
    "-n",
    "--num-iterations",
    type=int,
    help="Set number of iterations",
    default=None,
)
def resume(
    model_path: Path,
    training_log_path: Path,
    learning_rate: float,
    num_iterations: int | None,
):
    X_train, Y_train, X_test, Y_test = load_data()

    if num_iterations is None:
        if (mo := re.search(r"num_iterations=(\d+)", training_log_path.name)) is None:
            raise ValueError(f"Invalid training log path: {training_log_path}")
        total_iterations = int(mo.group(1))
        num_iterations = total_iterations

    if not (training_log := pd.read_csv(training_log_path)).empty:
        num_iterations = total_iterations - int(training_log.iloc[-1, 0])  # type: ignore[arg-type]

    model = load_model(model_path)
    model.train(
        learning_rate=learning_rate,
        num_iterations=num_iterations,
        total_iterations=total_iterations,
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        Y_test=Y_test,
        training_log_path=training_log_path,
    )


@cli.command(help="Run inference using the model")
@click.option(
    "-m",
    "--model-path",
    help="Set the path to the model file",
    type=Path,
    required=True,
)
def infer(model_path: Path):
    X_train, Y_train, X_test, Y_test = load_data()
    model = load_model(model_path)

    Y_pred = model.predict(X_train)
    Y_true = np.argmax(Y_train, axis=1)
    logger.info(f"Training Set Accuracy: {np.sum(Y_pred == Y_true) / len(Y_pred)}")

    Y_pred = model.predict(X_test)
    Y_true = np.argmax(Y_test, axis=1)
    logger.info(f"Test Set Accuracy: {np.sum(Y_pred == Y_true) / len(Y_pred)}")

    plt.figure(figsize=(15, 8))
    sample_indices = sample(range(len(X_test)), 25)
    for idx, i in enumerate(sample_indices):
        plt.subplot(5, 5, idx + 1)
        plt.imshow(X_test[i].reshape(28, 28), cmap="gray")
        plt.title(f"Pred: {Y_pred[i]}, True: {Y_true[i]}")
        plt.axis("off")
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
def explain(*, model_path: Path):
    X_train = load_data()[0]
    model = load_model(model_path)
    W = model._W[0].reshape(28, 28, 10)
    avg = np.average(X_train, axis=0).reshape(28, 28)

    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(np.multiply(W[:, :, i], avg), cmap="gray")
        plt.title(str(i))
        plt.axis("off")
    plt.show()


@cli.command(help="Sample input data", name="sample")
def sample_():
    X_train = load_data()[0]
    sample_indices = sample(range(len(X_train)), 25)
    for idx, i in enumerate(sample_indices):
        plt.subplot(5, 5, idx + 1)
        plt.imshow(X_train[i].reshape(28, 28), cmap="gray")
        plt.axis("off")
    plt.show()
