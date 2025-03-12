import pickle
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import IO, ClassVar, Final, Self

import click
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt
from more_itertools import sample
from tqdm import tqdm

DATA_DIR: Final[Path] = Path(__file__).parent.parent / "data"
DEFAULT_NUM_ITERATIONS: Final[int] = 10000
DEFAULT_LEARNING_RATE: Final[float] = 0.001
MAX_PIXEL_VALUE: Final[int] = 256
N_DIGITS: Final[int] = 10


def softmax(x: np.ndarray) -> np.ndarray:
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


def cross_entropy(Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
    Y_pred = np.clip(Y_pred, 1e-15, 1 - 1e-15)
    return -np.sum(Y_true * np.log(Y_pred))


class ModelBase(ABC):
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def dump(self, io: IO[bytes]) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, io: IO[bytes]) -> Self: ...

    def train(
        self,
        learning_rate: float,
        num_iterations: int,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_test: np.ndarray,
        Y_test: np.ndarray,
        training_log_path: Path,
    ) -> None:
        M = X_train.shape[0]
        scaling_factor = 1 / M
        last_update_time = time.time()

        if not training_log_path.exists():
            training_log = pd.DataFrame(
                columns=["iteration", "training_loss", "test_loss"]
            )
            training_log.to_csv(training_log_path, index=False)
        logger.info(
            f"Training model {self.__class__.__name__} for {num_iterations=} iterations with {learning_rate=}."
        )

        with tempfile.NamedTemporaryFile(delete=False) as f:
            for i in tqdm(range(num_iterations)):
                Z_train = self._forward_prop(X_train)
                self._backward_prop_and_update(
                    X_train, softmax(Z_train), Y_train, learning_rate
                )

                if i % 1024 == 0 and (time.time() - last_update_time) > 30:
                    Z_test = self._forward_prop(X_test)
                    L_train = scaling_factor * cross_entropy(softmax(Z_train), Y_train)
                    L_test = scaling_factor * cross_entropy(softmax(Z_test), Y_test)
                    tqdm.write(
                        f"Iteration {i}: Training Loss = {L_train}, Test Loss = {L_test}"
                    )
                    pd.DataFrame(
                        [[i, L_train, L_test]], columns=training_log.columns
                    ).to_csv(training_log_path, mode="a", header=False, index=False)
                    self.dump(open(f.name, "wb"))
                    last_update_time = time.time()


class LinearModel(ModelBase):
    @dataclass(frozen=True, kw_only=True)
    class Serialized:
        _tag: ClassVar[str] = "linear"
        W: np.ndarray
        b: np.ndarray

    def __init__(
        self,
        W: np.ndarray | None = None,
        b: np.ndarray | None = None,
    ):
        self._W = W
        self._b = b

    def _forward_prop(self, X: np.ndarray) -> np.ndarray:
        return X @ self._W + self._b

    def _backward_prop_and_update(
        self,
        X: np.ndarray,
        Y_pred: np.ndarray,
        Y_true: np.ndarray,
        learning_rate: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        dZ = Y_pred - Y_true
        dW = X.T @ dZ
        db = np.sum(dZ)
        self._W -= learning_rate * dW
        self._b -= learning_rate * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self._forward_prop(X), axis=1)

    def dump(self, io: IO[bytes]) -> None:
        pickle.dump(self.Serialized(W=self._W, b=self._b), io)

    @classmethod
    def load(cls, io: IO[bytes] | Serialized) -> Self:
        if isinstance(io, cls.Serialized):
            return cls(W=io.W, b=io.b)
        data = pickle.load(io)
        if data._tag != cls.Serialized._tag:
            raise ValueError(f"Invalid model type: {data._tag}")
        return cls(W=data.W, b=data.b)


def load_model(model_path: Path) -> ModelBase:
    serialized = pickle.load(open(model_path, "rb"))
    match serialized._tag:
        case LinearModel.Serialized._tag:
            logger.info(f"Loading linear model from {model_path}.")
            return LinearModel.load(serialized)
        case _:
            raise ValueError(f"Invalid model type: {serialized._tag}")


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(DATA_DIR / "mnist_test.csv")
    split_index: int = int(len(df) * 0.8)
    training_set = df.iloc[:split_index, :]
    test_set = df.iloc[split_index:, :]

    Y_train = np.eye(N_DIGITS)[training_set.iloc[:, 0].to_numpy()]
    Y_test = np.eye(N_DIGITS)[test_set.iloc[:, 0].to_numpy()]

    X_train = np.array(training_set.iloc[:, 1:]) / MAX_PIXEL_VALUE
    X_test = np.array(test_set.iloc[:, 1:]) / MAX_PIXEL_VALUE
    return X_train, Y_train, X_test, Y_test


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
    "-t",
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
def train(
    *,
    training_log_path: Path | None,
    num_iterations: int,
    learning_rate: float,
) -> None:
    X_train, Y_train, X_test, Y_test = load_data()

    seed = int(time.time())
    np.random.seed(seed)
    logger.info(f"Training model with {seed=}.")

    if training_log_path is None:
        model_path = DATA_DIR / f"model_{seed=}_{num_iterations=}_{learning_rate=}.pkl"
        training_log_path = model_path.with_name(f"{model_path.stem}_training_log.csv")
    else:
        model_path = training_log_path.with_name(
            f"{training_log_path.stem.replace('_training_log', '')}.pkl"
        )

    model = LinearModel(
        W=np.random.randn(X_train.shape[1], Y_train.shape[1]),
        b=np.random.randn(Y_train.shape[1]),
    )
    model.train(
        learning_rate=learning_rate,
        num_iterations=num_iterations,
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        Y_test=Y_test,
        training_log_path=training_log_path,
    )

    model.dump(open(model_path, "wb"))


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
    W = model._W.reshape(28, 28, 10)
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


if __name__ == "__main__":
    cli()
