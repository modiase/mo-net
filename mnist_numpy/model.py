from itertools import cycle
import pickle
import tempfile
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, ClassVar, Final, Self

import numpy as np
import pandas as pd
from loguru import logger
from more_itertools import pairwise
from tqdm import tqdm

DEFAULT_NUM_ITERATIONS: Final[int] = 10000
DEFAULT_LEARNING_RATE: Final[float] = 0.001


def softmax(x: np.ndarray) -> np.ndarray:
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


def cross_entropy(Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
    return -np.sum(Y_true * np.log(Y_pred))


def ReLU(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def deriv_ReLU(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, 1, 0)


class ModelBase(ABC):
    _W: Sequence[np.ndarray]
    _b: Sequence[np.ndarray]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(softmax(self._forward_prop(X)[1][-1]), axis=1)

    @classmethod
    @abstractmethod
    def get_description(cls) -> str: ...

    @classmethod
    @abstractmethod
    def initialize(cls, *dims: int) -> Self: ...

    @abstractmethod
    def dump(self, io: IO[bytes]) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, io: IO[bytes]) -> Self: ...

    @abstractmethod
    def _forward_prop(
        self, X: np.ndarray
    ) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]: ...

    @abstractmethod
    def _backward_prop_and_update(
        self,
        X: np.ndarray,
        Y_true: np.ndarray,
        Z: tuple[np.ndarray, ...],
        A: tuple[np.ndarray, ...],
        learning_rate: float,
    ) -> None: ...

    def train(
        self,
        *,
        learning_rate: float,
        num_iterations: int,
        total_iterations: int,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_test: np.ndarray,
        Y_test: np.ndarray,
        training_log_path: Path,
        training_log_interval_seconds: int = 30,
        batch_size: int | None = None,
    ) -> Path:
        last_update_time = time.time()

        if not training_log_path.exists():
            training_log = pd.DataFrame(
                columns=["iteration", "training_loss", "test_loss"]
            )
            training_log.to_csv(training_log_path, index=False)
        else:
            training_log = pd.read_csv(training_log_path)

        logger.info(
            f"Training model {self.__class__.__name__} for {num_iterations=} iterations with {learning_rate=}."
        )

        X_train_batched = cycle(np.array_split(X_train, batch_size))
        Y_train_batched = cycle(np.array_split(Y_train, batch_size))

        model_checkpoint_path = training_log_path.with_name(
            training_log_path.name.replace("training_log.csv", "partial.pkl")
        )

        logger.info(
            f"Training model..\nSaving partial results to: {model_checkpoint_path}."
        )
        logger.info(f"\n{training_log_path=}.")
        self.dump(open(model_checkpoint_path, "wb"))
        start_iteration = total_iterations - num_iterations
        for i in tqdm(
            range(start_iteration, total_iterations),
            initial=start_iteration,
            total=num_iterations,
        ):
            X_train_batch = next(X_train_batched)
            Y_train_batch = next(Y_train_batched)
            Z_train_batch, A_train_batch = self._forward_prop(X_train_batch)
            self._backward_prop_and_update(
                X_train_batch,
                Y_train_batch,
                Z_train_batch,
                A_train_batch,
                learning_rate,
            )

            if (
                i % 1024 == 0
                and (time.time() - last_update_time) > training_log_interval_seconds
            ):
                _, A_train = self._forward_prop(X_train)
                L_train = (
                    1 / X_train.shape[0] * cross_entropy(softmax(A_train[-1]), Y_train)
                )
                _, A_test = self._forward_prop(X_test)
                L_test = (
                    1 / X_test.shape[0] * cross_entropy(softmax(A_test[-1]), Y_test)
                )
                tqdm.write(
                    f"Iteration {i}: Training Loss = {L_train}, Test Loss = {L_test}"
                )
                pd.DataFrame(
                    [[i, L_train, L_test]], columns=training_log.columns
                ).to_csv(training_log_path, mode="a", header=False, index=False)
                self.dump(open(model_checkpoint_path, "wb"))
                last_update_time = time.time()

        return model_checkpoint_path


class LinearRegressionModel(ModelBase):
    @dataclass(frozen=True, kw_only=True)
    class Serialized:
        _tag: ClassVar[str] = "lreg"
        W: np.ndarray
        b: np.ndarray

    @classmethod
    def get_description(cls) -> str:
        return "Linear Regression Model"

    @classmethod
    def initialize(cls, *dims: int) -> Self:
        return cls(
            np.random.randn(dims[0], dims[1]),
            np.random.randn(dims[1]),
        )

    def __init__(
        self,
        W: np.ndarray,
        b: np.ndarray,
    ):
        self._W = [W]
        self._b = [b]

    def _forward_prop(
        self, X: np.ndarray
    ) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
        Z = X @ self._W[0] + self._b[0]
        return ((Z,), (Z,))

    def _backward_prop_and_update(
        self,
        X: np.ndarray,
        Y_true: np.ndarray,
        Z: tuple[np.ndarray, ...],
        A: tuple[np.ndarray, ...],
        learning_rate: float,
    ) -> None:
        del A  # unused
        Y_pred = softmax(Z[0])
        k = 1 / X.shape[0]
        dZ = Y_pred - Y_true
        dW = k * X.T @ dZ
        db = k * np.sum(dZ)
        if np.isnan(dW).any() or np.isnan(db).any():
            raise ValueError("dW or db is NaN Aborting training.")
        self._W[0] -= learning_rate * dW
        self._b[0] -= learning_rate * db

    def dump(self, io: IO[bytes]) -> None:
        pickle.dump(self.Serialized(W=self._W[0], b=self._b[0]), io)

    @classmethod
    def load(cls, source: IO[bytes] | Serialized) -> Self:
        if isinstance(source, cls.Serialized):
            return cls(W=source.W, b=source.b)
        data = pickle.load(source)
        if data._tag != cls.Serialized._tag:
            raise ValueError(f"Invalid model type: {data._tag}")
        return cls(W=data.W, b=data.b)


class MultilayerPerceptron(ModelBase):
    @dataclass(frozen=True, kw_only=True)
    class Serialized:
        _tag: ClassVar[str] = "mlp_relu"
        W: tuple[np.ndarray, ...]
        b: tuple[np.ndarray, ...]

    @classmethod
    def get_description(cls) -> str:
        return "Multilayer Perceptron with ReLU activation"

    @classmethod
    def initialize(cls, *dims: int) -> Self:
        return cls(
            [np.random.randn(dim_in, dim_out) for dim_in, dim_out in pairwise(dims)],
            [np.random.randn(dim_out) for dim_out in dims[1:]],
        )

    def __init__(
        self,
        W: Sequence[np.ndarray],
        b: Sequence[np.ndarray],
    ):
        self._W = W
        self._b = b

    def _forward_prop(
        self, X: np.ndarray
    ) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
        Z = [X @ self._W[0] + self._b[0]]
        for w, b in zip(self._W[1:], self._b[1:]):
            Z.append(ReLU(Z[-1]) @ w + b)
        return tuple(Z), tuple(map(ReLU, Z))

    def _backward_prop_and_update(
        self,
        X: np.ndarray,
        Y_true: np.ndarray,
        Z: Sequence[np.ndarray],
        A: Sequence[np.ndarray],
        learning_rate: float,
    ) -> None:
        Y_pred = softmax(Z[-1])
        dZ = Y_pred - Y_true
        _A = [X, *A]
        k = 1 / X.shape[0]

        for idx in range(len(self._W) - 1, -1, -1):
            dW = k * (_A[idx].T @ dZ)
            db = k * np.sum(dZ, axis=0)
            if np.isnan(dW).any() or np.isnan(db).any():
                raise ValueError("dW or db is NaN Aborting training.")
            self._W[idx] -= learning_rate * dW
            self._b[idx] -= learning_rate * db
            if idx > 0:
                dZ = (dZ @ self._W[idx].T) * deriv_ReLU(Z[idx - 1])

    def dump(self, io: IO[bytes]) -> None:
        pickle.dump(self.Serialized(W=tuple(self._W), b=tuple(self._b)), io)

    @classmethod
    def load(cls, source: IO[bytes] | Serialized) -> Self:
        if isinstance(source, cls.Serialized):
            return cls(W=source.W, b=source.b)
        data = pickle.load(source)
        if data._tag != cls.Serialized._tag:
            raise ValueError(f"Invalid model type: {data._tag}")
        return cls(W=data.W, b=data.b)


def load_model(model_path: Path) -> ModelBase:
    serialized = pickle.load(open(model_path, "rb"))
    match serialized._tag:
        case LinearRegressionModel.Serialized._tag:
            logger.info(
                f"Loading {LinearRegressionModel.get_description()} from {model_path}."
            )
            return LinearRegressionModel.load(serialized)
        case MultilayerPerceptron.Serialized._tag:
            logger.info(
                f"Loading {MultilayerPerceptron.get_description()} from {model_path}."
            )
            return MultilayerPerceptron.load(serialized)
        case _:
            raise ValueError(f"Invalid model type: {serialized._tag}")
