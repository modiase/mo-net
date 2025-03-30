import pickle
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import cycle
from pathlib import Path
from typing import IO, ClassVar, Final, Self

import numpy as np
import pandas as pd
from loguru import logger
from more_itertools import pairwise
from tqdm import tqdm

DEFAULT_BATCH_SIZE: Final[int] = 1000
DEFAULT_LEARNING_RATE_RESCALE_FACTOR: Final[float] = 0.00001
DEFAULT_LEARNING_RATE: Final[float] = 0.001
DEFAULT_MOMENTUM_PARAMETER: Final[float] = 0.9
DEFAULT_NUM_ITERATIONS: Final[int] = 1000000
DEFAULT_TRAINING_LOG_MIN_INTERVAL_SECONDS: Final[int] = 30
MAX_HISTORY_LENGTH: Final[int] = 2


def softmax(x: np.ndarray) -> np.ndarray:
    return (exp_x := np.exp(x - np.max(x, axis=1, keepdims=True))) / np.sum(
        exp_x, axis=1, keepdims=True
    )


def cross_entropy(Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
    Y_pred = np.clip(Y_pred, 1e-15, 1 - 1e-15)
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

    @abstractmethod
    def get_name(cls) -> str: ...

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
    def _update_weights(
        self,
        dWs: Sequence[np.ndarray],
        dbs: Sequence[np.ndarray],
        learning_rate: float,
        momentum_parameter: float,
    ) -> None: ...

    @abstractmethod
    def _undo_update(self) -> None: ...

    @abstractmethod
    def _backward_prop(
        self,
        X: np.ndarray,
        Y_true: np.ndarray,
        Z: tuple[np.ndarray, ...],
        A: tuple[np.ndarray, ...],
    ) -> tuple[Sequence[np.ndarray], Sequence[np.ndarray]]: ...

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
        batch_size: int | None = None,
        learning_rate_rescale_factor: float,
        momentum_parameter: float,
        training_log_path: Path,
    ) -> Path:
        if not training_log_path.exists():
            training_log = pd.DataFrame(
                columns=[
                    "timestamp",
                    "iteration",
                    "training_loss",
                    "test_loss",
                    "learning_rate",
                ]
            )
            training_log.to_csv(training_log_path, index=False)
        else:
            training_log = pd.read_csv(training_log_path)

        logger.info(
            f"Training model {self.__class__.__name__} for {num_iterations=} iterations with {learning_rate=}."
        )

        train_set_size = X_train.shape[0]
        if batch_size is None:
            batch_size = train_set_size

        X_train_batched = cycle(np.array_split(X_train, train_set_size // batch_size))
        Y_train_batched = cycle(np.array_split(Y_train, train_set_size // batch_size))

        model_checkpoint_path = training_log_path.with_name(
            training_log_path.name.replace("training_log.csv", "partial.pkl")
        )

        logger.info(
            f"Training model..\nSaving partial results to: {model_checkpoint_path}."
        )
        logger.info(f"\n{training_log_path=}.")
        self.dump(open(model_checkpoint_path, "wb"))

        current_learning_rate = learning_rate
        start_iteration = total_iterations - num_iterations

        k_train = 1 / train_set_size

        test_set_size = X_test.shape[0]
        k_test = 1 / test_set_size
        k_batch = 1 / batch_size

        L_train_min = k_train * cross_entropy(
            softmax(self._forward_prop(X_train)[1][-1]), Y_train
        )
        logger.info(f"Initial training loss: {L_train_min}.")

        for i in tqdm(
            range(start_iteration, total_iterations),
            initial=start_iteration,
            total=total_iterations,
        ):
            X_train_batch = next(X_train_batched)
            Y_train_batch = next(Y_train_batched)

            Z_train_batch, A_train_batch = self._forward_prop(X_train_batch)
            L_batch_before = k_batch * cross_entropy(
                softmax(A_train_batch[-1]), Y_train_batch
            )
            dW, db = self._backward_prop(
                X_train_batch,
                Y_train_batch,
                Z_train_batch,
                A_train_batch,
            )
            self._update_weights(dW, db, current_learning_rate, momentum_parameter)
            _, A_train_batch = self._forward_prop(X_train_batch)
            L_batch_after = k_batch * cross_entropy(
                softmax(A_train_batch[-1]), Y_train_batch
            )
            if L_batch_after < L_batch_before:
                current_learning_rate *= 1 + learning_rate_rescale_factor
            else:
                current_learning_rate *= 1 - 2 * learning_rate_rescale_factor

            if i % train_set_size == 0:
                permutation = np.random.permutation(train_set_size)
                X_train = X_train[permutation]
                Y_train = Y_train[permutation]

                X_train_batched = cycle(
                    np.array_split(X_train, train_set_size // batch_size)
                )
                Y_train_batched = cycle(
                    np.array_split(Y_train, train_set_size // batch_size)
                )
                _, A_train = self._forward_prop(X_train)
                L_train = k_train * cross_entropy(softmax(A_train[-1]), Y_train)
                _, A_test = self._forward_prop(X_test)
                L_test = k_test * cross_entropy(softmax(A_test[-1]), Y_test)
                tqdm.write(
                    f"Iteration {i}: Training Loss = {L_train}, Test Loss = {L_test}"
                    f" {current_learning_rate=}"
                )
                pd.DataFrame(
                    [[datetime.now(), i, L_train, L_test, current_learning_rate]],
                    columns=training_log.columns,
                ).to_csv(training_log_path, mode="a", header=False, index=False)
                if L_train < L_train_min:
                    self.dump(open(model_checkpoint_path, "wb"))
                    L_train_min = L_train

        return model_checkpoint_path


class MultilayerPerceptron(ModelBase):
    @dataclass(frozen=True, kw_only=True)
    class Serialized:
        _tag: ClassVar[str] = "mlp_relu"
        W: tuple[np.ndarray, ...]
        b: tuple[np.ndarray, ...]

    @property
    def layers(self) -> tuple[int, ...]:
        return tuple(w.shape[0] for w in self._W)

    def get_name(self) -> str:
        return f"mlp_relu_{'_'.join(str(layer) for layer in self.layers[1:])}"

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
        self._W = list(W)
        self._b = list(b)
        self._history = deque(
            ((tuple(np.zeros_like(w) for w in W), tuple(np.zeros_like(b) for b in b)),),
            maxlen=MAX_HISTORY_LENGTH,
        )

    def _forward_prop(
        self, X: np.ndarray
    ) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
        Z = [X @ self._W[0] + self._b[0]]
        for w, b in zip(self._W[1:], self._b[1:]):
            Z.append(ReLU(Z[-1]) @ w + b)
        return tuple(Z), tuple(map(ReLU, Z))

    def _backward_prop(
        self,
        X: np.ndarray,
        Y_true: np.ndarray,
        Z: Sequence[np.ndarray],
        A: Sequence[np.ndarray],
    ) -> tuple[Sequence[np.ndarray], Sequence[np.ndarray]]:
        Y_pred = softmax(Z[-1])
        dZ = Y_pred - Y_true
        _A = [X, *A]
        k = 1 / X.shape[0]

        dW = []
        db = []
        for idx in range(len(self._W) - 1, -1, -1):
            dW.append(k * (_A[idx].T @ dZ))
            db.append(k * np.sum(dZ, axis=0))
            if (
                np.isnan(dW[-1]).any()
                or np.isnan(db[-1]).any()
                or np.isnan(dZ[-1]).any()
            ):
                raise ValueError("Invalid gradient. Aborting training.")
            if idx > 0:
                dZ = (dZ @ self._W[idx].T) * deriv_ReLU(Z[idx - 1])

        return tuple(reversed(dW)), tuple(reversed(db))

    def _update_weights(
        self,
        dWs: Sequence[np.ndarray],
        dbs: Sequence[np.ndarray],
        learning_rate: float,
        momentum_parameter: float,
    ) -> None:
        (prev_dWs, prev_dbs) = self._history[-1]

        dWs_update = [
            learning_rate * (1 - momentum_parameter) * dW + momentum_parameter * prev_dW
            for prev_dW, dW in zip(prev_dWs, dWs)
        ]
        dbs_update = [
            learning_rate * (1 - momentum_parameter) * db + momentum_parameter * prev_db
            for prev_db, db in zip(prev_dbs, dbs)
        ]
        for w, dW in zip(self._W, dWs_update):
            w -= dW
        for b, db in zip(self._b, dbs_update):
            b -= db
        self._history.append((tuple(dWs_update), tuple(dbs_update)))

    def _undo_update(self) -> None:
        (dWs, dbs) = self._history.pop()
        for w, dW in zip(self._W, dWs):
            w += dW
        for b, db in zip(self._b, dbs):
            b += db

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
        case MultilayerPerceptron.Serialized._tag:
            logger.info(
                f"Loading {MultilayerPerceptron.get_description()} from {model_path}."
            )
            return MultilayerPerceptron.load(serialized)
        case _:
            raise ValueError(f"Invalid model type: {serialized._tag}")
