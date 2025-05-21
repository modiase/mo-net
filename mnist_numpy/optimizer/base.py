from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Literal, Protocol, Sequence, TypeVar, overload

import numpy as np

from mnist_numpy.model.model import Model
from mnist_numpy.protos import SupportsGradientOperations

ConfigT = TypeVar("ConfigT")


class AfterComputeUpdateHandler(Protocol):
    def __call__(self, learning_rate: float) -> None: ...


class Base(ABC, Generic[ConfigT]):
    def __init__(self, *, model: Model, config: ConfigT):
        self._model = model
        self._config = config
        self._iterations = 0
        self._after_compute_update_handlers: Sequence[AfterComputeUpdateHandler] = ()
        self._learning_rate_snapshot_iterations: int | None = None

    @overload
    def training_step(
        self,
        X_train_batch: np.ndarray,
        Y_train_batch: np.ndarray,
        return_gradients: Literal[False] = False,
    ) -> None: ...

    @overload
    def training_step(
        self,
        X_train_batch: np.ndarray,
        Y_train_batch: np.ndarray,
        return_gradients: Literal[True],
    ) -> tuple[
        Sequence[SupportsGradientOperations], Sequence[SupportsGradientOperations]
    ]: ...

    def training_step(
        self,
        X_train_batch: np.ndarray,
        Y_train_batch: np.ndarray,
        return_gradients: bool = False,
    ) -> (
        tuple[
            Sequence[SupportsGradientOperations], Sequence[SupportsGradientOperations]
        ]
        | None
    ):
        self._model.forward_prop(X=X_train_batch)
        self._model.backward_prop(Y_true=Y_train_batch)
        if return_gradients:
            gradient = self._model.get_gradient_caches()
        self.compute_update()
        for handler in self._after_compute_update_handlers:
            handler(self.learning_rate)
        if return_gradients:
            update = self._model.get_gradient_caches()
            self._model.update_parameters()
            return gradient, update
        self._model.update_parameters()
        return None

    @abstractmethod
    def compute_update(self) -> None: ...

    @abstractmethod
    def report(self) -> str: ...

    @property
    @abstractmethod
    def learning_rate(self) -> float: ...

    def set_model(self, model: Model) -> None:
        self._model = model

    def snapshot(self) -> None:
        self._learning_rate_snapshot_iterations = self._iterations

    def restore(self) -> None:
        if self._learning_rate_snapshot_iterations is None:
            raise ValueError("No snapshot to restore.")
        self._iterations = self._learning_rate_snapshot_iterations

    @property
    def config(self) -> ConfigT:
        return self._config

    def register_after_compute_update_handler(
        self, handler: AfterComputeUpdateHandler
    ) -> None:
        self._after_compute_update_handlers = tuple(
            [*self._after_compute_update_handlers, handler]
        )


@dataclass(frozen=True, kw_only=True)
class Config:
    learning_rate: float


class Null(Base[Config]):
    Config = Config

    def compute_update(self) -> None:
        for layer in self._model.grad_layers:
            layer.cache["dP"] *= -self.learning_rate

    def report(self) -> str:
        return ""

    @property
    def learning_rate(self) -> float:
        return self._config.learning_rate
