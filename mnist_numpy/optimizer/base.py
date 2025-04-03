from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

from mnist_numpy.model.base import ModelT

ConfigT = TypeVar("ConfigT")


class OptimizerBase(ABC, Generic[ModelT, ConfigT]):
    def __init__(self, config: ConfigT):
        self._config = config

    @abstractmethod
    def update(
        self, model: ModelT, X_train_batch: np.ndarray, Y_train_batch: np.ndarray
    ) -> None: ...

    @abstractmethod
    def report(self) -> str: ...

    @property
    @abstractmethod
    def learning_rate(self) -> float: ...


@dataclass(frozen=True, kw_only=True)
class NoConfig:
    learning_rate: float


class NoOptimizer(OptimizerBase[ModelT, NoConfig]):
    Config = NoConfig

    def update(
        self, model: ModelT, X_train_batch: np.ndarray, Y_train_batch: np.ndarray
    ) -> None:
        A_train_batch, Z_train_batch = model._forward_prop(X_train_batch)
        gradient = model._backward_prop(
            X_train_batch, Y_train_batch, Z_train_batch, A_train_batch
        )
        model.update_parameters(-self._config.learning_rate * gradient)

    def report(self) -> str:
        return ""

    @property
    def learning_rate(self) -> float:
        return self._config.learning_rate
