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
    def training_step(
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

    def training_step(
        self, model: ModelT, X_train_batch: np.ndarray, Y_train_batch: np.ndarray
    ) -> None:
        model.forward_prop(X=X_train_batch)
        gradient = model.backward_prop(Y_true=Y_train_batch)
        model.update_parameters(update=-self._config.learning_rate * gradient)

    def report(self) -> str:
        return ""

    @property
    def learning_rate(self) -> float:
        return self._config.learning_rate
