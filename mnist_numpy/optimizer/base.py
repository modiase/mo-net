from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

from mnist_numpy.model.base import ModelT
from mnist_numpy.model.mlp import MultiLayerPerceptron

ConfigT = TypeVar("ConfigT")


class OptimizerBase(ABC, Generic[ModelT, ConfigT]):
    def __init__(self, config: ConfigT):
        self._config = config

    def training_step(
        self,
        model: ModelT,
        X_train_batch: np.ndarray,
        Y_train_batch: np.ndarray,
    ) -> MultiLayerPerceptron.Gradient:
        model.forward_prop(X=X_train_batch)
        return model.backward_prop(Y_true=Y_train_batch)

    @abstractmethod
    def compute_update(
        self,
        gradient: MultiLayerPerceptron.Gradient,
    ) -> MultiLayerPerceptron.Gradient: ...

    @abstractmethod
    def report(self) -> str: ...

    @property
    @abstractmethod
    def learning_rate(self) -> float: ...


@dataclass(frozen=True, kw_only=True)
class NoOptimizerConfig:
    learning_rate: float


class NoOptimizer(OptimizerBase[ModelT, NoOptimizerConfig]):
    Config = NoOptimizerConfig

    def compute_update(
        self,
        gradient: MultiLayerPerceptron.Gradient,
    ) -> MultiLayerPerceptron.Gradient:
        return -self._config.learning_rate * gradient

    def report(self) -> str:
        return ""

    @property
    def learning_rate(self) -> float:
        return self._config.learning_rate
