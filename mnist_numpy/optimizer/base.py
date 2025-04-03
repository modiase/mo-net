from abc import ABC, abstractmethod
from typing import Generic

import numpy as np

from mnist_numpy.model.base import ModelT


class OptimizerBase(ABC, Generic[ModelT]):
    @abstractmethod
    def update(
        self, model: ModelT, X_train_batch: np.ndarray, Y_train_batch: np.ndarray
    ) -> None: ...

    @abstractmethod
    def report(self) -> str: ...

    @property
    @abstractmethod
    def learning_rate(self) -> float: ...


class NoOptimizer(OptimizerBase[ModelT]):
    def update(
        self, model: ModelT, X_train_batch: np.ndarray, Y_train_batch: np.ndarray
    ) -> None:
        A_train_batch, Z_train_batch = model._forward_prop(X_train_batch)
        gradient = model._backward_prop(
            X_train_batch, Y_train_batch, Z_train_batch, A_train_batch
        )
        model.update_parameters(-self._learning_rate * gradient)

    def report(self) -> str:
        return ""

    @property
    def learning_rate(self) -> float:
        return self._learning_rate
