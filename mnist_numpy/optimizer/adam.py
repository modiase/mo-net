from typing import Final

import numpy as np

from mnist_numpy.model.base import ModelT
from mnist_numpy.model.mlp import MLP_Gradient
from mnist_numpy.optimizer.base import OptimizerBase

DEFAULT_BETA_1: Final[float] = 0.9
DEFAULT_BETA_2: Final[float] = 0.999
DEFAULT_EPSILON: Final[float] = 1e-8


class AdamOptimizer(OptimizerBase[ModelT]):
    def __init__(
        self,
        *,
        model: ModelT,
        beta_1: float = DEFAULT_BETA_1,
        beta_2: float = DEFAULT_BETA_2,
        epsilon: float = DEFAULT_EPSILON,
        learning_rate: float,
    ):
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._epsilon = epsilon
        self._first_moment = model.empty_gradient()
        self._learning_rate = learning_rate
        self._second_moment = model.empty_gradient()
        self._iterations = 0

    def update(
        self,
        model: ModelT,
        X_train_batch: np.ndarray,
        Y_train_batch: np.ndarray,
    ) -> None:
        self._iterations += 1
        Z_train_batch, A_train_batch = model._forward_prop(X_train_batch)
        gradient = model._backward_prop(
            X_train_batch,
            Y_train_batch,
            Z_train_batch,
            A_train_batch,
        )
        self._first_moment = (
            self._beta_1 * self._first_moment + (1 - self._beta_1) * gradient
        )
        self._second_moment = (
            self._beta_2 * self._second_moment + (1 - self._beta_2) * gradient**2
        )
        first_moment_corrected = self._first_moment / (
            1 - self._beta_1**self._iterations
        )
        second_moment_corrected = self._second_moment / (
            1 - self._beta_2**self._iterations
        )

        update = MLP_Gradient(
            dWs=tuple(
                -self._learning_rate
                * first_moment_corrected
                / (np.sqrt(second_moment_corrected) + self._epsilon)
                for first_moment_corrected, second_moment_corrected in zip(
                    first_moment_corrected.dWs, second_moment_corrected.dWs
                )
            ),
            dbs=tuple(
                -self._learning_rate
                * first_moment_corrected
                / (np.sqrt(second_moment_corrected) + self._epsilon)
                for first_moment_corrected, second_moment_corrected in zip(
                    first_moment_corrected.dbs, second_moment_corrected.dbs
                )
            ),
        )
        model.update_parameters(update)

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    def report(self) -> str:
        return f"Learning Rate: {self._learning_rate:.10f}"
