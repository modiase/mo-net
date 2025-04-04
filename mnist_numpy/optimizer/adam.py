from dataclasses import dataclass, field
from typing import Final

import numpy as np

from mnist_numpy.model.base import ModelT
from mnist_numpy.model.mlp import MLP_Gradient
from mnist_numpy.model.scheduler import NoopScheduler, Scheduler
from mnist_numpy.optimizer.base import OptimizerBase

DEFAULT_BETA_1: Final[float] = 0.9
DEFAULT_BETA_2: Final[float] = 0.999
DEFAULT_EPSILON: Final[float] = 1e-8


@dataclass(frozen=True, kw_only=True)
class AdamConfig:
    beta_1: float = DEFAULT_BETA_1
    beta_2: float = DEFAULT_BETA_2
    epsilon: float = DEFAULT_EPSILON
    learning_rate: float
    scheduler: Scheduler = field(default_factory=NoopScheduler)


class AdamOptimizer(OptimizerBase[ModelT, AdamConfig]):
    Config = AdamConfig

    def __init__(
        self,
        *,
        model: ModelT,
        config: AdamConfig,
    ):
        super().__init__(config)

        self._first_moment = model.empty_gradient()
        self._second_moment = model.empty_gradient()
        self._iterations = 0
        self._scheduler = config.scheduler
        self._current_learning_rate = config.learning_rate

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
        self._current_learning_rate = self._scheduler(
            self._iterations, self._current_learning_rate, gradient
        )
        self._first_moment = (
            self._config.beta_1 * self._first_moment
            + (1 - self._config.beta_1) * gradient
        )
        self._second_moment = (
            self._config.beta_2 * self._second_moment
            + (1 - self._config.beta_2) * gradient**2
        )
        first_moment_corrected = self._first_moment / (
            1 - self._config.beta_1**self._iterations
        )
        second_moment_corrected = self._second_moment / (
            1 - self._config.beta_2**self._iterations
        )

        update = MLP_Gradient(
            dWs=tuple(
                -self._config.learning_rate
                * first_moment_corrected
                / (np.sqrt(second_moment_corrected) + self._config.epsilon)
                for first_moment_corrected, second_moment_corrected in zip(
                    first_moment_corrected.dWs, second_moment_corrected.dWs
                )
            ),
            dbs=tuple(
                -self._config.learning_rate
                * first_moment_corrected
                / (np.sqrt(second_moment_corrected) + self._config.epsilon)
                for first_moment_corrected, second_moment_corrected in zip(
                    first_moment_corrected.dbs, second_moment_corrected.dbs
                )
            ),
        )
        model.update_parameters(update)

    @property
    def learning_rate(self) -> float:
        return self._current_learning_rate

    def report(self) -> str:
        return f"Learning Rate: {self._current_learning_rate:.10f}"
