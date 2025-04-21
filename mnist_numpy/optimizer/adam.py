from dataclasses import dataclass, field
from typing import Final, Literal, overload

import numpy as np

from mnist_numpy.model.base import ModelT
from mnist_numpy.model.mlp import MultiLayerPerceptron
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

        self._current_learning_rate: float = 0.0
        self._first_moment = model.empty_gradient()
        self._iterations = 0
        self._scheduler = config.scheduler
        self._second_moment = model.empty_gradient()

    @overload
    def _training_step(
        self,
        model: ModelT,
        X_train_batch: np.ndarray,
        Y_train_batch: np.ndarray,
        do_update: Literal[False],
    ) -> tuple[MultiLayerPerceptron.Gradient, None]: ...

    @overload
    def _training_step(
        self,
        model: ModelT,
        X_train_batch: np.ndarray,
        Y_train_batch: np.ndarray,
        do_update: Literal[True],
    ) -> tuple[MultiLayerPerceptron.Gradient, MultiLayerPerceptron.Gradient]: ...

    def _training_step(
        self,
        model: ModelT,
        X_train_batch: np.ndarray,
        Y_train_batch: np.ndarray,
        do_update: bool = True,
    ) -> tuple[MultiLayerPerceptron.Gradient, MultiLayerPerceptron.Gradient]:
        model.forward_prop(X=X_train_batch)
        gradient = model.backward_prop(Y_true=Y_train_batch)
        if do_update:
            update = self.compute_update(gradient)
            model.update_parameters(update)
            return gradient, update
        return gradient, None

    def compute_update(
        self, gradient: MultiLayerPerceptron.Gradient
    ) -> MultiLayerPerceptron.Gradient:
        self._iterations += 1
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
        return -self.alpha_t * (
            self._first_moment / (self._second_moment**0.5 + self._config.epsilon)
        )

    @property
    def alpha_t(self) -> float:
        return self._current_learning_rate * (
            (1 - self._config.beta_2**self._iterations) ** 0.5
            / (1 - self._config.beta_1**self._iterations)
        )

    @property
    def learning_rate(self) -> float:
        return self._current_learning_rate

    def report(self) -> str:
        return f"Learning Rate: {self._current_learning_rate:.10f}"
