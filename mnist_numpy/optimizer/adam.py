from dataclasses import dataclass, field
from typing import Final

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

        self._current_learning_rate = config.learning_rate
        self._first_moment = model.empty_gradient()
        self._iterations = 0
        self._scheduler = config.scheduler
        self._second_moment = model.empty_gradient()

    def _training_step(
        self,
        model: ModelT,
        X_train_batch: np.ndarray,
        Y_train_batch: np.ndarray,
        do_update: bool,
    ) -> tuple[MultiLayerPerceptron.Gradient, MultiLayerPerceptron.Gradient]:
        self._iterations += 1
        model.forward_prop(X=X_train_batch)
        gradient = model.backward_prop(Y_true=Y_train_batch)
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
        update = -self.alpha_t * (
            self._first_moment / (self._second_moment**0.5 + self._config.epsilon)
        )
        if do_update:
            model.update_parameters(update)
        return gradient, update

    @property
    def alpha_t(self) -> float:
        return self._config.learning_rate * (
            (1 - self._config.beta_2**self._iterations) ** 0.5
            / (1 - self._config.beta_1**self._iterations)
        )

    @property
    def learning_rate(self) -> float:
        return self._current_learning_rate

    def report(self) -> str:
        return f"Learning Rate: {self._current_learning_rate:.10f}"
