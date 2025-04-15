import math
from collections import deque
from dataclasses import dataclass
from typing import Final

import numpy as np

from mnist_numpy.functions import cross_entropy, softmax
from mnist_numpy.model.base import ModelT
from mnist_numpy.model.mlp import MultiLayerPerceptron
from mnist_numpy.optimizer.base import OptimizerBase

MAX_HISTORY_LENGTH: Final[int] = 2


@dataclass(frozen=True, kw_only=True)
class AdalmConfig:
    batch_size: int
    learning_rate: float
    learning_rate_limits: tuple[float, float]
    learning_rate_rescale_factor_per_epoch: float
    momentum_coefficient: float
    num_epochs: int
    train_set_size: int


class AdalmOptimizer(OptimizerBase[ModelT, AdalmConfig]):
    """
    Naive implementation of an optimizer with an adaptive learning rate
    (adal) and momentum (m).
    """

    Config = AdalmConfig

    def __init__(
        self,
        *,
        model: ModelT,
        config: AdalmConfig,
    ):
        super().__init__(config)

        self._iterations_per_epoch = config.train_set_size / config.batch_size
        self._learning_rate = config.learning_rate
        self._momentum_coefficient = config.momentum_coefficient
        self._min_momentum_coefficient = 0.0
        self._max_momentum_coefficient = config.momentum_coefficient
        self._min_learning_rate = config.learning_rate_limits[0]
        self._max_learning_rate = config.learning_rate_limits[1]
        self._learning_rate_decay_factor = math.exp(
            (math.log(self._min_learning_rate) - math.log(self._max_learning_rate))
            / config.num_epochs
        )
        self._learning_rate_rescale_factor = math.exp(
            math.log(config.learning_rate_rescale_factor_per_epoch)
            / self._iterations_per_epoch
        )
        self._k_batch = 1 / config.batch_size
        self._history = deque(
            (model.empty_gradient(),),
            maxlen=MAX_HISTORY_LENGTH,
        )
        self._iterations = 0

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    def _training_step(
        self,
        model: ModelT,
        X_train_batch: np.ndarray,
        Y_train_batch: np.ndarray,
    ) -> tuple[MultiLayerPerceptron.Gradient, MultiLayerPerceptron.Gradient]:
        L_batch_before = self._k_batch * cross_entropy(
            model.forward_prop(X=X_train_batch), Y_true=Y_train_batch
        )
        gradient = model.backward_prop(Y_true=Y_train_batch)

        prev_update = self._history[-1]

        update = model.Gradient(
            dWs=tuple(
                -(
                    self._learning_rate * (1 - self._momentum_coefficient) * dW
                    + self._momentum_coefficient * prev_dW
                )
                for prev_dW, dW in zip(prev_update.dWs, gradient.dWs)
            ),
            dbs=tuple(
                -(
                    self._learning_rate * (1 - self._momentum_coefficient) * db
                    + self._momentum_coefficient * prev_db
                )
                for prev_db, db in zip(prev_update.dbs, gradient.dbs)
            ),
        )
        model.update_parameters(update)
        self._history.append(update)

        A_train_batch = model.forward_prop(X=X_train_batch)
        L_batch_after = self._k_batch * cross_entropy(
            softmax(A_train_batch[-1]), Y_train_batch
        )
        if L_batch_after < L_batch_before:
            self._learning_rate *= 1 + self._learning_rate_rescale_factor
            self._momentum_coefficient += 0.05
        else:
            self._learning_rate *= 1 - 2 * self._learning_rate_rescale_factor
            self._momentum_coefficient -= 0.05

        self._momentum_coefficient = min(
            self._max_momentum_coefficient,
            max(
                self._momentum_coefficient,
                self._min_momentum_coefficient,
            ),
        )
        self._learning_rate = min(
            self._max_learning_rate,
            max(
                self._learning_rate,
                self._min_learning_rate,
            ),
        )
        self._iterations += 1
        if self._iterations % self._iterations_per_epoch == 0:
            self._max_learning_rate *= self._learning_rate_decay_factor
        return gradient, update

    def report(self) -> str:
        return (
            f"Learning Rate: {self._learning_rate:.10f}, Maximum Learning Rate: {self._max_learning_rate:.10f}"
            f", Momentum Parameter: {self._momentum_coefficient:.2f}"
        )
