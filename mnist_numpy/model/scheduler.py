import math
from typing import Protocol

from mnist_numpy.model import MultiLayerPerceptron


class Scheduler(Protocol):
    def __call__(
        self,
        current_iteration: int,
        current_learning_rate: float,
        gradient: MultiLayerPerceptron.Gradient,
    ) -> float: ...


class NoopScheduler:
    def __call__(
        self,
        current_iteration: object,
        current_learning_rate: float,
        gradient: MultiLayerPerceptron.Gradient,
    ) -> float:
        del current_iteration, gradient  # unused
        return current_learning_rate


class DecayScheduler:
    def __init__(
        self,
        *,
        batch_size: int,
        learning_rate_limits: tuple[float, float] | None = None,
        learning_rate_rescale_factor_per_epoch: float,
        train_set_size: int,
    ):
        self._batches_per_epoch = math.ceil(train_set_size / batch_size)
        self._learning_rate_rescale_factor_per_batch = math.exp(
            math.log(learning_rate_rescale_factor_per_epoch) / self._batches_per_epoch
        )
        self._learning_rate_limits = (
            learning_rate_limits
            if learning_rate_limits is not None
            else (-float("inf"), float("inf"))
        )

    def _apply_limits(self, learning_rate: float) -> float:
        minimum, maximum = self._learning_rate_limits
        return max(min(learning_rate, maximum), minimum)

    def __call__(
        self,
        current_iteration: int,
        current_learning_rate: float,
        gradient: MultiLayerPerceptron.Gradient,
    ):
        del current_iteration, gradient  # unused
        return self._apply_limits(
            current_learning_rate / self._learning_rate_rescale_factor_per_batch
        )
