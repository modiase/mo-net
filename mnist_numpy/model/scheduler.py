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


def _apply_limits(learning_rate: float, limits: tuple[float, float]) -> float:
    minimum, maximum = limits
    return max(min(learning_rate, maximum), minimum)


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

    def __call__(
        self,
        current_iteration: int,
        current_learning_rate: float,
        gradient: MultiLayerPerceptron.Gradient,
    ):
        del current_iteration, gradient  # unused
        return _apply_limits(
            current_learning_rate / self._learning_rate_rescale_factor_per_batch,
            self._learning_rate_limits,
        )


class CosineScheduler:
    def __init__(
        self,
        *,
        batch_size: int,
        num_epochs: int,
        train_set_size: int,
        start_learning_rate: float,
        learning_rate_limits: tuple[float, float] | None = None,
    ):
        self._num_epochs = num_epochs
        self._batches_per_epoch = math.ceil(train_set_size / batch_size)
        self._current_epoch = 0
        self._start_learning_rate = start_learning_rate
        self._current_learning_rate = start_learning_rate
        self._learning_rate_limits = (
            learning_rate_limits
            if learning_rate_limits is not None
            else (-float("inf"), float("inf"))
        )

    def __call__(
        self,
        current_iteration: int,
        current_learning_rate: float,
        gradient: MultiLayerPerceptron.Gradient,
    ) -> float:
        del current_learning_rate, gradient  # unused
        if current_iteration % self._batches_per_epoch == 0:
            self._current_epoch += 1
            self._current_learning_rate = _apply_limits(
                self._start_learning_rate
                * (1 + math.cos(self._current_epoch / self._num_epochs * math.pi))
                / 2,
                self._learning_rate_limits,
            )

        return self._current_learning_rate
