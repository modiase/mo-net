import math
from typing import Protocol, Self

from mnist_numpy.config import TrainingParameters
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
    @classmethod
    def of(cls, *, training_parameters: TrainingParameters) -> Self:
        return cls(
            batch_size=training_parameters.batch_size,
            learning_rate_limits=training_parameters.learning_rate_limits,
            learning_rate_rescale_factor_per_epoch=training_parameters.learning_rate_rescale_factor_per_epoch,
            train_set_size=training_parameters.train_set_size,
        )

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
    @classmethod
    def of(cls, *, training_parameters: TrainingParameters) -> Self:
        return cls(
            batch_size=training_parameters.batch_size,
            num_epochs=training_parameters.num_epochs,
            train_set_size=training_parameters.train_set_size,
            start_learning_rate=training_parameters.learning_rate_limits[1],
            learning_rate_limits=training_parameters.learning_rate_limits,
        )

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
        self._current_iteration = 0
        self._start_learning_rate = start_learning_rate
        self._current_learning_rate = start_learning_rate
        self._learning_rate_limits = (
            learning_rate_limits
            if learning_rate_limits is not None
            else (-float("inf"), float("inf"))
        )

    @property
    def _current_epoch(self) -> int:
        return self._current_iteration // self._batches_per_epoch

    def __call__(
        self,
        current_iteration: int,
        current_learning_rate: float,
        gradient: MultiLayerPerceptron.Gradient,
    ) -> float:
        del current_learning_rate, gradient  # unused
        self._current_iteration = current_iteration
        if current_iteration % self._batches_per_epoch == 0:
            self._current_learning_rate = _apply_limits(
                self._start_learning_rate
                * (1 + math.cos(self._current_epoch / self._num_epochs * math.pi))
                / 2,
                self._learning_rate_limits,
            )

        return self._current_learning_rate


class WarmupScheduler:
    @classmethod
    def of(
        cls,
        *,
        training_parameters: TrainingParameters,
        next_scheduler: Scheduler,
    ) -> Self:
        return cls(
            batch_size=training_parameters.batch_size,
            num_epochs=training_parameters.num_epochs,
            train_set_size=training_parameters.train_set_size,
            warmup_epochs=training_parameters.warmup_epochs,
            learning_rate_limits=training_parameters.learning_rate_limits,
            next_scheduler=next_scheduler,
        )

    def __init__(
        self,
        *,
        batch_size: int,
        num_epochs: int,
        train_set_size: int,
        warmup_epochs: int,
        learning_rate_limits: tuple[float, float],
        next_scheduler: Scheduler,
    ):
        self._num_epochs = num_epochs
        self._batches_per_epoch = math.ceil(train_set_size / batch_size)
        self._warmup_epochs = warmup_epochs
        self._start_learning_rate, self._end_learning_rate = learning_rate_limits
        self._current_learning_rate = self._start_learning_rate
        self._current_iteration = 0
        self._next_scheduler = next_scheduler

    @property
    def _current_epoch(self) -> int:
        return self._current_iteration // self._batches_per_epoch

    def __call__(
        self,
        current_iteration: int,
        current_learning_rate: float,
        gradient: MultiLayerPerceptron.Gradient,
    ) -> float:
        self._current_iteration = current_iteration
        if self._current_epoch > self._warmup_epochs:
            return self._next_scheduler(
                current_iteration, current_learning_rate, gradient
            )
        if current_iteration % self._batches_per_epoch == 0:
            self._current_learning_rate = (
                self._end_learning_rate - self._start_learning_rate
            ) * (self._current_epoch / self._warmup_epochs)

        return self._current_learning_rate
