import math
from collections import deque
from collections.abc import MutableSequence
from typing import Generic, Protocol, TypeVar

import numpy as np

from mnist_numpy.protocols import Gradient, HasCosineDistance

_GradientT_contra = TypeVar("_GradientT_contra", bound=Gradient, contravariant=True)


class Scheduler(Protocol, Generic[_GradientT_contra]):
    def __call__(
        self,
        current_iteration: int,
        current_learning_rate: float,
        gradient: _GradientT_contra,
    ) -> float: ...


class NoopScheduler:
    def __call__(
        self,
        current_iteration: object,
        current_learning_rate: float,
        gradient: Gradient,
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
        gradient: Gradient,
    ):
        del current_iteration, gradient  # unused
        return self._apply_limits(
            current_learning_rate / self._learning_rate_rescale_factor_per_batch
        )


class GradientWithCosineDistance(Gradient, HasCosineDistance): ...


class CosineScheduler:
    def __init__(
        self,
        *,
        batch_size: int,
        learning_rate_limits: tuple[float, float] | None = None,
        learning_rate_rescale_factor_per_epoch: float,
        train_set_size: int,
    ):
        self._batches_per_epoch = math.ceil(train_set_size / batch_size)
        self._cosine_distances: MutableSequence[float] = deque(
            maxlen=self._batches_per_epoch
        )
        self._learning_rate_rescale_factor_per_batch = math.exp(
            math.log(learning_rate_rescale_factor_per_epoch) / self._batches_per_epoch
        )
        self._previous_gradient: GradientWithCosineDistance | None = None
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
        gradient: GradientWithCosineDistance,
    ):
        if self._previous_gradient is None:
            self._previous_gradient = gradient
            return current_learning_rate

        if not np.isnan(
            cosine_distance := gradient.cosine_distance(self._previous_gradient)
        ):
            self._cosine_distances.append(cosine_distance)

        if current_iteration % self._batches_per_epoch == 0:
            if np.average(self._cosine_distances) > 0.1:
                return self._apply_limits(
                    current_learning_rate
                    / self._learning_rate_rescale_factor_per_batch,
                )
        return current_learning_rate
