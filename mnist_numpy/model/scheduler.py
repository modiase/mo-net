import math
from collections.abc import MutableSequence
from typing import Protocol

import numpy as np

from mnist_numpy.model.mlp import MLP_Gradient


class Scheduler(Protocol):
    def __call__(
        self,
        current_iteration: int,
        current_learning_rate: float,
        gradient: MLP_Gradient,
    ) -> float: ...


class NoopScheduler:
    def __call__(self, _: object, current_learning_rate: float, __: object) -> float:
        return current_learning_rate


class CosineScheduler:
    def __init__(
        self,
        *,
        batch_size: int,
        learning_rate_rescale_factor: float,
        train_set_size: int,
    ):
        self._cosines: MutableSequence[float] = []
        self._iterations_per_batch = math.ceil(train_set_size / batch_size)
        self._learning_rate_rescale_factor = learning_rate_rescale_factor
        self._previous_gradient: MLP_Gradient | None = None

    def __call__(
        self,
        current_iteration: int,
        current_learning_rate: float,
        gradient: MLP_Gradient,
    ):
        if self._previous_gradient is None:
            self._previous_gradient = gradient
            return current_learning_rate

        if not np.isnan(
            cosine_distance := gradient.cosine_distance(self._previous_gradient)
        ):
            self._cosines.append(cosine_distance)

        if current_iteration % self._iterations_per_batch == 0:
            if np.average(self._cosines) > 0.1:
                self._cosines.clear()
                return current_learning_rate / self._learning_rate_rescale_factor
            self._cosines.clear()
        return current_learning_rate
