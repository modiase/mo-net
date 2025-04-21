from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Final, Self, Sequence

import numpy as np

from mnist_numpy.model.layer import DenseParameters
from mnist_numpy.model.mlp import MultiLayerPerceptron
from mnist_numpy.trainer.exceptions import AbortTraining

# This value has been found empirically to be a good threshold for exploding
# gradients. Obviously, a Z score of 20 is an insanely high value, but it can be
# understood as recognising that the weights are not being modelled by a normal
# distribution since the likelihood of a Z score of 20 a random variable truly
# normally distributed is 1 - erf(20) which is approximately 0.
MAX_Z_SCORE: Final[float] = 20.0


@dataclass(frozen=True, kw_only=True)
class WeightGradientRunningAverages:
    sums: np.ndarray
    sums_of_squares: np.ndarray

    @classmethod
    def from_weights(cls, weights_seq: Sequence[np.ndarray]) -> Self:
        return cls(
            sums=np.array([np.sum(weights) for weights in weights_seq]),
            sums_of_squares=np.array([np.sum(weights**2) for weights in weights_seq]),
        )

    @classmethod
    def from_weights_and_update(
        cls,
        running_average: WeightGradientRunningAverages,
        update: WeightGradientRunningAverages,
        update_count: int,
    ) -> Self:
        r = 1 / update_count
        return cls(
            sums=running_average.sums * (1 - r) + update.sums * r,
            sums_of_squares=running_average.sums_of_squares * (1 - r)
            + update.sums_of_squares * r,
        )

    @classmethod
    def none(cls) -> Self:
        return cls(sums=np.zeros(1), sums_of_squares=np.zeros(1))


class Monitor:
    def __init__(
        self,
        *,
        low_gradient_abort_threshold: float,
        high_gradient_abort_threshold: float,
        history_max_len: int,
        warmup_batches: int,
        X_train: np.ndarray,
        Y_train: np.ndarray,
    ):
        self._L_history_max_len = history_max_len
        self._L_history: deque[float] = deque(maxlen=self._L_history_max_len)
        self._X_train = X_train
        self._Y_train = Y_train
        self._high_gradient_abort_threshold = high_gradient_abort_threshold
        self._low_gradient_abort_threshold = low_gradient_abort_threshold
        self._running_update_count = 0
        self._running_weights: WeightGradientRunningAverages = (
            WeightGradientRunningAverages.none()
        )
        self._warmup_batches = warmup_batches

    def post_batch(
        self,
        gradient: MultiLayerPerceptron.Gradient,
        update: MultiLayerPerceptron.Gradient,
    ) -> None:
        del update  # unused
        dense_layer_gradients = [
            param for param in gradient.dParams if isinstance(param, DenseParameters)
        ]
        self._running_update_count += 1
        self._running_weights = WeightGradientRunningAverages.from_weights_and_update(
            self._running_weights,
            WeightGradientRunningAverages.from_weights(
                tuple(param._W for param in dense_layer_gradients)
            ),
            self._running_update_count,
        )
        ns = np.array([param._W.size for param in dense_layer_gradients])
        means = self._running_weights.sums / ns
        variances = self._running_weights.sums_of_squares / ns - means**2

        weight_gradients_max_Z_scores = np.array(
            [
                np.max((weights - mean) / np.sqrt(variance))
                for (weights, mean, variance) in zip(
                    [param._W for param in dense_layer_gradients], means, variances
                )
            ]
        )
        if self._running_update_count > self._warmup_batches:
            if (
                weight_gradients_max_Z_score := np.max(weight_gradients_max_Z_scores)
            ) > MAX_Z_SCORE:
                raise AbortTraining(
                    f"Exploding gradients detected. {weight_gradients_max_Z_score=}"
                )

    def post_epoch(self, L: float) -> None:
        self._L_history.append(L)
        if len(self._L_history) < self._L_history_max_len:
            return
        if np.polyfit(range(len(self._L_history)), self._L_history, 1)[0] > 0.001:
            raise AbortTraining("Model is diverging.")
