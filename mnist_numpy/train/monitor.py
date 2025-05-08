from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Final, Self, Sequence

import numpy as np

from mnist_numpy.model.layer.linear import Parameters
from mnist_numpy.protos import RawGradientType, UpdateGradientType
from mnist_numpy.train.exceptions import AbortTraining

# This value has been found empirically to be a good threshold for exploding
# gradients. Obviously, a Z score of 50 is an insanely high value, but it can be
# understood as recognising that the weights are not being modelled by a normal
# distribution since the likelihood of a Z score of 50 a random variable truly
# normally distributed is 1 - erf(50) which is approximately 0.
EPSILON: Final[float] = 1e-8
MAX_Z_SCORE_UPPER_BOUND: Final[float] = 50.0
MAX_Z_SCORE_LOWER_BOUND: Final[float] = 20.0


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
        batches_per_epoch: int,
        history_max_len: int,
        warmup_epochs: int,
        X_train: np.ndarray,
        Y_train: np.ndarray,
    ):
        self._L_history_max_len = history_max_len
        self._L_history: deque[float] = deque(maxlen=self._L_history_max_len)
        self._L_history_snapshot: Sequence[float] = ()
        self._X_train = X_train
        self._Y_train = Y_train
        self._running_update_count = 0
        self._running_weights: WeightGradientRunningAverages = (
            WeightGradientRunningAverages.none()
        )
        self._warmup_batches = warmup_epochs * batches_per_epoch

    def reset(self, restore_history: bool = False) -> None:
        if restore_history:
            self._L_history = deque(
                self._L_history_snapshot, maxlen=self._L_history_max_len
            )
        else:
            self._L_history.clear()
        self._running_update_count = 0
        self._running_weights = WeightGradientRunningAverages.none()

    def post_batch(
        self,
        raw_gradient: RawGradientType,
        update: UpdateGradientType,
    ) -> None:
        del update  # unused
        linear_layer_gradients = [
            gradient for gradient in raw_gradient if isinstance(gradient, Parameters)
        ]
        self._running_update_count += 1
        self._running_weights = WeightGradientRunningAverages.from_weights_and_update(
            self._running_weights,
            WeightGradientRunningAverages.from_weights(
                tuple(param._W for param in linear_layer_gradients)
            ),
            self._running_update_count,
        )
        ns = np.array([param._W.size for param in linear_layer_gradients])
        means = self._running_weights.sums / ns
        variances = self._running_weights.sums_of_squares / ns - means**2

        weight_gradients_max_Z_scores = np.array(
            [
                np.max((weights - mean) / (np.sqrt(variance) + EPSILON))
                for (weights, mean, variance) in zip(
                    [param._W for param in linear_layer_gradients], means, variances
                )
            ]
        )
        if self._running_update_count > self._warmup_batches:
            if (
                weight_gradients_max_Z_score := np.max(weight_gradients_max_Z_scores)
            ) > max(
                MAX_Z_SCORE_UPPER_BOUND / np.log(np.log(self._running_update_count)),
                MAX_Z_SCORE_LOWER_BOUND,
            ):
                raise AbortTraining(
                    f"Exploding gradients detected. {weight_gradients_max_Z_score=}"
                )

    def post_epoch(self, L: float) -> None:
        self._L_history.append(L)
        if len(self._L_history) < self._L_history_max_len:
            return
        if np.polyfit(range(len(self._L_history)), self._L_history, 1)[0] >= 0:
            raise AbortTraining("Model is not learning.")

    def clear_history(self) -> None:
        self._L_history.clear()
        self._L_history_snapshot = ()
