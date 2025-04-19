from collections import deque
from typing import Final

import numpy as np

from mnist_numpy.model.mlp import MultiLayerPerceptron
from mnist_numpy.trainer.exceptions import AbortTraining

# This value has been found empirically to be a good threshold for exploding
# gradients. Obviously, a Z score of 30 is an insanely high value, but it can be
# understood as recognising that the weights are not being modelled by a normal
# distribution since the likelihood of a Z score of 30 is vanishingly small for
# a random variable truly normally distributed is 1 - erf(30) which is
# approximately 0.
MAX_Z_SCORE: Final[float] = 30.0


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
        self._running_weights_sums = None
        self._running_weights_sums_squared = None
        self._warmup_batches = warmup_batches

    def post_batch(
        self,
        gradient: MultiLayerPerceptron.Gradient,
        update: MultiLayerPerceptron.Gradient,
    ) -> None:
        del update
        sums_with_sums_squared = np.array(
            [(np.sum(param._W), np.sum(param._W**2)) for param in gradient.dParams]
        )
        self._running_update_count += 1
        if self._running_weights_sums is None:
            self._running_weights_sums, self._running_weights_sums_squared = map(
                np.array, zip(*sums_with_sums_squared)
            )
        else:
            running_weights_sums, running_weights_sums_squared = map(
                np.array, zip(*sums_with_sums_squared)
            )
            self._running_weights_sums = self._running_weights_sums * (
                1 - 1 / self._running_update_count
            ) + running_weights_sums * (1 / self._running_update_count)
            self._running_weights_sums_squared = self._running_weights_sums_squared * (
                1 - 1 / self._running_update_count
            ) + running_weights_sums_squared * (1 / self._running_update_count)

        ns = np.array([param._W.size for param in gradient.dParams])
        means = self._running_weights_sums / ns
        variances = self._running_weights_sums_squared / ns - means**2
        weights = [param._W for param in gradient.dParams]

        max_Z_scores = np.array(
            [
                np.max((weights - mean) / np.sqrt(variance))
                for (weights, mean, variance) in zip(weights, means, variances)
            ]
        )
        if self._running_update_count > self._warmup_batches:
            if (max_Z_score := np.max(max_Z_scores)) > MAX_Z_SCORE:
                raise AbortTraining(f"Exploding gradients detected. {max_Z_score=}")

    def post_epoch(self, L: float) -> None:
        self._L_history.append(L)
        if len(self._L_history) < self._L_history_max_len:
            return
        if np.polyfit(range(len(self._L_history)), self._L_history, 1)[0] > 0:
            raise AbortTraining("Model is diverging.")
