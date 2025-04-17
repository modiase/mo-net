from collections import deque
from itertools import pairwise

import numpy as np
from more_itertools import batched

from mnist_numpy.model.mlp import MultiLayerPerceptron
from mnist_numpy.monitor.exceptions import AbortTraining


class Monitor:
    def __init__(
        self,
        *,
        low_gradient_abort_threshold: float,
        high_gradient_abort_threshold: float,
        X_train: np.ndarray,
        Y_train: np.ndarray,
    ):
        self._low_gradient_abort_threshold = low_gradient_abort_threshold
        self._high_gradient_abort_threshold = high_gradient_abort_threshold
        self._X_train = X_train
        self._Y_train = Y_train
        self._L_history_max_len = 100
        self._L_history: deque[float] = deque(maxlen=self._L_history_max_len)

    def post_update(
        self,
        gradient: MultiLayerPerceptron.Gradient,
        update: MultiLayerPerceptron.Gradient,
    ) -> None:
        del update
        for param in gradient.dParams:
            if (val := np.max(np.abs(param._W))) < self._low_gradient_abort_threshold:
                raise AbortTraining(f"Vanishing gradients detected. Value: {val}.")
            if (val := np.max(np.abs(param._W))) > self._high_gradient_abort_threshold:
                raise AbortTraining(f"Exploding gradients detected. Value: {val}.")

    def post_epoch(self, L: float) -> None:
        self._L_history.append(L)
        if len(self._L_history) < self._L_history_max_len:
            return
        if all(
            diff > 0
            for diff in (
                b2 - b1
                for b1, b2 in pairwise(
                    sum(batch) for batch in batched(self._L_history, 10)
                )
            )
        ):
            raise AbortTraining("Model is diverging.")
