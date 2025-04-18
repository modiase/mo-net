from collections import deque

import numpy as np

from mnist_numpy.model.mlp import MultiLayerPerceptron
from mnist_numpy.trainer.exceptions import AbortTraining


class Monitor:
    def __init__(
        self,
        *,
        low_gradient_abort_threshold: float,
        high_gradient_abort_threshold: float,
        history_max_len: int,
        X_train: np.ndarray,
        Y_train: np.ndarray,
    ):
        self._low_gradient_abort_threshold = low_gradient_abort_threshold
        self._high_gradient_abort_threshold = high_gradient_abort_threshold
        self._X_train = X_train
        self._Y_train = Y_train
        self._L_history_max_len = history_max_len
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
        if np.polyfit(range(len(self._L_history)), self._L_history, 1)[0] > 0:
            raise AbortTraining("Model is diverging.")
