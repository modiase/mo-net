import numpy as np

from mnist_numpy.model.mlp import MultiLayerPerceptron
from mnist_numpy.monitor.exceptions import AbortTraining


class Monitor:
    def __init__(
        self,
        *,
        low_gradient_abort_threshold: float,
        high_gradient_abort_threshold: float,
    ):
        self._low_gradient_abort_threshold = low_gradient_abort_threshold
        self._high_gradient_abort_threshold = high_gradient_abort_threshold

    def __call__(
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
