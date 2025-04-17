from typing import MutableSequence

import numpy as np
from more_itertools import one

from mnist_numpy.model.mlp import MultiLayerPerceptron
from mnist_numpy.types import Activations, D, TrainingStepHandler


class LayerDropout(TrainingStepHandler):
    def __init__(self, *, keep_prob: float):
        self._keep_prob = keep_prob
        self._dropout_mask: np.ndarray | None = None

    def post_forward(self, As: Activations) -> Activations:
        self._dropout_mask = (np.random.rand(*As.shape) < self._keep_prob).astype(
            As.dtype
        )
        return As * self._dropout_mask / self._keep_prob

    def pre_backward(self, dZ: D[Activations]) -> D[Activations]:
        # Activations are just np.ndarray, so we can multiply by the dropout mask
        if self._dropout_mask is None:
            raise RuntimeError("Dropout mask is not set")
        return dZ * self._dropout_mask / self._dropout_mask.mean()  # type: ignore[operator] # TODO: Fix types


class DropoutRegulariser:
    def __init__(self, *, keep_probs: tuple[float, ...]):
        self._keep_probs = keep_probs
        self._layer_dropouts: MutableSequence[LayerDropout] = []

    def __call__(self, model: MultiLayerPerceptron) -> None:
        if len(self._keep_probs) == 0:
            return
        elif len(self._keep_probs) == 1:
            self._keep_probs = tuple(
                one(self._keep_probs) for _ in range(len(model.hidden_layers))
            )
        elif isinstance(self._keep_probs, tuple) and len(self._keep_probs) != len(
            model.hidden_layers
        ):
            raise ValueError(
                f"Number of keep probabilities ({len(self._keep_probs)}) must match number of hidden layers ({len(model.hidden_layers)})"
            )
        elif not isinstance(self._keep_probs, tuple):
            raise ValueError(
                f"keep_probs must be a float or a tuple. Got {type(self._keep_probs)}"
            )

        for layer, keep_prob in zip(model.hidden_layers, self._keep_probs):
            layer_dropout = LayerDropout(keep_prob=keep_prob)
            layer.register_training_step_handler(layer_dropout)
            self._layer_dropouts.append(layer_dropout)
