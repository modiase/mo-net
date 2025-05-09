from itertools import chain

import numpy as np

from mnist_numpy.model.layer.linear import Linear
from mnist_numpy.model.mlp import MultiLayerPerceptron
from mnist_numpy.optimizer.base import Base as BaseOptimizer
from mnist_numpy.protos import TrainingStepHandler, d


class WeightDecayRegulariser(TrainingStepHandler):
    """https://arxiv.org/pdf/1711.05101"""

    def __init__(self, *, lambda_: float, batch_size: int, layer: Linear):
        self._lambda = lambda_
        self._layer = layer
        self._batch_size = batch_size

    def after_compute_update(self, learning_rate: float) -> None:
        dP = self._layer.cache.get("dP", self._layer.empty_gradient())
        self._layer.cache["dP"] = d(
            dP
            + Linear.Parameters(
                _W=self._lambda * learning_rate * self._layer.parameters._W,
                _B=np.zeros_like(dP._B),  # type: ignore[attr-defined]
            )
        )

    def compute_regularisation_loss(self) -> float:
        return (
            0.5 * self._lambda * np.sum(self._layer.parameters._W**2) / self._batch_size
        )

    def __call__(self) -> float:
        return self.compute_regularisation_loss()


def attach_weight_decay_regulariser(
    *,
    lambda_: float,
    batch_size: int,
    optimizer: BaseOptimizer,
    model: MultiLayerPerceptron,
) -> None:
    for layer in chain.from_iterable(block.layers for block in model.blocks):
        if not isinstance(layer, Linear):
            continue
        layer_regulariser = WeightDecayRegulariser(
            lambda_=lambda_, batch_size=batch_size, layer=layer
        )
        optimizer.register_after_compute_update_handler(
            layer_regulariser.after_compute_update
        )
        model.register_loss_contributor(layer_regulariser)
