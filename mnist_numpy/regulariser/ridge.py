from itertools import chain

import numpy as np

from mnist_numpy.model.layer.linear import Linear
from mnist_numpy.model.mlp import MultiLayerPerceptron
from mnist_numpy.protos import Activations, D, TrainingStepHandler, d


class LayerL2Regulariser(TrainingStepHandler):
    def __init__(self, *, lambda_: float, batch_size: int, layer: Linear):
        self._lambda = lambda_
        self._layer = layer
        self._batch_size = batch_size

    def post_backward(self, dZ: D[Activations]) -> D[Activations]:
        if self._layer.cache["dP"] is None:
            raise ValueError("dP was not set during forward pass.")
        dP = self._layer.cache["dP"]
        self._layer.cache["dP"] = d(
            dP
            + Linear.Parameters(
                _W=dP._W  # type: ignore[attr-defined]
                + self._lambda * self._layer.parameters._W,
                _B=dP._B,  # type: ignore[attr-defined]
            )
        )
        return dZ

    def compute_regularisation_loss(self) -> float:
        return (
            0.5 * self._lambda * np.sum(self._layer.parameters._W**2) / self._batch_size
        )

    def __call__(self) -> float:
        return self.compute_regularisation_loss()


def attach_l2_regulariser(
    *,
    lambda_: float,
    batch_size: int,
    model: MultiLayerPerceptron,
) -> None:
    for layer in chain.from_iterable(block.layers for block in model.blocks):
        if not isinstance(layer, Linear):
            continue
        layer_regulariser = LayerL2Regulariser(
            lambda_=lambda_, batch_size=batch_size, layer=layer
        )
        layer.register_training_step_handler(layer_regulariser)
        model.register_loss_contributor(layer_regulariser)
