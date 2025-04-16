from collections.abc import MutableSequence

import numpy as np

from mnist_numpy.model.layer import DenseParameters, NonInputLayer
from mnist_numpy.model.mlp import MultiLayerPerceptron
from mnist_numpy.types import Activations, D, TrainingStepHandler


class LayerL2Regulariser(TrainingStepHandler):
    def __init__(self, *, lambda_: float, layer: NonInputLayer):
        self._lambda = lambda_
        self._layer = layer

    def forward(self, As: Activations) -> Activations:
        return As

    def pre_backward(self, dZ: D[Activations]) -> D[Activations]:
        return dZ

    def post_backward(
        self, dP: D[DenseParameters], dZ: D[Activations]
    ) -> tuple[D[DenseParameters], D[Activations]]:
        return dP.__class__(
            _W=dP._W + self._lambda * self._layer.parameters._W,
            _B=dP._B,
        ), dZ

    def compute_regularisation_loss(self) -> float:
        return 0.5 * self._lambda * np.sum(self._layer.parameters._W**2)

    def __call__(self) -> float:
        return self.compute_regularisation_loss()


class L2Regulariser:
    def __init__(self, *, lambda_: float):
        self._lambda = lambda_
        self._layer_regularisers: MutableSequence[LayerL2Regulariser] = []

    def __call__(self, model: MultiLayerPerceptron) -> float:
        for layer in model.non_input_layers:
            layer_regulariser = LayerL2Regulariser(lambda_=self._lambda, layer=layer)
            layer.register_training_step_handler(layer_regulariser)
            self._layer_regularisers.append(layer_regulariser)
            model.register_loss_contributor(layer_regulariser)
