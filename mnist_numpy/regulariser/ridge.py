from collections.abc import MutableSequence

import numpy as np

from mnist_numpy.model.layer import NonInputLayer
from mnist_numpy.model.mlp import MultiLayerPerceptron
from mnist_numpy.types import Activations, D, TrainingStepHandler, _ParamType


class LayerL2Regulariser(TrainingStepHandler):
    def __init__(self, *, lambda_: float, layer: NonInputLayer):
        self._lambda = lambda_
        self._layer = layer

    def forward(self, As: Activations) -> Activations:
        return As

    def pre_backward(self, dZ: D[Activations]) -> D[Activations]:
        return dZ

    def post_backward(
        self, dP: D[_ParamType], dZ: D[Activations]
    ) -> tuple[D[_ParamType], D[Activations]]:
        # D is preventing the __init__ from being resolved by the type checker
        return dP.__class__(  # type: ignore[call-arg] # TODO: Fix types
            _W=dP._W + self._lambda * self._layer.parameters._W,  # type: ignore[attr-defined]
            _B=dP._B,  # type: ignore[attr-defined]
        ), dZ

    def compute_regularisation_loss(self) -> float:
        return 0.5 * self._lambda * np.sum(self._layer.parameters._W**2)

    def __call__(self) -> float:
        return self.compute_regularisation_loss()


class L2Regulariser:
    def __init__(self, *, lambda_: float):
        self._lambda = lambda_
        self._layer_regularisers: MutableSequence[LayerL2Regulariser] = []

    def __call__(self, model: MultiLayerPerceptron) -> None:
        for layer in model.non_input_layers:
            layer_regulariser = LayerL2Regulariser(lambda_=self._lambda, layer=layer)
            layer.register_training_step_handler(layer_regulariser)
            self._layer_regularisers.append(layer_regulariser)
            model.register_loss_contributor(layer_regulariser)
