from collections.abc import MutableSequence

import numpy as np

from mnist_numpy.model.layer import DenseParameters, NonInputLayer
from mnist_numpy.model.mlp import MultiLayerPerceptron
from mnist_numpy.types import Activations, D, TrainingStepHandler, _ParamType


class LayerL2Regulariser(TrainingStepHandler):
    def __init__(self, *, lambda_: float, batch_size: int, layer: NonInputLayer):
        self._lambda = lambda_
        self._layer = layer
        self._batch_size = batch_size

    def post_backward(
        self, dP: D[_ParamType], dZ: D[Activations]
    ) -> tuple[D[_ParamType], D[Activations]]:
        # D is preventing the __init__ from being resolved by the type checker
        return dP.__class__(  # type: ignore[call-arg] # TODO: Fix types
            _W=dP._W + (self._lambda / self._batch_size) * self._layer.parameters._W,  # type: ignore[attr-defined]
            _B=dP._B,  # type: ignore[attr-defined]
        ), dZ

    def compute_regularisation_loss(self) -> float:
        if not isinstance(self._layer.parameters, DenseParameters):
            # TODO: Regulariser should not need to know about the type of the layer
            return 0
        return (
            0.5 * self._lambda * np.sum(self._layer.parameters._W**2) / self._batch_size
        )

    def __call__(self) -> float:
        return self.compute_regularisation_loss()


class L2Regulariser:
    def __init__(self, *, lambda_: float, batch_size: int):
        self._lambda = lambda_
        self._batch_size = batch_size
        self._layer_regularisers: MutableSequence[LayerL2Regulariser] = []

    def __call__(self, model: MultiLayerPerceptron) -> None:
        for layer in model.non_input_layers:
            if not isinstance(layer.parameters, DenseParameters):
                # TODO: Regulariser should not need to know about the type of the layer
                continue
            layer_regulariser = LayerL2Regulariser(
                lambda_=self._lambda, batch_size=self._batch_size, layer=layer
            )
            layer.register_training_step_handler(layer_regulariser)
            self._layer_regularisers.append(layer_regulariser)
            model.register_loss_contributor(layer_regulariser)
