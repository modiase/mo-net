import functools
from collections.abc import Callable

import numpy as np
from more_itertools import one

from mnist_numpy.model.layer import DenseParameters, HiddenLayerBase
from mnist_numpy.model.mlp import MultiLayerPerceptron
from mnist_numpy.types import Activations, D, PreActivations


def dropout(*, keep_prob: float) -> Callable[[HiddenLayerBase], HiddenLayerBase]:
    def wrapped(layer: HiddenLayerBase) -> HiddenLayerBase:
        if keep_prob < 0.0 or keep_prob > 1.0:
            raise ValueError("keep_prob must be between 0.0 and 1.0")

        if keep_prob == 1.0:
            return layer

        dropout_mask: np.ndarray | None = None

        original_forward_prop = layer.forward_prop

        @functools.wraps(original_forward_prop)
        def forward_prop(
            *,
            As_prev: Activations,
        ) -> tuple[Activations, ...]:
            # TODO: Fix dropping out the wrong layer
            nonlocal dropout_mask
            dropout_mask = (np.random.rand(*As_prev.shape) < keep_prob).astype(
                As_prev.dtype
            )
            As = Activations((As_prev * dropout_mask) / keep_prob)
            return original_forward_prop(As_prev=As)

        layer._forward_prop = forward_prop  # type: ignore[method-assign, assignment]

        original_backward_prop = layer._backward_prop

        @functools.wraps(original_backward_prop)
        def backward_prop(
            *,
            As_prev: Activations,
            Zs_prev: PreActivations,
            dZ: D[PreActivations],
        ) -> tuple[D[DenseParameters], D[PreActivations]]:
            nonlocal dropout_mask
            dp, dZ = original_backward_prop(As_prev=As_prev, Zs_prev=Zs_prev, dZ=dZ)
            if dropout_mask is not None:
                dZ *= dropout_mask
                dZ /= dropout_mask.mean()
            return dp, dZ

        layer._backward_prop = backward_prop  # type: ignore[method-assign, assignment]

        return layer

    return wrapped


class DropoutVisitor:
    def __init__(self, *, model: MultiLayerPerceptron, keep_prob: tuple[float, ...]):
        self.model = model
        if len(keep_prob) == 0:
            self.keep_prob = tuple(1.0 for _ in range(len(model.hidden_layers)))
        elif len(keep_prob) == 1:
            self.keep_prob = tuple(
                one(keep_prob) for _ in range(len(model.hidden_layers))
            )
        elif len(keep_prob) == len(model.hidden_layers):
            self.keep_prob = keep_prob
        else:
            raise ValueError(
                f"Number of keep probabilities ({len(keep_prob)}) must match number of hidden layers ({len(model.hidden_layers)})"
            )

    def visit(self, layer: HiddenLayerBase, layer_index: int) -> HiddenLayerBase:
        return dropout(keep_prob=self.keep_prob[layer_index])(layer)

    def __call__(self, layer: HiddenLayerBase, layer_index: int) -> HiddenLayerBase:
        return self.visit(layer, layer_index)
