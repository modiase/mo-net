import functools
from collections.abc import Callable
from itertools import chain
from operator import itemgetter

import numpy as np
from more_itertools import one

from mnist_numpy.model.layer import DenseParameters, HiddenLayerBase
from mnist_numpy.model.mlp import MultiLayerPerceptron
from mnist_numpy.types import Activations, D


def dropout(*, keep_prob: float) -> Callable[[HiddenLayerBase], HiddenLayerBase]:
    # TODO: Instead of wrapping the layer it should register itself on the layer
    # as part of the forward or backward prop chain
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
            preactivations, og_activations = itemgetter(slice(-1), 1)(
                original_forward_prop(As_prev=As_prev)
            )
            dropout_mask = (np.random.rand(*og_activations.shape) < keep_prob).astype(
                og_activations.dtype
            )
            return tuple(
                chain(
                    preactivations,
                    (og_activations * dropout_mask / keep_prob,),
                )
            )

        layer.forward_prop = forward_prop  # type: ignore[method-assign, assignment]

        original_backward_prop = layer._backward_prop

        @functools.wraps(original_backward_prop)
        def backward_prop(
            *,
            As_prev: Activations,
            Zs_prev: Activations,
            dZ: D[Activations],
        ) -> tuple[D[DenseParameters], D[Activations]]:
            nonlocal dropout_mask
            if dropout_mask is not None:
                dZ *= dropout_mask
                dZ /= dropout_mask.mean()
            dp, dZ = original_backward_prop(As_prev=As_prev, Zs_prev=Zs_prev, dZ=dZ)
            return dp, dZ

        layer._backward_prop = backward_prop  # type: ignore[method-assign, assignment]

        return layer

    return wrapped


class DropoutVisitor:
    # TODO: Replace this with a regulariser as defined in ridge.py
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
