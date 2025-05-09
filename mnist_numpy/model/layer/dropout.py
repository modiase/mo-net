from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypedDict

import numpy as np

from mnist_numpy.model.layer.base import Hidden
from mnist_numpy.model.mlp import MultiLayerPerceptron
from mnist_numpy.protos import Activations, D


class DropoutLayer(Hidden):
    """
    https://arxiv.org/abs/1207.0580
    """

    @dataclass(frozen=True, kw_only=True)
    class Serialized:
        input_dimensions: int
        keep_prob: float

        def deserialize(self, *, training: bool) -> DropoutLayer:
            return DropoutLayer(
                input_dimensions=self.input_dimensions,
                keep_prob=self.keep_prob,
                training=training,
            )

    class Cache(TypedDict):
        mask: np.ndarray | None
        input_activations: Activations | None

    def __init__(
        self,
        *,
        input_dimensions: int,
        keep_prob: float,
        training: bool = False,
    ):
        super().__init__(
            input_dimensions=input_dimensions,
            output_dimensions=input_dimensions,
        )
        self._training = training

        if not 0.0 < keep_prob <= 1.0:
            raise ValueError(f"keep_prob must be in (0, 1], got {keep_prob}")

        self._keep_prob = keep_prob
        self._cache: DropoutLayer.Cache = {
            "mask": None,
            "input_activations": None,
        }

    def _forward_prop(self, *, input_activations: Activations) -> Activations:
        if self._keep_prob == 1.0 or not self._training:
            return input_activations
        self._cache["input_activations"] = input_activations

        mask = np.random.binomial(1, self._keep_prob, size=input_activations.shape)
        self._cache["mask"] = mask

        return input_activations * mask / self._keep_prob

    def _backward_prop(self, *, dZ: D[Activations]) -> D[Activations]:
        if self._keep_prob == 1.0 or not self._training:
            return dZ

        if self._cache["mask"] is None:
            raise ValueError("Mask not set during forward pass.")

        return dZ * self._cache["mask"] / self._keep_prob

    def serialize(self) -> DropoutLayer.Serialized:
        return DropoutLayer.Serialized(
            input_dimensions=self.input_dimensions,
            keep_prob=self._keep_prob,
        )


def attach_dropout_layers(
    *,
    model: MultiLayerPerceptron,
    keep_probs: Sequence[float],
    training: bool,
) -> None:
    if len(keep_probs) != len(model.hidden_blocks):
        raise ValueError(
            f"Number of keep probabilities ({len(keep_probs)}) must match the number of hidden blocks ({len(model.hidden_blocks)})"
        )
    for block, keep_prob in zip(model.hidden_blocks, keep_probs, strict=True):
        block.append_layer(
            DropoutLayer(
                input_dimensions=block.output_dimensions,
                keep_prob=keep_prob,
                training=training,
            )
        )
