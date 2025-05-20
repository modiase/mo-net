from __future__ import annotations

import operator
from dataclasses import dataclass
from functools import reduce

from mnist_numpy.model.layer.base import Hidden
from mnist_numpy.protos import Activations, D, Dimensions


class Reshape(Hidden):
    @dataclass(frozen=True, kw_only=True)
    class Serialized:
        input_dimensions: tuple[int, ...]
        output_dimensions: tuple[int, ...]

        def deserialize(self, *, training: bool = False) -> Reshape:
            return Reshape(
                input_dimensions=self.input_dimensions,
                output_dimensions=self.output_dimensions,
            )

    def _forward_prop(self, *, input_activations: Activations) -> Activations:
        return Activations(input_activations.reshape(-1, *self.output_dimensions))

    def _backward_prop(self, *, dZ: D[Activations]) -> D[Activations]:
        return dZ.reshape(-1, *self.input_dimensions)  # type: ignore[attr-defined]

    def serialize(self) -> Reshape.Serialized:
        return Reshape.Serialized(
            input_dimensions=tuple(self.input_dimensions),
            output_dimensions=tuple(self.output_dimensions),
        )


class Flatten(Hidden):
    @dataclass(frozen=True, kw_only=True)
    class Serialized:
        input_dimensions: tuple[int, ...]
        output_dimensions: tuple[int, ...]

        def deserialize(self, *, training: bool = False) -> Flatten:
            del training  # unused
            return Flatten(input_dimensions=self.input_dimensions)

    def __init__(self, *, input_dimensions: Dimensions):
        super().__init__(
            input_dimensions=input_dimensions,
            output_dimensions=(reduce(operator.mul, input_dimensions, 1),),
        )

    def _forward_prop(self, *, input_activations: Activations) -> Activations:
        return Activations(input_activations.reshape(input_activations.shape[0], -1))

    def _backward_prop(self, *, dZ: D[Activations]) -> D[Activations]:
        return dZ.reshape(dZ.shape[0], *self.input_dimensions)  # type: ignore[attr-defined]

    def serialize(self) -> Flatten.Serialized:
        return Flatten.Serialized(
            input_dimensions=tuple(self.input_dimensions),
            output_dimensions=tuple(self.output_dimensions),
        )
