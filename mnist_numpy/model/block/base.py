from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import reduce
from itertools import pairwise

import numpy as np
from more_itertools import first, last

from mnist_numpy.model.layer.base import (
    _Hidden,
)
from mnist_numpy.model.layer.output import _OutputLayer
from mnist_numpy.protos import (
    Activations,
    D,
    HasParameterCount,
    SupportsDeserialize,
    SupportsSerialize,
    SupportsUpdateParameters,
)


class Base:
    def __init__(self, *, layers: Sequence[_Hidden]):
        for prev, curr in pairwise(layers):
            if prev.output_dimensions != curr.input_dimensions:
                raise ValueError(
                    f"Output dimensions of layer ({prev.output_dimensions}) "
                    f"do not match input dimensions of layer to append ({curr.input_dimensions})."
                )
        self._layers: Sequence[_Hidden] = tuple(layers)

    @property
    def input_dimensions(self) -> int:
        return first(self._layers).input_dimensions

    @property
    def output_dimensions(self) -> int:
        return last(self._layers).output_dimensions

    @property
    def dimensions(self) -> tuple[int, int]:
        return (self.input_dimensions, self.output_dimensions)

    def forward_prop(self, *, input_activations: Activations) -> Activations:
        return reduce(
            lambda activations, layer: layer.forward_prop(
                input_activations=activations
            ),
            self._layers,
            input_activations,
        )

    def prepend_layer(self, layer: _Hidden) -> None:
        if not layer.output_dimensions == self.input_dimensions:
            raise ValueError(
                f"Output dimensions of layer to prepend ({layer.output_dimensions}) "
                f"do not match input dimensions of block ({self.input_dimensions})."
            )
        self._layers = tuple([layer, *self._layers])

    def append_layer(self, layer: _Hidden) -> None:
        if not self.output_dimensions == layer.input_dimensions:
            raise ValueError(
                f"Output dimensions of block ({self.output_dimensions}) "
                f"do not match input dimensions of layer to append ({layer.input_dimensions})."
            )
        self._layers = tuple([*self._layers, layer])

    @property
    def layers(self) -> Sequence[_Hidden]:
        return self._layers

    def update_parameters(self) -> None:
        for layer in self._layers:
            if isinstance(layer, SupportsUpdateParameters):
                layer.update_parameters()

    @property
    def parameter_count(self) -> int:
        return sum(
            layer.parameter_count
            for layer in self._layers
            if isinstance(layer, HasParameterCount)
        )


class Hidden(Base):
    @dataclass(frozen=True, kw_only=True)
    class Serialized:
        layers: Sequence[SupportsDeserialize[_Hidden]]

        def deserialize(
            self,
            *,
            training: bool = False,
        ) -> Hidden:
            return Hidden(
                layers=tuple(
                    layer.deserialize(training=training) for layer in self.layers
                ),
            )

    def backward_prop(self, *, dZ: D[Activations]) -> D[Activations]:
        return reduce(
            lambda dZ, layer: layer.backward_prop(dZ=dZ),
            reversed(self._layers),
            dZ,
        )

    def serialize(self) -> Hidden.Serialized:
        return Hidden.Serialized(
            layers=tuple(
                layer.serialize()
                for layer in self._layers
                if isinstance(layer, SupportsSerialize)
            ),
        )


class Output(Base):
    @dataclass(frozen=True, kw_only=True)
    class Serialized:
        layers: Sequence[SupportsDeserialize[_Hidden]]
        output_layer: SupportsDeserialize[_OutputLayer]

        def deserialize(
            self,
            *,
            training: bool = False,
        ) -> Output:
            return Output(
                layers=tuple(
                    layer.deserialize(training=training) for layer in self.layers
                ),
                output_layer=self.output_layer.deserialize(training=training),
            )

    def __init__(
        self,
        *,
        layers: Sequence[_Hidden],
        output_layer: _OutputLayer,
    ):
        super().__init__(layers=layers)
        self._output_layer = output_layer

    def forward_prop(self, *, input_activations: Activations) -> Activations:
        return self._output_layer.forward_prop(
            input_activations=super().forward_prop(input_activations=input_activations)
        )

    def backward_prop(self, *, Y_true: np.ndarray) -> D[Activations]:
        return reduce(
            lambda dZ, layer: layer.backward_prop(dZ=dZ),
            reversed(self._layers),
            self._output_layer.backward_prop(Y_true=Y_true),
        )

    def serialize(self) -> Output.Serialized:
        return Output.Serialized(
            layers=tuple(
                layer.serialize()
                for layer in self._layers
                if isinstance(layer, SupportsSerialize)
            ),
            output_layer=self._output_layer.serialize(),
        )

    @property
    def output_layer(self) -> _OutputLayer:
        return self._output_layer
