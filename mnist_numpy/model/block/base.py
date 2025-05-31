from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import reduce
from itertools import pairwise

import numpy as np
from more_itertools import first, last

from mnist_numpy.model.layer.base import (
    Hidden as HiddenLayer,
)
from mnist_numpy.model.layer.output import OutputLayer
from mnist_numpy.protos import (
    Activations,
    D,
    Dimensions,
    HasDimensions,
    HasParameterCount,
    SupportsDeserialize,
    SupportsSerialize,
    SupportsUpdateParameters,
)


class Base(HasDimensions):
    def __init__(self, *, layers: Sequence[HiddenLayer]):
        for prev, curr in pairwise(layers):
            if prev.output_dimensions != curr.input_dimensions:
                raise ValueError(
                    f"Output dimensions of layer {prev} ({prev.output_dimensions}) "
                    f"do not match input dimensions of layer {curr} ({curr.input_dimensions})."
                )
        self._layers: Sequence[HiddenLayer] = tuple(layers)

    @property
    def input_dimensions(self) -> Dimensions:
        return first(self._layers).input_dimensions

    @property
    def output_dimensions(self) -> Dimensions:
        return last(self._layers).output_dimensions

    def forward_prop(self, *, input_activations: Activations) -> Activations:
        return reduce(
            lambda activations, layer: layer.forward_prop(
                input_activations=activations
            ),
            self._layers,
            input_activations,
        )

    def prepend_layer(self, layer: HiddenLayer) -> None:
        if not layer.output_dimensions == self.input_dimensions:
            raise ValueError(
                f"Output dimensions of layer to prepend ({layer.output_dimensions}) "
                f"do not match input dimensions of block ({self.input_dimensions})."
            )
        self._layers = tuple([layer, *self._layers])

    def append_layer(self, layer: HiddenLayer) -> None:
        if not self.output_dimensions == layer.input_dimensions:
            raise ValueError(
                f"Output dimensions of block ({self.output_dimensions}) "
                f"do not match input dimensions of layer to append ({layer.input_dimensions})."
            )
        self._layers = tuple([*self._layers, layer])

    @property
    def layers(self) -> Sequence[HiddenLayer]:
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
        layers: Sequence[SupportsDeserialize[HiddenLayer]]

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
        layers: Sequence[SupportsDeserialize[HiddenLayer]]
        output_layer: SupportsDeserialize[OutputLayer]

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
        layers: Sequence[HiddenLayer] | None = None,
        output_layer: OutputLayer,
    ):
        if layers is None:
            layers = []
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
    def input_dimensions(self) -> Dimensions:
        if self._layers:
            return first(self._layers).input_dimensions
        else:
            return self._output_layer.input_dimensions

    @property
    def output_dimensions(self) -> Dimensions:
        if self._layers:
            return last(self._layers).output_dimensions
        else:
            return self._output_layer.output_dimensions

    @property
    def output_layer(self) -> OutputLayer:
        return self._output_layer
