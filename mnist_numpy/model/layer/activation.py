from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from more_itertools import last

from mnist_numpy.functions import get_activation_fn
from mnist_numpy.model.layer.base import _Hidden
from mnist_numpy.protos import ActivationFn, ActivationFnName, Activations, D


class Activation(_Hidden):
    @dataclass(frozen=True, kw_only=True)
    class Serialized:
        input_dimensions: int
        activation_fn: ActivationFnName

        def deserialize(
            self,
            *,
            training: bool = False,
        ) -> Activation:
            del training  # unused
            return Activation(
                activation_fn=get_activation_fn(self.activation_fn),
                input_dimensions=self.input_dimensions,
            )

    class Cache(TypedDict):
        input_activations: Activations | None

    def __init__(
        self,
        *,
        activation_fn: ActivationFn,
        input_dimensions: int,
    ):
        super().__init__(
            input_dimensions=input_dimensions,
            output_dimensions=input_dimensions,
        )
        self._activation_fn = activation_fn
        self._cache: Activation.Cache = {
            "input_activations": None,
        }

    def _forward_prop(self, *, input_activations: Activations) -> Activations:
        if last(input_activations.shape) != self._input_dimensions:
            raise ValueError(
                f"Input activations of layer ({last(input_activations.shape)}) "
                f"do not match input dimensions of layer ({self._input_dimensions})."
            )
        self._cache["input_activations"] = input_activations
        return self._activation_fn(input_activations)

    def _backward_prop(self, *, dZ: D[Activations]) -> D[Activations]:
        if (input_activations := self._cache["input_activations"]) is None:
            raise ValueError("Input activations not set during forward pass.")
        return self._activation_fn.deriv(input_activations) * dZ

    @property
    def input_dimensions(self) -> int:
        return self._input_dimensions

    @property
    def output_dimensions(self) -> int:
        return self._input_dimensions

    def serialize(self) -> Activation.Serialized:
        return Activation.Serialized(
            input_dimensions=self._input_dimensions,
            activation_fn=self._activation_fn.name,
        )
