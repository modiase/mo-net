from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import jax.numpy as jnp

from mo_net.functions import ACTIVATION_FUNCTIONS, ActivationFn, get_activation_fn
from mo_net.model.layer.base import Hidden
from mo_net.protos import (
    Activations,
    D,
    Dimensions,
)
from typing import cast


class Activation(Hidden):
    @dataclass(frozen=True, kw_only=True)
    class Serialized:
        input_dimensions: tuple[int, ...]
        activation_fn: str

        def deserialize(
            self,
            *,
            training: bool = False,
            freeze_parameters: bool = False,
        ) -> Activation:
            del training, freeze_parameters  # unused
            return Activation(
                activation_fn=get_activation_fn(self.activation_fn),
                input_dimensions=self.input_dimensions,
            )

    class Cache(TypedDict):
        input_activations: Activations | None

    def __init__(
        self,
        *,
        activation_fn: ActivationFn[jnp.ndarray],
        input_dimensions: Dimensions,
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
        self._cache["input_activations"] = input_activations
        return Activations(self._activation_fn(input_activations))

    def _backward_prop(self, *, dZ: D[Activations]) -> D[Activations]:
        if (input_activations := self._cache["input_activations"]) is None:
            raise ValueError("Input activations not set during forward pass.")

        derivative = self._activation_fn.deriv(input_activations)
        # derivative * dZ returns D[Activations] due to __rmul__ on D protocol
        return cast(D[Activations], derivative * dZ)

    @property
    def input_dimensions(self) -> Dimensions:
        return self._input_dimensions

    @property
    def output_dimensions(self) -> Dimensions:
        return self._input_dimensions

    def serialize(self) -> Activation.Serialized:
        for name, func in ACTIVATION_FUNCTIONS.items():
            if func == self._activation_fn:
                return Activation.Serialized(
                    input_dimensions=tuple(self._input_dimensions),
                    activation_fn=name,
                )
        raise ValueError(f"Unknown activation function: {self._activation_fn}")
