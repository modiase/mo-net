from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypedDict

import jax
import jax.numpy as jnp

from mo_net.functions import ACTIVATION_FUNCTIONS, get_activation_fn
from mo_net.model.layer.base import Hidden
from mo_net.protos import (
    Activations,
    D,
    Dimensions,
)


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
        activation_fn: Callable[[jnp.ndarray], jnp.ndarray],
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

        # For common activation functions, we can compute the derivative directly
        # This is more efficient than using automatic differentiation
        if self._activation_fn.__name__ == "relu":
            return jnp.where(input_activations > 0, 1, 0) * dZ
        elif self._activation_fn.__name__ == "tanh":
            return (1 - jnp.tanh(input_activations) ** 2) * dZ
        elif self._activation_fn.__name__ == "leaky_relu":
            return jnp.where(input_activations > 0, 1, 0.01) * dZ
        elif self._activation_fn.__name__ == "identity":
            return dZ
        else:
            input_float = input_activations.astype(jnp.float32)
            grad_fn = jax.grad(lambda x: jnp.sum(self._activation_fn(x)))
            return grad_fn(input_float) * dZ

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
