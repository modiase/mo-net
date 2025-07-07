from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict, TypeVar

import jax
import jax.numpy as jnp
from pyparsing import abstractmethod

from mo_net.model.layer.base import _Base
from mo_net.protos import (
    Activations,
    D,
    Dimensions,
    SupportsDeserialize,
    SupportsSerialize,
)

OutputLayerT_co = TypeVar("OutputLayerT_co", bound="OutputLayer", covariant=True)


class OutputLayer(_Base, SupportsSerialize[OutputLayerT_co]):
    class Cache(TypedDict):
        output_activations: Activations | None

    Serialized: type[SupportsDeserialize[OutputLayerT_co]]

    def __init__(
        self,
        *,
        input_dimensions: Dimensions,
    ):
        super().__init__(
            input_dimensions=input_dimensions,
            output_dimensions=input_dimensions,
        )
        self._cache: OutputLayer.Cache = {
            "output_activations": None,
        }

    def backward_prop(self, *, Y_true: jnp.ndarray) -> D[Activations]:
        return self._backward_prop(Y_true=Y_true)

    @abstractmethod
    def _backward_prop(
        self,
        *,
        Y_true: jnp.ndarray,
    ) -> D[Activations]: ...

    @abstractmethod
    def serialize(self) -> SupportsDeserialize: ...


class SoftmaxOutputLayer(OutputLayer):
    @dataclass(frozen=True, kw_only=True)
    class Serialized:
        input_dimensions: tuple[int, ...]

        def deserialize(
            self,
            *,
            training: bool = False,
            freeze_parameters: bool = False,
        ) -> SoftmaxOutputLayer:
            del training, freeze_parameters  # unused
            return SoftmaxOutputLayer(
                input_dimensions=self.input_dimensions,
            )

    def _forward_prop(
        self,
        *,
        input_activations: Activations,
    ) -> Activations:
        self._cache["output_activations"] = (
            output_activations := jax.nn.softmax(input_activations)
        )
        return output_activations

    def _backward_prop(
        self,
        *,
        Y_true: jnp.ndarray,
    ) -> D[Activations]:
        if (output_activations := self._cache["output_activations"]) is None:
            raise ValueError("Output activations not set during forward pass.")
        return jnp.atleast_1d(output_activations - Y_true)

    @property
    def output_dimensions(self) -> Dimensions:
        return self._input_dimensions

    def serialize(self) -> SoftmaxOutputLayer.Serialized:
        return self.Serialized(input_dimensions=tuple(self._input_dimensions))


class RawOutputLayer(SoftmaxOutputLayer):
    """
    Uses softmax backprop but does not apply softmax to the output.
    """

    @dataclass(frozen=True, kw_only=True)
    class Serialized:
        input_dimensions: tuple[int, ...]

        def deserialize(
            self,
            *,
            training: bool = False,
            freeze_parameters: bool = False,
        ) -> RawOutputLayer:
            del training, freeze_parameters  # unused
            return RawOutputLayer(input_dimensions=self.input_dimensions)

    def _forward_prop(
        self,
        *,
        input_activations: Activations,
    ) -> Activations:
        self._cache["output_activations"] = input_activations
        return input_activations


class SparseCategoricalCrossentropyOutputLayer(OutputLayer):
    @dataclass(frozen=True, kw_only=True)
    class Serialized:
        input_dimensions: tuple[int, ...]

        def deserialize(
            self,
            *,
            training: bool = False,
            freeze_parameters: bool = False,
        ) -> SparseCategoricalCrossentropyOutputLayer:
            del training, freeze_parameters  # unused
            return SparseCategoricalCrossentropyOutputLayer(
                input_dimensions=self.input_dimensions,
            )

    def _forward_prop(
        self,
        *,
        input_activations: Activations,
    ) -> Activations:
        # Store both raw logits and softmax outputs for backward pass
        self._cache["output_activations"] = (
            output_activations := jax.nn.softmax(input_activations)
        )
        self._cache["logits"] = input_activations
        return output_activations

    def _backward_prop(
        self,
        *,
        Y_true: jnp.ndarray,
    ) -> D[Activations]:
        if (output_activations := self._cache["output_activations"]) is None:
            raise ValueError("Output activations not set during forward pass.")

        # Y_true contains integer indices
        # For sparse categorical crossentropy, the gradient is:
        # ∂L/∂z_i = softmax(z_i) - δ(i, target)
        # where δ(i, target) = 1 if i == target, 0 otherwise

        # Create a sparse gradient: subtract 1 from the target index position
        # and keep all other positions as softmax outputs
        vocab_size = self._input_dimensions[0]

        # Use advanced indexing to efficiently compute the gradient
        # This avoids creating the full one-hot matrix
        batch_size = Y_true.shape[0]
        batch_indices = jnp.arange(batch_size)

        # Start with softmax outputs
        gradient = output_activations.copy()

        # Subtract 1 from the target positions
        gradient = gradient.at[batch_indices, Y_true].add(-1.0)

        return jnp.atleast_1d(gradient)

    @property
    def output_dimensions(self) -> Dimensions:
        return self._input_dimensions

    def serialize(self) -> SparseCategoricalCrossentropyOutputLayer.Serialized:
        return self.Serialized(input_dimensions=tuple(self._input_dimensions))
