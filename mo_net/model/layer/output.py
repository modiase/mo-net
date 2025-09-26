from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict, TypeVar, cast

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
            output_activations := Activations(jax.nn.softmax(input_activations))
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


class MseOutputLayer(OutputLayer):
    @dataclass(frozen=True, kw_only=True)
    class Serialized:
        input_dimensions: tuple[int, ...]

        def deserialize(
            self,
            *,
            training: bool = False,
            freeze_parameters: bool = False,
        ) -> MseOutputLayer:
            del freeze_parameters  # unused
            return MseOutputLayer(
                input_dimensions=self.input_dimensions,
                training=training,
            )

    def __init__(self, *, input_dimensions: Dimensions, training: bool = True):
        super().__init__(input_dimensions=input_dimensions)
        self._training = training

    def _forward_prop(
        self,
        *,
        input_activations: Activations,
    ) -> Activations:
        if self._training:
            self._cache["output_activations"] = input_activations
        return input_activations

    def _backward_prop(
        self,
        *,
        Y_true: jnp.ndarray,
    ) -> D[Activations]:
        if (output_activations := self._cache["output_activations"]) is None:
            raise ValueError("Output activations not set during forward pass.")

        batch_size = output_activations.shape[0]
        return 2.0 * (output_activations - Y_true) / batch_size

    @property
    def output_dimensions(self) -> Dimensions:
        return self._input_dimensions

    def serialize(self) -> MseOutputLayer.Serialized:
        return self.Serialized(input_dimensions=tuple(self._input_dimensions))


class SparseCategoricalSoftmaxOutputLayer(OutputLayer):
    @dataclass(frozen=True, kw_only=True)
    class Serialized:
        input_dimensions: tuple[int, ...]

        def deserialize(
            self,
            *,
            training: bool = False,
            freeze_parameters: bool = False,
        ) -> SparseCategoricalSoftmaxOutputLayer:
            del training, freeze_parameters  # unused
            return SparseCategoricalSoftmaxOutputLayer(
                input_dimensions=self.input_dimensions,
            )

    def _forward_prop(
        self,
        *,
        input_activations: Activations,
    ) -> Activations:
        self._cache["output_activations"] = (
            output_activations := Activations(jax.nn.softmax(input_activations))
        )
        return output_activations

    def backward_prop_with_negative(
        self,
        *,
        Y_true: jnp.ndarray,
        Y_negative: jnp.ndarray,
    ) -> D[Activations]:
        return self._backward_prop(Y_true=Y_true, Y_negative=Y_negative)

    def _backward_prop(
        self,
        *,
        Y_true: jnp.ndarray,
        Y_negative: jnp.ndarray | None = None,
    ) -> D[Activations]:
        if (output_activations := self._cache["output_activations"]) is None:
            raise ValueError("Output activations not set during forward pass.")

        result = output_activations.copy()
        result = result.at[jnp.arange(Y_true.shape[0]), Y_true].add(-1.0)

        if Y_negative is not None:
            if Y_negative.ndim == 2:
                batch_size, num_neg = Y_negative.shape
                row_idx = jnp.repeat(jnp.arange(batch_size), num_neg)
                col_idx = Y_negative.reshape(-1)
                result = result.at[row_idx, col_idx].add(1.0)
            elif Y_negative.ndim == 1:
                # - (batch_size,) for a single negative per example
                batch_size = result.shape[0]
                total = Y_negative.shape[0]
                if total == batch_size:
                    result = result.at[jnp.arange(batch_size), Y_negative].add(1.0)
                else:
                    if total % batch_size != 0:
                        raise ValueError(
                            "Y_negative length must be a multiple of batch size when flattened."
                        )
                    num_neg = total // batch_size
                    row_idx = jnp.repeat(jnp.arange(batch_size), num_neg)
                    result = result.at[row_idx, Y_negative].add(1.0)
            else:
                raise ValueError("Y_negative must be 1D or 2D array of indices.")

        return cast(D[Activations], jnp.atleast_1d(result))

    @property
    def output_dimensions(self) -> Dimensions:
        return self._input_dimensions

    def serialize(self) -> SparseCategoricalSoftmaxOutputLayer.Serialized:
        return self.Serialized(input_dimensions=tuple(self._input_dimensions))
