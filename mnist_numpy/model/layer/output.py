from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import TypedDict, TypeVar

import numpy as np
from pyparsing import abstractmethod

from mnist_numpy.functions import softmax
from mnist_numpy.model.layer.base import _Base
from mnist_numpy.protos import Activations, D, SupportsDeserialize, SupportsSerialize

OutputLayerT_co = TypeVar("OutputLayerT_co", bound="_OutputLayer", covariant=True)


class _OutputLayer(_Base, SupportsSerialize[OutputLayerT_co]):
    class Cache(TypedDict):
        output_activations: Activations | None

    Serialized: type[SupportsDeserialize[OutputLayerT_co]]

    def __init__(
        self,
        *,
        input_dimensions: int,
    ):
        super().__init__(
            input_dimensions=input_dimensions,
            output_dimensions=input_dimensions,
        )
        self._cache: _OutputLayer.Cache = {
            "output_activations": None,
        }

    def backward_prop(self, *, Y_true: np.ndarray) -> D[Activations]:
        return reduce(
            lambda dZ, handler: handler.post_backward(dZ=dZ),
            reversed(self._training_step_handlers),
            self._backward_prop(Y_true=Y_true),
        )

    @abstractmethod
    def _backward_prop(
        self,
        *,
        Y_true: np.ndarray,
    ) -> D[Activations]: ...

    @abstractmethod
    def serialize(self) -> SupportsDeserialize: ...


class SoftmaxOutputLayer(_OutputLayer):
    @dataclass(frozen=True, kw_only=True)
    class Serialized:
        input_dimensions: int

        def deserialize(
            self,
            *,
            training: bool = False,
        ) -> SoftmaxOutputLayer:
            del training  # unused
            return SoftmaxOutputLayer(input_dimensions=self.input_dimensions)

    def _forward_prop(
        self,
        *,
        input_activations: Activations,
    ) -> Activations:
        self._cache["output_activations"] = (
            output_activations := softmax(input_activations)
        )
        return output_activations

    def _backward_prop(
        self,
        *,
        Y_true: np.ndarray,
    ) -> D[Activations]:
        if (output_activations := self._cache["output_activations"]) is None:
            raise ValueError("Output activations not set during forward pass.")
        return np.atleast_1d(output_activations - Y_true)

    @property
    def output_dimensions(self) -> int:
        return self._input_dimensions

    def serialize(self) -> SoftmaxOutputLayer.Serialized:
        return self.Serialized(input_dimensions=self._input_dimensions)


class RawOutputLayer(SoftmaxOutputLayer):
    @dataclass(frozen=True, kw_only=True)
    class Serialized:
        input_dimensions: int

        def deserialize(
            self,
            *,
            training: bool = False,
        ) -> RawOutputLayer:
            del training  # unused
            return RawOutputLayer(input_dimensions=self.input_dimensions)

    def _forward_prop(
        self,
        *,
        input_activations: Activations,
    ) -> Activations:
        self._cache["output_activations"] = input_activations
        return input_activations
