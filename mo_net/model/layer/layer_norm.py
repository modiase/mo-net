from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, TypedDict

import numpy as np
from more_itertools import one

from mo_net.model.layer.base import (
    Hidden,
)
from mo_net.protos import (
    Activations,
    D,
    SupportsDeserialize,
    d,
)


class Cache(TypedDict):
    output_activations: Activations | None
    std: np.ndarray | None


type CacheType = Cache


class LayerNorm(Hidden):
    """https://arxiv.org/pdf/1607.06450"""

    _EPSILON: ClassVar[float] = 1e-8
    Cache = Cache

    @dataclass(frozen=True, kw_only=True)
    class Serialized:
        neurons: int

        def deserialize(
            self,
            *,
            training: bool = False,
        ) -> LayerNorm:
            return LayerNorm(
                neurons=self.neurons,
                training=training,
            )

    def __init__(
        self,
        *,
        neurons: int,
        store_output_activations: bool = False,
        training: bool = True,
    ):
        super().__init__(
            input_dimensions=(neurons,),
            output_dimensions=(neurons,),
        )
        self._training = training
        self._cache: CacheType = {
            "output_activations": None,
            "std": None,
        }
        self._store_output_activations = store_output_activations

    def _forward_prop(self, *, input_activations: Activations) -> Activations:
        layer_mean = np.mean(input_activations, axis=1, keepdims=True)
        layer_std = np.sqrt(
            np.var(input_activations, axis=1, keepdims=True) + self._EPSILON
        )
        self._cache["std"] = layer_std

        normalised_activations = (input_activations - layer_mean) / layer_std

        if self._store_output_activations or self._training:
            self._cache["output_activations"] = normalised_activations

        return normalised_activations

    def _backward_prop(
        self,
        *,
        dZ: D[Activations],
    ) -> D[Activations]:
        if (std := self._cache["std"]) is None:
            raise RuntimeError(
                "Standard deviation is not populated during backward pass."
            )
        return d(Activations(dZ * 1 / std))

    @property
    def cache(self) -> CacheType:
        return self._cache

    def serialize(self) -> SupportsDeserialize[LayerNorm]:
        return self.Serialized(
            neurons=one(self._input_dimensions),
        )
