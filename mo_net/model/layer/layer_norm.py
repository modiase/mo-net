from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import IO, Self

import jax.numpy as jnp

from mo_net.constants import EPSILON
from mo_net.model.layer.base import (
    BadLayerId,
    ParametrisedHidden,
)
from mo_net.protos import (
    Activations,
    D,
    Dimensions,
    GradCache,
    GradLayer,
    SupportsDeserialize,
    SupportsGradientOperations,
    d,
)


@dataclass(frozen=True, kw_only=True)
class Parameters(SupportsGradientOperations):
    weights: jnp.ndarray
    biases: jnp.ndarray

    def __getitem__(
        self, index: int | tuple[int, ...]
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self.weights[index], self.biases[index]

    def __pow__(self, other: float | int) -> Self:
        match other:
            case float() | int():
                return self.__class__(
                    weights=self.weights**other,
                    biases=self.biases**other,
                )
            case _:
                return NotImplemented

    def __truediv__(self, other: Self | float | int) -> Self:
        match other:
            case float() | int():
                return self.__mul__(1 / other)
            case self.__class__():
                return self.__class__(
                    weights=self.weights / other.weights,
                    biases=self.biases / other.biases,
                )
            case _:
                return NotImplemented

    def __add__(self, other: Self | float | int) -> Self:
        match other:
            case float() | int():
                return self.__class__(
                    weights=self.weights + other,
                    biases=self.biases + other,
                )
            case self.__class__():
                return self.__class__(
                    weights=self.weights + other.weights,
                    biases=self.biases + other.biases,
                )
            case _:
                return NotImplemented

    def __radd__(self, other: Self | float | int) -> Self:
        match other:
            case float() | int():
                return self.__class__(
                    weights=other + self.weights,
                    biases=other + self.biases,
                )
            case _:
                return NotImplemented

    def __neg__(self) -> Self:
        return self.__class__(
            weights=-self.weights,
            biases=-self.biases,
        )

    def __sub__(self, other: Self | float) -> Self:
        match other:
            case float() | int():
                return self.__class__(
                    weights=self.weights - other,
                    biases=self.biases - other,
                )
            case self.__class__():
                return self.__class__(
                    weights=self.weights - other.weights,
                    biases=self.biases - other.biases,
                )
            case _:
                return NotImplemented

    def __mul__(self, other: float | Self) -> Self:
        match other:
            case float() | int():
                return self.__class__(
                    weights=self.weights * other,
                    biases=self.biases * other,
                )
            case self.__class__():
                return self.__class__(
                    weights=self.weights * other.weights,
                    biases=self.biases * other.biases,
                )
            case _:
                return NotImplemented

    def __rmul__(self, other: float | Self) -> Self:
        return self.__mul__(other)

    def __rsub__(self, other: float | Self) -> Self:
        match other:
            case float() | int():
                return self.__class__(
                    weights=other - self.weights,
                    biases=other - self.biases,
                )
            case self.__class__():
                return self.__class__(
                    weights=other.weights - self.weights,
                    biases=other.biases - self.biases,
                )
            case _:
                return NotImplemented

    def __rtruediv__(self, other: float | Self) -> Self:
        match other:
            case float() | int():
                return self.__class__(
                    weights=other / self.weights,
                    biases=other / self.biases,
                )
            case self.__class__():
                return self.__class__(
                    weights=other.weights / self.weights,
                    biases=other.biases / self.biases,
                )
            case _:
                return NotImplemented

    @classmethod
    def empty(cls, *, input_dimensions: Dimensions) -> Self:
        return cls(
            weights=jnp.ones(input_dimensions),
            biases=jnp.zeros(input_dimensions),
        )

    def from_bytes(self, data: IO[bytes]) -> Self:
        return self.__class__(
            weights=jnp.frombuffer(
                data.read(self.weights.nbytes), dtype=self.weights.dtype
            ).reshape(self.weights.shape),
            biases=jnp.frombuffer(
                data.read(self.biases.nbytes), dtype=self.biases.dtype
            ).reshape(self.biases.shape),
        )


type ParametersType = Parameters


class Cache(GradCache[ParametersType]):
    input_activations: Activations | None
    mean: jnp.ndarray | None
    output_activations: Activations | None
    var: jnp.ndarray | None


type CacheType = Cache


class LayerNorm(ParametrisedHidden[ParametersType, CacheType]):
    """https://arxiv.org/pdf/1607.06450"""

    Parameters = Parameters
    Cache = Cache

    @dataclass(frozen=True, kw_only=True)
    class Serialized:
        layer_id: str
        input_dimensions: tuple[int, ...]
        parameters: Parameters

        def deserialize(
            self,
            *,
            training: bool = False,
        ) -> LayerNorm:
            return LayerNorm(
                layer_id=self.layer_id,
                input_dimensions=self.input_dimensions,
                parameters=self.parameters,
                training=training,
            )

    def __init__(
        self,
        *,
        input_dimensions: Dimensions,
        layer_id: str | None = None,
        parameters: ParametersType | None = None,
        store_output_activations: bool = False,
        training: bool = True,
    ):
        super().__init__(
            layer_id=layer_id,
            input_dimensions=input_dimensions,
            output_dimensions=input_dimensions,
        )
        self._training = training
        self._cache: CacheType = {
            "dP": None,
            "input_activations": None,
            "mean": None,
            "output_activations": None,
            "var": None,
        }
        self._store_output_activations = store_output_activations
        self._parameters = (
            parameters if parameters is not None else self.empty_parameters()
        )

    def _forward_prop(self, *, input_activations: Activations) -> Activations:
        self._cache.update(
            {
                "input_activations": input_activations,
                "mean": jnp.mean(
                    input_activations,
                    axis=tuple(range(1, input_activations.ndim)),
                    keepdims=True,
                ),
                "var": jnp.var(
                    input_activations,
                    axis=tuple(range(1, input_activations.ndim)),
                    keepdims=True,
                ),
            }
        )
        normalized = (input_activations - self._cache["mean"]) / (
            jnp.sqrt(self._cache["var"]) + EPSILON  # type: ignore[arg-type]
        )
        if self._store_output_activations or self._training:
            self._cache["output_activations"] = normalized
        return self._parameters.weights * normalized + self._parameters.biases

    def _backward_prop(self, *, dZ: D[Activations]) -> D[Activations]:
        if self._cache["mean"] is None:
            raise RuntimeError("Cache not properly populated during forward pass.")
        if self._cache["var"] is None:
            raise RuntimeError("Cache not properly populated during forward pass.")
        if self._cache["output_activations"] is None:
            raise RuntimeError("Cache not properly populated during forward pass.")
        if self._cache["input_activations"] is None:
            raise RuntimeError("Cache not properly populated during forward pass.")

        normalized = self._cache["output_activations"]
        dX_norm = dZ * self._parameters.weights

        if self._training:
            self._cache["dP"] = d(
                self.Parameters(
                    weights=jnp.sum(dZ * normalized, axis=0), biases=jnp.sum(dZ, axis=0)
                )
            )

        std = jnp.sqrt(self._cache["var"])
        x_centered = self._cache["input_activations"] - self._cache["mean"]
        N = jnp.prod(jnp.array(x_centered.shape[1:]))

        dX = dX_norm / std
        dX -= (1 / N) * (
            jnp.sum(dX_norm, axis=tuple(range(1, dX_norm.ndim)), keepdims=True)
            + x_centered
            * jnp.sum(
                dX_norm * x_centered, axis=tuple(range(1, dX_norm.ndim)), keepdims=True
            )
            / self._cache["var"]
        )

        return d(Activations(dX))

    def empty_gradient(self) -> D[ParametersType]:
        return d(
            self.Parameters(
                weights=jnp.zeros_like(self._parameters.weights),
                biases=jnp.zeros_like(self._parameters.biases),
            )
        )

    def empty_parameters(self) -> ParametersType:
        return self.Parameters.empty(input_dimensions=self._input_dimensions)

    def update_parameters(self) -> None:
        if (dP := self._cache["dP"]) is None:
            raise RuntimeError("Gradient is not populated during update.")
        self._parameters = self._parameters + dP
        self._cache["dP"] = None

    def reinitialise(self) -> None:
        self._parameters = self.empty_parameters()

    def gradient_operation(self, f: Callable[[GradLayer], None]) -> None:
        f(self)

    @property
    def cache(self) -> CacheType:
        return self._cache

    @property
    def parameters(self) -> ParametersType:
        return self._parameters

    def serialize(self) -> SupportsDeserialize[LayerNorm]:
        return self.Serialized(
            layer_id=self._layer_id,
            input_dimensions=tuple(self._input_dimensions),
            parameters=self._parameters,
        )

    @property
    def parameter_count(self) -> int:
        return self._parameters.weights.size + self._parameters.biases.size

    @property
    def parameter_nbytes(self) -> int:
        return self._parameters.weights.nbytes + self._parameters.biases.nbytes

    def write_serialized_parameters(self, buffer: IO[bytes]) -> None:
        self._write_header(buffer)
        if self._cache is None or self._cache["dP"] is None:
            raise RuntimeError("Cache is not populated during serialization.")
        buffer.write(memoryview(self._cache["dP"].weights))
        buffer.write(memoryview(self._cache["dP"].biases))

    def read_serialized_parameters(self, data: IO[bytes]) -> None:
        if (layer_id := self.get_layer_id(data)) != self._layer_id:
            raise BadLayerId(f"Layer ID mismatch: {layer_id} != {self._layer_id}")
        update = self._parameters.from_bytes(data)
        if self._cache["dP"] is None:
            self._cache["dP"] = d(update)
        else:
            self._cache["dP"] += d(update)
