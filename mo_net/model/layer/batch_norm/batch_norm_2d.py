from __future__ import annotations

from collections.abc import Callable

from dataclasses import dataclass
from typing import IO, Self

import numpy as np

from mo_net.constants import EPSILON
from mo_net.model.layer.base import BadLayerId, ParametrisedHidden
from mo_net.protos import (
    Activations,
    D,
    Dimensions,
    GradCache,
    GradLayer,
    SupportsGradientOperations,
    d,
)


@dataclass(frozen=True, kw_only=True)
class Parameters2D(SupportsGradientOperations):
    """Parameters for BatchNorm2D, with per-channel scaling and bias."""

    weights: np.ndarray  # Shape: (n_channels,)
    biases: np.ndarray  # Shape: (n_channels,)

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

    def __pow__(self, other: float | int) -> Self:
        match other:
            case float() | int():
                return self.__class__(
                    weights=self.weights**other,
                    biases=self.biases**other,
                )
            case _:
                return NotImplemented

    @classmethod
    def empty(cls, *, input_dimensions: Dimensions) -> Self:
        return cls(
            weights=np.ones(input_dimensions[0]),  # Only need per-channel weights
            biases=np.zeros(input_dimensions[0]),  # Only need per-channel biases
        )

    def from_bytes(self, data: IO[bytes]) -> Self:
        return self.__class__(
            weights=np.frombuffer(
                data.read(self.weights.nbytes), dtype=self.weights.dtype
            ).reshape(self.weights.shape),
            biases=np.frombuffer(
                data.read(self.biases.nbytes), dtype=self.biases.dtype
            ).reshape(self.biases.shape),
        )


type ParametersType = Parameters2D


class Cache(GradCache[ParametersType]):
    input_activations: Activations | None
    mean: np.ndarray | None
    output_activations: Activations | None
    var: np.ndarray | None
    batch_size: int | None
    spatial_size: int | None


type CacheType = Cache


class BatchNorm2D(ParametrisedHidden[ParametersType, CacheType]):
    """Batch normalization for 2D inputs (e.g. Conv2D outputs)."""

    Parameters = Parameters2D
    Cache = Cache

    @dataclass(frozen=True, kw_only=True)
    class Serialized:
        layer_id: str
        input_dimensions: tuple[int, ...]
        momentum: float
        parameters: Parameters2D
        running_mean: np.ndarray
        running_variance: np.ndarray

        def deserialize(
            self,
            *,
            training: bool = False,
        ) -> BatchNorm2D:
            return BatchNorm2D(
                layer_id=self.layer_id,
                input_dimensions=self.input_dimensions,
                momentum=self.momentum,
                parameters=self.parameters,
                running_mean=self.running_mean,
                running_variance=self.running_variance,
                training=training,
            )

    def __init__(
        self,
        *,
        input_dimensions: Dimensions,
        momentum: float = 0.9,
        layer_id: str | None = None,
        parameters: Parameters2D | None = None,
        running_mean: np.ndarray | None = None,
        running_variance: np.ndarray | None = None,
        store_output_activations: bool = False,
        training: bool = True,
    ):
        if len(input_dimensions) != 3:
            raise ValueError(
                f"input_dimensions must be a 3-tuple: (channels, height, width). Got {input_dimensions}."
            )
        super().__init__(
            layer_id=layer_id,
            input_dimensions=input_dimensions,
            output_dimensions=input_dimensions,
        )
        self._momentum = momentum
        self._running_mean = (
            running_mean if running_mean is not None else np.zeros(input_dimensions[0])
        )
        self._running_variance = (
            running_variance
            if running_variance is not None
            else np.ones(input_dimensions[0])
        )
        self._training = training
        self._cache: CacheType = {
            "dP": None,
            "input_activations": None,
            "mean": None,
            "output_activations": None,
            "batch_size": None,
            "var": None,
            "spatial_size": None,
        }
        self._store_output_activations = store_output_activations
        self._parameters = (
            parameters if parameters is not None else self.empty_parameters()
        )

    def _forward_prop(self, *, input_activations: Activations) -> Activations:
        batch_size, n_channels = input_activations.shape[:2]
        reshaped_input = input_activations.reshape(batch_size, n_channels, -1)
        batch_mean = np.mean(reshaped_input, axis=(0, 2))
        batch_variance = np.var(reshaped_input, axis=(0, 2))

        if self._training:
            self._running_mean = (
                self._momentum * self._running_mean + (1 - self._momentum) * batch_mean
            )
            self._running_variance = (
                self._momentum * self._running_variance
                + (1 - self._momentum) * batch_variance
            )

            normalised_activations = (
                input_activations - batch_mean[:, None, None]
            ) / np.sqrt(batch_variance[:, None, None] + EPSILON)
            self._cache.update({
                "input_activations": input_activations,
                "mean": batch_mean,
                "var": batch_variance,
                "batch_size": batch_size,
            })
        else:
            normalised_activations = (
                input_activations - self._running_mean[:, None, None]
            ) / np.sqrt(self._running_variance[:, None, None] + EPSILON)

        if self._store_output_activations or self._training:
            self._cache["output_activations"] = normalised_activations

        return (
            self._parameters.weights[:, None, None] * normalised_activations
            + self._parameters.biases[:, None, None]
        )

    def _backward_prop(self, *, dZ: D[Activations]) -> D[Activations]:
        if (mean := self._cache["mean"]) is None:
            raise RuntimeError("Mean is not populated during backward pass.")
        if (var := self._cache["var"]) is None:
            raise RuntimeError("Variance is not populated during backward pass.")
        if (input_activations := self._cache["input_activations"]) is None:
            raise RuntimeError(
                "input_activations is not populated during backward pass."
            )
        if (output_activations := self._cache["output_activations"]) is None:
            raise RuntimeError(
                "output_activations is not populated during backward pass."
            )

        dX_norm = dZ * self._parameters.weights[:, None, None]  # type: ignore[operator]
        d_weights = np.sum(dZ * output_activations, axis=(0, 2, 3))
        d_beta = np.sum(dZ, axis=(0, 2, 3))

        if self._training:
            self._cache["dP"] = d(
                self.Parameters(
                    weights=-d_weights,
                    biases=-d_beta,
                )
            )

        batch_size = self._cache["batch_size"]
        if batch_size is None:
            raise RuntimeError("batch size not set during forward pass.")
        spatial_size = input_activations.shape[2] * input_activations.shape[3]

        d_batch_variance = -0.5 * np.sum(
            dX_norm.reshape(batch_size, -1, spatial_size)
            * (input_activations.reshape(batch_size, -1, spatial_size) - mean[:, None])
            * np.power(var[:, None] + EPSILON, -1.5),
            axis=(0, 2),
        )

        d_batch_mean = -np.sum(
            dX_norm.reshape(batch_size, -1, spatial_size)
            / np.sqrt(var[:, None] + EPSILON),
            axis=(0, 2),
        ) + d_batch_variance * np.sum(
            -2
            * (input_activations.reshape(batch_size, -1, spatial_size) - mean[:, None]),
            axis=(0, 2),
        ) / (batch_size * spatial_size)

        dX = (
            dX_norm / np.sqrt(var[:, None, None] + EPSILON)
            + d_batch_variance[:, None, None]
            * 2
            * (input_activations - mean[:, None, None])
            / (batch_size * spatial_size)
            + d_batch_mean[:, None, None] / (batch_size * spatial_size)
        )

        return d(Activations(dX))

    def empty_parameters(self) -> Parameters2D:
        return self.Parameters(
            weights=np.ones(self._input_dimensions[0]),
            biases=np.zeros(self._input_dimensions[0]),
        )

    def empty_gradient(self) -> D[ParametersType]:
        return d(
            self.Parameters(
                weights=np.zeros_like(self._parameters.weights),
                biases=np.zeros_like(self._parameters.biases),
            )
        )

    @property
    def parameters(self) -> ParametersType:
        return self._parameters

    @property
    def cache(self) -> CacheType:
        return self._cache

    def gradient_operation(self, f: Callable[[GradLayer], None]) -> None:
        f(self)

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
