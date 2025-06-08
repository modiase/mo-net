from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import IO, ClassVar, Self

import numpy as np

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
    _gamma: np.ndarray
    _beta: np.ndarray

    def __getitem__(
        self, index: int | tuple[int, ...]
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._gamma[index], self._beta[index]

    def __pow__(self, other: float | int) -> Self:
        match other:
            case float() | int():
                return self.__class__(
                    _gamma=self._gamma**other,
                    _beta=self._beta**other,
                )
            case _:
                return NotImplemented

    def __truediv__(self, other: Self | float | int) -> Self:
        match other:
            case float() | int():
                return self.__mul__(1 / other)
            case self.__class__():
                return self.__class__(
                    _gamma=self._gamma / other._gamma,
                    _beta=self._beta / other._beta,
                )
            case _:
                return NotImplemented

    def __add__(self, other: Self | float | int) -> Self:
        match other:
            case float() | int():
                return self.__class__(
                    _gamma=self._gamma + other,
                    _beta=self._beta + other,
                )
            case self.__class__():
                return self.__class__(
                    _gamma=self._gamma + other._gamma,
                    _beta=self._beta + other._beta,
                )
            case _:
                return NotImplemented

    def __radd__(self, other: Self | float | int) -> Self:
        match other:
            case float() | int():
                return self.__class__(
                    _gamma=other + self._gamma,
                    _beta=other + self._beta,
                )
            case _:
                return NotImplemented

    def __neg__(self) -> Self:
        return self.__class__(
            _gamma=-self._gamma,
            _beta=-self._beta,
        )

    def __sub__(self, other: Self | float) -> Self:
        match other:
            case float() | int():
                return self.__class__(
                    _gamma=self._gamma - other,
                    _beta=self._beta - other,
                )
            case self.__class__():
                return self.__class__(
                    _gamma=self._gamma - other._gamma,
                    _beta=self._beta - other._beta,
                )
            case _:
                return NotImplemented

    def __mul__(self, other: float | Self) -> Self:
        match other:
            case float() | int():
                return self.__class__(
                    _gamma=self._gamma * other,
                    _beta=self._beta * other,
                )
            case self.__class__():
                return self.__class__(
                    _gamma=self._gamma * other._gamma,
                    _beta=self._beta * other._beta,
                )
            case _:
                return NotImplemented

    def __rmul__(self, other: float | Self) -> Self:
        return self.__mul__(other)

    @classmethod
    def empty(cls, *, input_dimensions: Dimensions) -> Self:
        return cls(
            _gamma=np.ones(input_dimensions),
            _beta=np.zeros(input_dimensions),
        )

    def from_bytes(self, data: IO[bytes]) -> Self:
        return self.__class__(
            _gamma=np.frombuffer(
                data.read(self._gamma.nbytes), dtype=self._gamma.dtype
            ).reshape(self._gamma.shape),
            _beta=np.frombuffer(
                data.read(self._beta.nbytes), dtype=self._beta.dtype
            ).reshape(self._beta.shape),
        )


type ParametersType = Parameters


class Cache(GradCache):
    input_activations: Activations | None
    mean: np.ndarray | None
    output_activations: Activations | None
    var: np.ndarray | None
    batch_size: int | None


type CacheType = Cache


class BatchNorm(ParametrisedHidden[ParametersType, CacheType]):
    """https://arxiv.org/abs/1502.03167"""

    _EPSILON: ClassVar[float] = 1e-8
    Parameters = Parameters
    Cache = Cache

    @dataclass(frozen=True, kw_only=True)
    class Serialized:
        layer_id: str
        input_dimensions: tuple[int, ...]
        momentum: float
        parameters: Parameters
        running_mean: np.ndarray
        running_variance: np.ndarray

        def deserialize(
            self,
            *,
            training: bool = False,
        ) -> BatchNorm:
            return BatchNorm(
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
        parameters: ParametersType | None = None,
        running_mean: np.ndarray | None = None,
        running_variance: np.ndarray | None = None,
        store_output_activations: bool = False,
        training: bool = True,
    ):
        super().__init__(
            layer_id=layer_id,
            input_dimensions=input_dimensions,
            output_dimensions=input_dimensions,
        )
        self._momentum = momentum
        self._running_mean = (
            running_mean if running_mean is not None else np.zeros(input_dimensions)
        )
        self._running_variance = (
            running_variance
            if running_variance is not None
            else np.ones(input_dimensions)
        )
        self._training = training
        self._cache: CacheType = {
            "dP": None,
            "input_activations": None,
            "mean": None,
            "output_activations": None,
            "batch_size": None,
            "var": None,
        }
        self._store_output_activations = store_output_activations
        self._parameters = (
            parameters if parameters is not None else self.empty_parameters()
        )

    def _forward_prop(self, *, input_activations: Activations) -> Activations:
        batch_mean = np.mean(input_activations, axis=0)
        batch_variance = np.var(input_activations, axis=0)

        if self._training:
            self._running_mean = (
                self._momentum * self._running_mean + (1 - self._momentum) * batch_mean
            )
            self._running_variance = (
                self._momentum * self._running_variance
                + (1 - self._momentum) * batch_variance
            )

            normalised_activations = (input_activations - batch_mean) / np.sqrt(
                batch_variance + self._EPSILON
            )
            self._cache.update(
                {
                    "input_activations": input_activations,
                    "mean": batch_mean,
                    "var": batch_variance,
                    "batch_size": input_activations.shape[0],
                }
            )
        else:
            normalised_activations = (input_activations - self._running_mean) / np.sqrt(
                self._running_variance + self._EPSILON
            )

        if self._store_output_activations or self._training:
            self._cache["output_activations"] = normalised_activations

        return self._parameters._gamma * normalised_activations + self._parameters._beta

    def _backward_prop(
        self,
        *,
        dZ: D[Activations],
    ) -> D[Activations]:
        if (mean := self._cache["mean"]) is None:
            raise RuntimeError("Mean is not populated during backward pass.")
        if (var := self._cache["var"]) is None:
            raise RuntimeError("Variance is not populated during backward pass.")
        if (output_activations := self._cache["output_activations"]) is None:
            raise RuntimeError(
                "Output activations are not populated during backward pass."
            )
        if (input_activations := self._cache["input_activations"]) is None:
            raise RuntimeError(
                "input_activations is not populated during backward pass."
            )

        dX_norm = dZ * self._parameters._gamma
        d_gamma = np.sum(dZ * output_activations, axis=0)
        d_beta = np.sum(dZ, axis=0)
        if self._training:
            self._cache["dP"] = d(
                self.Parameters(
                    _gamma=-d_gamma,
                    _beta=-d_beta,
                )
            )
        batch_size = self._cache["batch_size"]

        d_batch_variance = -0.5 * np.sum(
            dX_norm * (input_activations - mean) * np.power(var + self._EPSILON, -1.5),
            axis=0,
        )
        d_batch_mean = (
            -np.sum(dX_norm / np.sqrt(var + self._EPSILON), axis=0)
            + d_batch_variance
            * np.sum(-2 * (input_activations - mean), axis=0)
            / batch_size
        )

        dX = (
            dX_norm / np.sqrt(var + self._EPSILON)
            + d_batch_variance * 2 * (input_activations - mean) / batch_size
            + d_batch_mean / batch_size
        )

        return dX

    def empty_gradient(self) -> D[ParametersType]:
        return d(
            self.Parameters(
                _gamma=np.zeros_like(self._parameters._gamma),
                _beta=np.zeros_like(self._parameters._beta),
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

    def serialize(self) -> SupportsDeserialize[BatchNorm]:
        return self.Serialized(
            layer_id=self._layer_id,
            input_dimensions=tuple(self._input_dimensions),
            momentum=self._momentum,
            parameters=self._parameters,
            running_mean=self._running_mean,
            running_variance=self._running_variance,
        )

    @property
    def parameter_count(self) -> int:
        return self._parameters._gamma.size + self._parameters._beta.size

    @property
    def parameter_nbytes(self) -> int:
        return self._parameters._gamma.nbytes + self._parameters._beta.nbytes

    def serialize_parameters(self, buffer: IO[bytes]) -> None:
        self._write_header(buffer)
        buffer.write(memoryview(self._cache["dP"]._gamma))
        buffer.write(memoryview(self._cache["dP"]._beta))

    def deserialize_parameters(self, data: IO[bytes]) -> None:
        if (layer_id := self.get_layer_id(data)) != self._layer_id:
            raise BadLayerId(f"Layer ID mismatch: {layer_id} != {self._layer_id}")
        update = self._parameters.from_bytes(data)
        if self._cache["dP"] is None:
            self._cache["dP"] = d(update)
        else:
            self._cache["dP"] += d(update)
