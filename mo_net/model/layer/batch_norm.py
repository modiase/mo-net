from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Self

import numpy as np

from mo_net.model.layer.base import (
    Hidden,
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


type ParametersType = Parameters


class Cache(GradCache):
    input_activations: Activations | None
    mean: np.ndarray | None
    output_activations: Activations | None
    var: np.ndarray | None
    batch_size: int | None
    spatial_size: int | None


type CacheType = Cache


def _compute_batch_stats(
    input_activations: np.ndarray, is_spatial: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    if is_spatial:
        # For conv inputs: reshape to (N*H*W, C) and compute per-channel stats
        n_channels = input_activations.shape[1]
        reshaped = input_activations.transpose(0, 2, 3, 1).reshape(-1, n_channels)
        return np.mean(reshaped, axis=0), np.var(reshaped, axis=0)
    else:
        # For FC inputs: compute stats across batch dimension
        return np.mean(input_activations, axis=0), np.var(input_activations, axis=0)


def _update_running_stats(
    running_mean: np.ndarray,
    running_var: np.ndarray,
    batch_mean: np.ndarray,
    batch_var: np.ndarray,
    momentum: float,
) -> tuple[np.ndarray, np.ndarray]:
    new_mean = momentum * running_mean + (1 - momentum) * batch_mean
    new_var = momentum * running_var + (1 - momentum) * batch_var
    return new_mean, new_var


class BatchNorm(Hidden, GradLayer[ParametersType, CacheType]):
    """https://arxiv.org/abs/1502.03167"""

    _EPSILON: ClassVar[float] = 1e-8
    Parameters = Parameters
    Cache = Cache

    @dataclass(frozen=True, kw_only=True)
    class Serialized:
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
        parameters: ParametersType | None = None,
        running_mean: np.ndarray | None = None,
        running_variance: np.ndarray | None = None,
        store_output_activations: bool = False,
        training: bool = True,
    ):
        super().__init__(
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
            "spatial_size": None,
        }
        self._store_output_activations = store_output_activations
        self._parameters = (
            parameters if parameters is not None else self.empty_parameters()
        )

    def _forward_prop(self, *, input_activations: Activations) -> Activations:
        if len(input_activations.shape) != 2:
            raise ValueError(
                f"BatchNorm expects 2D input (batch_size, features), got shape {input_activations.shape}"
            )

        batch_mean, batch_variance = _compute_batch_stats(input_activations)

        if self._training:
            self._running_mean, self._running_variance = _update_running_stats(
                self._running_mean,
                self._running_variance,
                batch_mean,
                batch_variance,
                self._momentum,
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

    def _backward_prop(self, *, dZ: D[Activations]) -> D[Activations]:
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
            self._cache["dP"] = d(self.Parameters(_gamma=-d_gamma, _beta=-d_beta))

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
            input_dimensions=tuple(self._input_dimensions),
            momentum=self._momentum,
            parameters=self._parameters,
            running_mean=self._running_mean,
            running_variance=self._running_variance,
        )

    @property
    def parameter_count(self) -> int:
        return self._parameters._gamma.size + self._parameters._beta.size


class BatchNorm2D(BatchNorm):
    def __init__(
        self,
        *,
        input_dimensions: Dimensions,
        momentum: float = 0.9,
        parameters: ParametersType | None = None,
        running_mean: np.ndarray | None = None,
        running_variance: np.ndarray | None = None,
        store_output_activations: bool = False,
        training: bool = True,
    ):
        if len(input_dimensions) != 3:
            raise ValueError(
                f"BatchNorm2D expects 3D input dimensions (channels, height, width), got {input_dimensions}"
            )

        n_channels = input_dimensions[0]
        param_shape = (n_channels,)

        super().__init__(
            input_dimensions=input_dimensions,
            momentum=momentum,
            parameters=parameters,
            running_mean=running_mean
            if running_mean is not None
            else np.zeros(param_shape),
            running_variance=running_variance
            if running_variance is not None
            else np.ones(param_shape),
            store_output_activations=store_output_activations,
            training=training,
        )

    def _forward_prop(self, *, input_activations: Activations) -> Activations:
        if len(input_activations.shape) != 4:
            raise ValueError(
                f"BatchNorm2D expects 4D input (batch_size, channels, height, width), got shape {input_activations.shape}"
            )

        batch_size, n_channels = input_activations.shape[0], input_activations.shape[1]
        spatial_size = input_activations.shape[2] * input_activations.shape[3]

        batch_mean, batch_variance = _compute_batch_stats(
            input_activations, is_spatial=True
        )

        if self._training:
            self._running_mean, self._running_variance = _update_running_stats(
                self._running_mean,
                self._running_variance,
                batch_mean,
                batch_variance,
                self._momentum,
            )

            norm_mean = batch_mean.reshape(1, n_channels, 1, 1)
            norm_var = batch_variance.reshape(1, n_channels, 1, 1)
            normalised_activations = (input_activations - norm_mean) / np.sqrt(
                norm_var + self._EPSILON
            )

            self._cache.update(
                {
                    "input_activations": input_activations,
                    "mean": batch_mean,
                    "var": batch_variance,
                    "batch_size": batch_size,
                    "spatial_size": spatial_size,
                }
            )
        else:
            norm_mean = self._running_mean.reshape(1, n_channels, 1, 1)
            norm_var = self._running_variance.reshape(1, n_channels, 1, 1)
            normalised_activations = (input_activations - norm_mean) / np.sqrt(
                norm_var + self._EPSILON
            )

        if self._store_output_activations or self._training:
            self._cache["output_activations"] = normalised_activations

        gamma = self._parameters._gamma.reshape(1, n_channels, 1, 1)
        beta = self._parameters._beta.reshape(1, n_channels, 1, 1)

        return gamma * normalised_activations + beta

    def _backward_prop(self, *, dZ: D[Activations]) -> D[Activations]:
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

        batch_size = self._cache["batch_size"]
        spatial_size = self._cache["spatial_size"]
        n_channels = mean.shape[0]

        dZ_reshaped = dZ.transpose(0, 2, 3, 1).reshape(-1, n_channels)
        output_reshaped = output_activations.transpose(0, 2, 3, 1).reshape(
            -1, n_channels
        )

        d_gamma = np.sum(dZ_reshaped * output_reshaped, axis=0)
        d_beta = np.sum(dZ_reshaped, axis=0)

        if self._training:
            self._cache["dP"] = d(self.Parameters(_gamma=-d_gamma, _beta=-d_beta))

        mean_reshaped = mean.reshape(1, n_channels, 1, 1)
        dX_norm_reshaped = dZ_reshaped * self._parameters._gamma
        x_minus_mean_reshaped = (
            (input_activations - mean_reshaped)
            .transpose(0, 2, 3, 1)
            .reshape(-1, n_channels)
        )
        total_elements = batch_size * spatial_size

        d_var = -0.5 * np.sum(
            dX_norm_reshaped
            * x_minus_mean_reshaped
            * np.power(var + self._EPSILON, -1.5),
            axis=0,
        )

        d_mean = (
            -np.sum(dX_norm_reshaped / np.sqrt(var + self._EPSILON), axis=0)
            + d_var * np.sum(-2 * x_minus_mean_reshaped, axis=0) / total_elements
        )

        dX_reshaped = (
            dX_norm_reshaped / np.sqrt(var + self._EPSILON)
            + d_var * 2 * x_minus_mean_reshaped / total_elements
            + d_mean / total_elements
        )

        dX = dX_reshaped.reshape(
            batch_size,
            input_activations.shape[2],
            input_activations.shape[3],
            n_channels,
        ).transpose(0, 3, 1, 2)

        return dX

    def empty_parameters(self) -> ParametersType:
        n_channels = self._input_dimensions[0]
        return self.Parameters(
            _gamma=np.ones(n_channels),
            _beta=np.zeros(n_channels),
        )
