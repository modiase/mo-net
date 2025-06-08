from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TypedDict, assert_never, cast

import numpy as np
from numpy.lib.stride_tricks import as_strided

from mo_net.model.layer.base import Hidden
from mo_net.protos import (
    Activations,
    D,
    Dimensions,
    GradLayer,
    SupportsGradientOperations,
    d,
)


class Cache(TypedDict):
    input_activations: Activations | None
    output_activations: Activations | None
    dP: D[Parameters] | None


type CacheType = Cache


@dataclass(frozen=True, kw_only=True)
class Parameters(SupportsGradientOperations):
    weights: np.ndarray
    biases: np.ndarray

    def __add__(self, other: Parameters | float | int) -> Parameters:
        match other:
            case Parameters():
                return Parameters(
                    weights=self.weights + other.weights,
                    biases=self.biases + other.biases,
                )
            case float() | int():
                return Parameters(
                    weights=self.weights + other,
                    biases=self.biases + other,
                )
            case never:
                assert_never(never)

    def __radd__(self, other: Parameters | float | int) -> Parameters:
        match other:
            case Parameters() | float() | int():
                return other + self
            case never:
                assert_never(never)

    def __mul__(self, other: Parameters | float | int) -> Parameters:
        match other:
            case Parameters():
                return Parameters(
                    weights=self.weights * other.weights,
                    biases=self.biases * other.biases,
                )
            case float() | int():
                return Parameters(
                    weights=self.weights * other,
                    biases=self.biases * other,
                )
            case never:
                assert_never(never)

    def __rmul__(self, other: Parameters | float | int) -> Parameters:
        match other:
            case Parameters() | float() | int():
                return self * other
            case never:
                assert_never(never)

    def __truediv__(self, other: Parameters | float | int) -> Parameters:
        match other:
            case Parameters():
                return Parameters(
                    weights=self.weights / other.weights,
                    biases=self.biases / other.biases,
                )
            case float() | int():
                return Parameters(
                    weights=self.weights / other,
                    biases=self.biases / other,
                )
            case never:
                assert_never(never)

    def __neg__(self) -> Parameters:
        return Parameters(
            weights=-self.weights,
            biases=-self.biases,
        )

    def __pow__(self, other: float | int) -> Parameters:
        match other:
            case float() | int():
                return Parameters(
                    weights=self.weights**other,
                    biases=self.biases**other,
                )
            case never:
                assert_never(never)

    def __sub__(self, other: Parameters | float | int) -> Parameters:
        return self + (-other)

    @classmethod
    def random(
        cls,
        n_kernels: int,
        in_channels: int,
        in_height: int,
        in_width: int,
    ) -> Parameters:
        return cls(
            weights=np.random.uniform(
                -np.sqrt(1 / (in_channels * in_height * in_width)),
                np.sqrt(1 / (in_channels * in_height * in_width)),
                (n_kernels, in_channels, in_height, in_width),
            ),
            biases=np.random.rand(n_kernels),
        )

    @classmethod
    def ones(
        cls,
        n_kernels: int,
        in_channels: int,
        in_height: int,
        in_width: int,
    ) -> Parameters:
        return cls(
            weights=np.ones((n_kernels, in_channels, in_height, in_width)),
            biases=np.zeros(n_kernels),
        )

    @classmethod
    def xavier(
        cls,
        n_kernels: int,
        in_channels: int,
        in_height: int,
        in_width: int,
    ) -> Parameters:
        fan_in = in_channels * in_height * in_width
        limit = np.sqrt(6.0 / fan_in)
        return cls(
            weights=np.random.uniform(
                -limit, limit, (n_kernels, in_channels, in_height, in_width)
            ),
            biases=np.zeros(n_kernels),
        )

    @classmethod
    def he(
        cls,
        n_kernels: int,
        in_channels: int,
        in_height: int,
        in_width: int,
    ) -> Parameters:
        return cls(
            weights=np.random.normal(
                0,
                np.sqrt(2 / (in_channels * in_height * in_width)),
                (n_kernels, in_channels, in_height, in_width),
            ),
            biases=np.zeros(n_kernels),
        )


type ParametersType = Parameters
type KernelInitFn = Callable[[int, int, int, int], ParametersType]


class Convolution2D(Hidden):
    Cache = Cache
    Parameters = Parameters

    @dataclass(frozen=True, kw_only=True)
    class Serialized:
        input_dimensions: tuple[int, int, int]
        n_kernels: int
        kernel_size: tuple[int, int]
        stride: tuple[int, int]
        output_dimensions: tuple[int, int, int]
        parameters: Parameters

        def deserialize(self, *, training: bool = False) -> Convolution2D:
            def _kernel_init_fn(
                *args: object, **kwargs: Mapping[str, object]
            ) -> Parameters:
                del args, kwargs  # unused
                return self.parameters

            return Convolution2D(
                input_dimensions=self.input_dimensions,
                n_kernels=self.n_kernels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                kernel_init_fn=_kernel_init_fn,
                freeze_parameters=not training,
            )

    def __init__(
        self,
        *,
        input_dimensions: Dimensions,
        n_kernels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        # padding: int | tuple[int, int] = 0, # TODO: Implement padding.
        kernel_init_fn: KernelInitFn = Parameters.he,
        freeze_parameters: bool = False,
        clip_gradients: bool = False,
        weight_max_norm: float = 1.0,
        bias_max_norm: float = 1.0,
    ):
        if len(input_dimensions) != 3:
            raise ValueError(
                f"input_dimensions must be a 3-tuple: (channel, dim_x, dim_y). Got {input_dimensions}."
            )
        if not isinstance(kernel_size, (int, tuple)):
            raise ValueError("Invalid kernel_size. Must be an integer or a pair.")
        self._kernel_height, self._kernel_width = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        if not isinstance(stride, (int, tuple)):
            raise ValueError("Invalid stride. Must be an integer or a pair.")
        self._stride_x, self._stride_y = (
            stride if isinstance(stride, tuple) else (stride, stride)
        )
        self._out_height = (
            input_dimensions[1] - self._kernel_height
        ) // self._stride_y + 1
        self._out_width = (
            input_dimensions[2] - self._kernel_width
        ) // self._stride_x + 1
        self._in_channels = input_dimensions[0]
        output_dimensions = (
            n_kernels,  # Channel
            self._out_height,  # Y
            self._out_width,  # X
        )
        super().__init__(
            input_dimensions=input_dimensions, output_dimensions=output_dimensions
        )
        self._freeze_parameters = freeze_parameters
        self._n_kernels = n_kernels
        self._clip_gradients = clip_gradients
        self._weight_max_norm = weight_max_norm
        self._bias_max_norm = bias_max_norm
        self._parameters = kernel_init_fn(
            n_kernels,
            self._in_channels,
            self._kernel_height,
            self._kernel_width,
        )
        self._cache = self.Cache(
            input_activations=None, output_activations=None, dP=None
        )

    def _forward_prop(self, *, input_activations: Activations) -> Activations:
        batch_size = input_activations.shape[0]
        self._cache["input_activations"] = input_activations

        input_windows = as_strided(
            input_activations,
            shape=(
                batch_size,
                self._in_channels,
                self._out_height,
                self._out_width,
                self._kernel_height,
                self._kernel_width,
            ),
            strides=(
                input_activations.strides[0],
                input_activations.strides[1],
                input_activations.strides[2] * self._stride_y,
                input_activations.strides[3] * self._stride_x,
                input_activations.strides[2],
                input_activations.strides[3],
            ),
            writeable=False,
        )

        output = np.einsum(
            "bchwij,kcij->bkhw", input_windows, self._parameters.weights, optimize=True
        )

        output += self._parameters.biases[np.newaxis, :, np.newaxis, np.newaxis]

        return Activations(output)

    def _backward_prop(self, *, dZ: D[Activations]) -> D[Activations]:
        batch_size, n_kernels, dZ_height, dZ_width = cast(np.ndarray, dZ).shape
        input_activations = self._cache["input_activations"]
        if input_activations is None:
            raise ValueError("input_activations not set during forward_prop.")

        input_windows = as_strided(
            input_activations,
            shape=(
                batch_size,
                self._in_channels,
                dZ_height,
                dZ_width,
                self._kernel_height,
                self._kernel_width,
            ),
            strides=(
                input_activations.strides[0],
                input_activations.strides[1],
                input_activations.strides[2] * self._stride_y,
                input_activations.strides[3] * self._stride_x,
                input_activations.strides[2],
                input_activations.strides[3],
            ),
            writeable=False,
        )

        dK = np.einsum("bchwij,bmhw->mcij", input_windows, cast(np.ndarray, dZ))

        db = np.sum(cast(np.ndarray, dZ), axis=(0, 2, 3))

        if self._clip_gradients:
            dK *= min(
                1.0,
                self._weight_max_norm * np.sqrt(dK.size) / np.linalg.norm(dK),
            )
            db *= min(
                1.0,
                self._bias_max_norm * np.sqrt(db.size) / np.linalg.norm(db),
            )

        pad_h = self._kernel_height - 1
        pad_w = self._kernel_width - 1
        dZ_padded = np.pad(
            cast(np.ndarray, dZ),
            ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
            mode="constant",
            constant_values=0,
        )

        in_height, in_width = input_activations.shape[2:]
        dZ_windows = as_strided(
            dZ_padded,
            shape=(
                batch_size,
                n_kernels,
                in_height,
                in_width,
                self._kernel_height,
                self._kernel_width,
            ),
            strides=(
                dZ_padded.strides[0],
                dZ_padded.strides[1],
                dZ_padded.strides[2],
                dZ_padded.strides[3],
                dZ_padded.strides[2],
                dZ_padded.strides[3],
            ),
            writeable=False,
        )

        flipped_weights = np.flip(self._parameters.weights, axis=(2, 3))
        dX = np.einsum("bmhwij,mcij->bchw", dZ_windows, flipped_weights)

        self._cache["dP"] = d(self.Parameters(weights=dK, biases=db))
        return d(Activations(dX))

    def gradient_operation(self, f: Callable[[GradLayer], None]) -> None:
        f(self)

    def empty_gradient(self) -> D[ParametersType]:
        return d(
            self.Parameters(
                weights=np.zeros_like(self._parameters.weights),
                biases=np.zeros_like(self._parameters.biases),
            )
        )

    @property
    def cache(self) -> CacheType:
        return self._cache

    @property
    def parameters(self) -> ParametersType:
        return self._parameters

    def update_parameters(self) -> None:
        if self._cache["dP"] is None:
            raise ValueError("Gradient not set during backward pass.")
        if not self._freeze_parameters:
            self._parameters = self._parameters + self._cache["dP"]
        self._cache["dP"] = None

    def serialize(self) -> Convolution2D.Serialized:
        channels, height, width = self.input_dimensions
        return Convolution2D.Serialized(
            input_dimensions=(channels, height, width),
            n_kernels=self._n_kernels,
            kernel_size=(self._kernel_height, self._kernel_width),
            stride=(self._stride_x, self._stride_y),
            output_dimensions=(self._n_kernels, self._out_height, self._out_width),
            parameters=self._parameters,
        )
