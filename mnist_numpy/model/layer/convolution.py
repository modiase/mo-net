from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TypedDict, assert_never

import numpy as np
from scipy import signal

from mnist_numpy.model.layer.base import Hidden
from mnist_numpy.protos import Activations, D, Dimensions, GradLayer, d


class Cache(TypedDict):
    output_activations: Activations | None
    dP: D[Activations] | None


@dataclass(frozen=True, kw_only=True)
class Parameters:
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

    @classmethod
    def random(
        cls,
        n_kernels: int,
        in_channels: int,
        in_height: int,
        in_width: int,
        out_height: int,
        out_width: int,
    ) -> Parameters:
        return cls(
            weights=np.random.rand(n_kernels, in_channels, in_height, in_width),
            biases=np.random.rand(n_kernels, out_height, out_width),
        )

    @classmethod
    def ones(
        cls,
        n_kernels: int,
        in_channels: int,
        in_height: int,
        in_width: int,
        out_height: int,
        out_width: int,
    ) -> Parameters:
        return cls(
            weights=np.ones((n_kernels, in_channels, in_height, in_width)),
            biases=np.zeros((n_kernels, out_height, out_width)),
        )

    @classmethod
    def xavier(
        cls,
        n_kernels: int,
        in_channels: int,
        in_height: int,
        in_width: int,
        out_height: int,
        out_width: int,
    ) -> Parameters:
        return cls(
            weights=np.random.rand(n_kernels, in_channels, in_height, in_width)
            * np.sqrt(1 / (in_channels * in_height * in_width)),
            biases=np.zeros((n_kernels, out_height, out_width)),
        )

    @classmethod
    def he(
        cls,
        n_kernels: int,
        in_channels: int,
        in_height: int,
        in_width: int,
        out_height: int,
        out_width: int,
    ) -> Parameters:
        return cls(
            weights=np.random.normal(
                0,
                np.sqrt(2 / (in_channels * in_height * in_width)),
                (n_kernels, in_channels, in_height, in_width),
            ),
            biases=np.zeros((n_kernels, out_height, out_width)),
        )


type ParametersType = Parameters
type KernelInitFn = Callable[[int, int, int, int, int, int], ParametersType]


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
    ):
        if len(input_dimensions) != 3:
            raise ValueError(
                "input_dimensions must be a 3-tuple: (channel, dim_x, dim_y)."
            )
        if not isinstance(kernel_size, (int, tuple)):
            raise ValueError("Invalid kernel_size. Must be an integer or a pair.")
        self._kernel_width, self._kernel_height = (
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
        self._parameters = kernel_init_fn(
            n_kernels,
            self._in_channels,
            self._kernel_height,
            self._kernel_width,
            self._out_height,
            self._out_width,
        )
        self._cache = self.Cache(output_activations=None, dP=None)

    def _pad_input(self, input_activations: Activations) -> Activations:
        if self._padding_x == 0 and self._padding_y == 0:
            return input_activations
        return np.pad(
            input_activations,
            (
                (0, 0),
                (0, 0),
                (self._padding_y, self._padding_y),
                (self._padding_x, self._padding_x),
            ),
            mode="constant",
            constant_values=(0, 0),
        )

    def _forward_prop(self, *, input_activations: Activations) -> Activations:
        batch_size, in_channels = input_activations.shape[:2]
        self._cache["input_activations"] = input_activations

        output = np.zeros(
            (
                batch_size,
                self._n_kernels,
                self._out_height,
                self._out_width,
            )
        )

        # TODO: This is a slow implementation because it uses explicit looping.
        # Explore vectorized implementation (k.w. search for 'unroll' method).
        for batch_idx in range(batch_size):
            for kernel_idx in range(self._n_kernels):
                for channel_idx in range(in_channels):
                    output[batch_idx, kernel_idx] += (
                        signal.correlate2d(
                            input_activations[batch_idx, channel_idx],
                            self._parameters.weights[kernel_idx, channel_idx],
                            mode="valid",
                        )[:: self._stride_y, :: self._stride_x]
                        + self._parameters.biases[kernel_idx]
                    )

        return output

    def _backward_prop(self, *, dZ: D[Activations]) -> D[Activations]:
        batch_size, n_kernels, _, _ = dZ.shape
        input_activations = self._cache["input_activations"]
        if input_activations is None:
            raise ValueError("input_activations not set during forward_prop.")

        dX = np.zeros(self.input_dimensions[1:])
        dK = np.zeros(self._parameters.weights.shape)
        db = np.zeros(self._parameters.biases.shape)
        # TODO: This is a slow implementation because it uses explicit looping.
        # Explore vectorized implementation (k.w. search for 'unroll' method).
        for batch_idx in range(batch_size):
            for kernel_idx in range(n_kernels):
                for channel_idx in range(self._in_channels):
                    dK[kernel_idx, channel_idx] += signal.correlate2d(
                        input_activations[batch_idx, channel_idx],
                        dZ[batch_idx, kernel_idx],
                        mode="valid",
                    )
                    db[kernel_idx] += np.sum(dZ[batch_idx, kernel_idx])
                    dX += signal.convolve2d(
                        self._parameters.weights[kernel_idx, channel_idx],
                        dZ[batch_idx, kernel_idx],
                        mode="full",
                    )
        self._cache["dP"] = self.Parameters(weights=dK, biases=db)
        return dX

    def gradient_operation(self, *, f: Callable[[GradLayer], None]) -> None:
        f(self)

    def empty_gradient(self) -> D[Parameters]:
        return d(
            self.Parameters(
                weights=np.zeros_like(self._parameters.weights),
                biases=np.zeros_like(self._parameters.biases),
            )
        )

    @property
    def cache(self) -> Cache:
        return self._cache

    @property
    def parameters(self) -> Parameters:
        return self._parameters

    def update_parameters(self) -> None:
        if self._cache["dP"] is None:
            raise ValueError("Gradient not set during backward pass.")
        if not self._freeze_parameters:
            self._parameters = self._parameters + self._cache["dP"]
        self._cache["dP"] = None

    def serialize(self) -> Convolution2D.Serialized:
        return Convolution2D.Serialized(
            input_dimensions=self.input_dimensions,
            n_kernels=self._n_kernels,
            kernel_size=self._kernel_width,
            stride=self._stride_x,
            output_dimensions=self.output_dimensions,
            parameters=self._parameters,
        )
