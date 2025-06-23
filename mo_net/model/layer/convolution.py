from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import IO, Final, TypedDict, assert_never, cast

import jax.numpy as jnp
from jax import lax, random

from mo_net.model.layer.base import BadLayerId, ParametrisedHidden
from mo_net.protos import (
    Activations,
    D,
    Dimensions,
    GradLayer,
    SupportsGradientOperations,
    d,
)

EPSILON: Final[float] = 1e-8


class Cache(TypedDict):
    input_activations: Activations | None
    output_activations: Activations | None
    dP: D[Parameters] | None


type CacheType = Cache


@dataclass(frozen=True, kw_only=True)
class Parameters(SupportsGradientOperations):
    weights: jnp.ndarray
    biases: jnp.ndarray

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
        limit = jnp.sqrt(1 / (in_channels * in_height * in_width))
        return cls(
            weights=random.uniform(
                random.PRNGKey(0),
                shape=(n_kernels, in_channels, in_height, in_width),
                minval=-limit,
                maxval=limit,
            ),
            biases=random.uniform(
                random.PRNGKey(1),
                shape=(n_kernels,),
                minval=-limit,
                maxval=limit,
            ),
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
            weights=jnp.ones((n_kernels, in_channels, in_height, in_width)),
            biases=jnp.zeros(n_kernels),
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
        limit = jnp.sqrt(6.0 / fan_in)
        return cls(
            weights=random.uniform(
                random.PRNGKey(0),
                shape=(n_kernels, in_channels, in_height, in_width),
                minval=-limit,
                maxval=limit,
            ),
            biases=jnp.zeros(n_kernels),
        )

    @classmethod
    def he(
        cls,
        n_kernels: int,
        in_channels: int,
        in_height: int,
        in_width: int,
    ) -> Parameters:
        std = jnp.sqrt(2 / (in_channels * in_height * in_width))
        return cls(
            weights=random.normal(
                random.PRNGKey(0),
                (n_kernels, in_channels, in_height, in_width),
            )
            * std,
            biases=jnp.zeros(n_kernels),
        )

    def from_bytes(self, data: IO[bytes]) -> Parameters:
        return Parameters(
            weights=jnp.frombuffer(
                data.read(self.weights.nbytes), dtype=self.weights.dtype
            ).reshape(self.weights.shape),
            biases=jnp.frombuffer(
                data.read(self.biases.nbytes), dtype=self.biases.dtype
            ).reshape(self.biases.shape),
        )


type ParametersType = Parameters
type KernelInitFn = Callable[[int, int, int, int], ParametersType]


class Convolution2D(ParametrisedHidden[ParametersType, CacheType]):
    Cache = Cache
    Parameters = Parameters

    @dataclass(frozen=True, kw_only=True)
    class Serialized:
        layer_id: str
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
                layer_id=self.layer_id,
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
        layer_id: str | None = None,
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
            layer_id=layer_id,
            input_dimensions=input_dimensions,
            output_dimensions=output_dimensions,
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

        # Use JAX's conv_general_dilated for convolution
        # Our format is NCHW (batch, channels, height, width)
        # Kernel format is OIHW (out_channels, in_channels, height, width)
        output = lax.conv_general_dilated(
            input_activations,
            self._parameters.weights,
            window_strides=(self._stride_y, self._stride_x),
            padding="VALID",
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
            feature_group_count=1,
        )

        # Add bias
        output += self._parameters.biases[jnp.newaxis, :, jnp.newaxis, jnp.newaxis]

        return Activations(output)

    def _backward_prop(self, *, dZ: D[Activations]) -> D[Activations]:
        batch_size, n_kernels, dZ_height, dZ_width = cast(jnp.ndarray, dZ).shape
        input_activations = self._cache["input_activations"]
        if input_activations is None:
            raise ValueError("input_activations not set during forward_prop.")

        # Compute weight gradients using transposed convolution
        # We treat the input as the "kernel" and dZ as the "input" to get weight gradients

        # First, we need to extract patches from the input
        # Create indices for the patches
        batch_size, in_channels, in_height, in_width = input_activations.shape

        # Use lax.conv_general_dilated_patches to extract input patches
        patches = lax.conv_general_dilated_patches(
            input_activations,
            filter_shape=(self._kernel_height, self._kernel_width),
            window_strides=(self._stride_y, self._stride_x),
            padding="VALID",
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )

        # patches shape: (batch_size, in_channels * kernel_h * kernel_w, out_h, out_w)
        # Reshape to (batch_size, in_channels, kernel_h, kernel_w, out_h, out_w)
        patches_reshaped = patches.reshape(
            batch_size,
            in_channels,
            self._kernel_height,
            self._kernel_width,
            dZ_height,
            dZ_width,
        )

        # Transpose to (batch_size, out_h, out_w, in_channels, kernel_h, kernel_w)
        patches_transposed = jnp.transpose(patches_reshaped, (0, 4, 5, 1, 2, 3))

        # dZ shape: (batch_size, n_kernels, out_h, out_w)
        # Transpose to (batch_size, out_h, out_w, n_kernels)
        dZ_transposed = jnp.transpose(cast(jnp.ndarray, dZ), (0, 2, 3, 1))

        # Compute weight gradients using einsum
        # We want: dK[o, i, h, w] = sum over batch and positions of: patches[b, y, x, i, h, w] * dZ[b, y, x, o]
        dK = jnp.einsum("byxihw,byxo->oihw", patches_transposed, dZ_transposed)

        # Compute bias gradients
        db = jnp.sum(cast(jnp.ndarray, dZ), axis=(0, 2, 3))

        if self._clip_gradients:
            dK *= jnp.minimum(
                1.0,
                self._weight_max_norm
                * jnp.sqrt(dK.size)
                / (jnp.linalg.norm(dK) + EPSILON),
            )
            db *= jnp.minimum(
                1.0,
                self._bias_max_norm
                * jnp.sqrt(db.size)
                / (jnp.linalg.norm(db) + EPSILON),
            )

        # Compute gradients with respect to input
        # For multiple kernels, we need to sum contributions from all output channels
        # This is essentially a transposed convolution

        # Pad dZ for full convolution
        pad_h = self._kernel_height - 1
        pad_w = self._kernel_width - 1
        dZ_padded = jnp.pad(
            cast(jnp.ndarray, dZ),
            ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
            mode="constant",
        )

        # Flip weights for convolution (rotate 180 degrees)
        weights_flipped = jnp.flip(self._parameters.weights, axis=(2, 3))

        # For backward pass through convolution, we need to sum over output channels
        # dZ_padded shape: (batch, n_kernels, padded_height, padded_width)
        # weights shape: (n_kernels, in_channels, kernel_height, kernel_width)
        # We want dX shape: (batch, in_channels, in_height, in_width)

        # We can compute this as a sum of convolutions for each output channel
        dX = jnp.zeros((batch_size, in_channels, in_height, in_width))

        for k in range(self._n_kernels):
            # Extract gradients for this kernel
            dZ_k = dZ_padded[
                :, k : k + 1, :, :
            ]  # Shape: (batch, 1, padded_h, padded_w)
            # Extract weights for this kernel
            weights_k = weights_flipped[
                k : k + 1, :, :, :
            ]  # Shape: (1, in_channels, kh, kw)
            # Transpose to (in_channels, 1, kh, kw) for convolution
            weights_k_t = jnp.transpose(weights_k, (1, 0, 2, 3))

            # Convolve to get contribution to dX
            dX_k = lax.conv_general_dilated(
                dZ_k,
                weights_k_t,
                window_strides=(1, 1),
                padding="VALID",
                dimension_numbers=("NCHW", "OIHW", "NCHW"),
            )
            dX += dX_k

        self._cache["dP"] = d(self.Parameters(weights=dK, biases=db))
        return d(Activations(dX))

    def gradient_operation(self, f: Callable[[GradLayer], None]) -> None:
        f(self)

    def empty_gradient(self) -> D[ParametersType]:
        return d(
            self.Parameters(
                weights=jnp.zeros_like(self._parameters.weights),
                biases=jnp.zeros_like(self._parameters.biases),
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
            layer_id=self._layer_id,
            input_dimensions=(channels, height, width),
            n_kernels=self._n_kernels,
            kernel_size=(self._kernel_height, self._kernel_width),
            stride=(self._stride_x, self._stride_y),
            output_dimensions=(self._n_kernels, self._out_height, self._out_width),
            parameters=self._parameters,
        )

    def write_serialized_parameters(self, buffer: IO[bytes]) -> None:
        self._write_header(buffer)
        if self._cache["dP"] is None:
            raise RuntimeError("Cache is not populated during serialization.")
        buffer.write(memoryview(self._cache["dP"].weights))  # type: ignore[attr-defined]
        buffer.write(memoryview(self._cache["dP"].biases))  # type: ignore[attr-defined]

    def read_serialized_parameters(self, data: IO[bytes]) -> None:
        if (layer_id := self.get_layer_id(data)) != self._layer_id:
            raise BadLayerId(f"Layer ID mismatch: {layer_id} != {self._layer_id}")
        update = self._parameters.from_bytes(data)
        if self._cache["dP"] is None:
            self._cache["dP"] = d(update)
        else:
            self._cache["dP"] += d(update)

    @property
    def parameter_nbytes(self) -> int:
        return self._parameters.weights.nbytes + self._parameters.biases.nbytes

    @property
    def parameter_count(self) -> int:
        return self._parameters.weights.size + self._parameters.biases.size
