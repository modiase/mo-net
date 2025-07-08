from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import IO, TypedDict, assert_never, cast

import jax
import jax.numpy as jnp
from jax import jit, lax, random

from mo_net.constants import EPSILON
from mo_net.model.layer.base import BadLayerId, ParametrisedHidden
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
        key: jax.Array,
    ) -> Parameters:
        key1, key2 = jax.random.split(key)
        limit = jnp.sqrt(1 / (in_channels * in_height * in_width))
        return cls(
            weights=jax.random.uniform(
                key1,
                shape=(n_kernels, in_channels, in_height, in_width),
                minval=-limit,
                maxval=limit,
            ),
            biases=random.uniform(
                key2,
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
        key: jax.Array,
    ) -> Parameters:
        fan_in = in_channels * in_height * in_width
        limit = jnp.sqrt(6.0 / fan_in)
        return cls(
            weights=jax.random.uniform(
                key,
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
        *,
        key: jax.Array,
    ) -> Parameters:
        key1 = jax.random.split(key)[0]
        return cls(
            weights=jax.random.normal(
                key1,
                (n_kernels, in_channels, in_height, in_width),
            )
            * jnp.sqrt(2 / (in_channels * in_height * in_width)),
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

    @staticmethod
    @jit
    def _clip_gradient_impl(grad: jnp.ndarray, max_norm: float) -> jnp.ndarray:
        """JIT-compiled gradient clipping implementation."""
        return grad * jnp.minimum(
            1.0, max_norm * jnp.sqrt(grad.size) / (jnp.linalg.norm(grad) + EPSILON)
        )

    @staticmethod
    def _no_clip_gradient(grad: jnp.ndarray, max_norm: float) -> jnp.ndarray:
        """No-op gradient clipping function."""
        return grad

    @dataclass(frozen=True, kw_only=True)
    class Serialized:
        layer_id: str
        input_dimensions: tuple[int, int, int]
        n_kernels: int
        kernel_size: tuple[int, int]
        stride: tuple[int, int]
        output_dimensions: tuple[int, int, int]
        parameters: Parameters

        def deserialize(
            self,
            *,
            training: bool = False,
            freeze_parameters: bool = False,
        ) -> Convolution2D:
            del training  # unused

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
                freeze_parameters=freeze_parameters,
            )

    def __init__(
        self,
        *,
        input_dimensions: Dimensions,
        n_kernels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        # padding: int | tuple[int, int] = 0, # TODO: Implement padding.
        kernel_init_fn: KernelInitFn,
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
        self._cache = Cache(input_activations=None, output_activations=None, dP=None)

    def _forward_prop(self, *, input_activations: Activations) -> Activations:
        self._cache["input_activations"] = input_activations

        output = lax.conv_general_dilated(
            input_activations,
            self._parameters.weights,
            window_strides=(self._stride_y, self._stride_x),
            padding="VALID",
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
            feature_group_count=1,
        )

        output += self._parameters.biases[jnp.newaxis, :, jnp.newaxis, jnp.newaxis]

        return Activations(output)

    def _backward_prop(self, *, dZ: D[Activations]) -> D[Activations]:
        batch_size, _, dZ_height, dZ_width = cast(jnp.ndarray, dZ).shape
        input_activations = self._cache["input_activations"]
        if input_activations is None:
            raise ValueError("input_activations not set during forward_prop.")

        batch_size, in_channels, in_height, in_width = input_activations.shape

        patches = lax.conv_general_dilated_patches(
            input_activations,
            filter_shape=(self._kernel_height, self._kernel_width),
            window_strides=(self._stride_y, self._stride_x),
            padding="VALID",
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )

        patches_reshaped = patches.reshape(
            batch_size,
            in_channels,
            self._kernel_height,
            self._kernel_width,
            dZ_height,
            dZ_width,
        )

        patches_transposed = jnp.transpose(patches_reshaped, (0, 4, 5, 1, 2, 3))

        dZ_transposed = jnp.transpose(cast(jnp.ndarray, dZ), (0, 2, 3, 1))

        dK = jnp.einsum("byxihw,byxo->oihw", patches_transposed, dZ_transposed)

        db = jnp.sum(cast(jnp.ndarray, dZ), axis=(0, 2, 3))

        if self._clip_gradients:
            dK = self._clip_gradient_impl(dK, self._weight_max_norm)
            db = self._clip_gradient_impl(db, self._bias_max_norm)

        pad_h = self._kernel_height - 1
        pad_w = self._kernel_width - 1
        dZ_padded = jnp.pad(
            cast(jnp.ndarray, dZ),
            ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
            mode="constant",
        )

        weights_flipped = jnp.flip(self._parameters.weights, axis=(2, 3))

        dX = jnp.zeros((batch_size, in_channels, in_height, in_width))

        for k in range(self._n_kernels):
            dZ_k = dZ_padded[:, k : k + 1, :, :]
            weights_k = weights_flipped[k : k + 1, :, :, :]
            weights_k_t = jnp.transpose(weights_k, (1, 0, 2, 3))

            dX_k = lax.conv_general_dilated(
                dZ_k,
                weights_k_t,
                window_strides=(1, 1),
                padding="VALID",
                dimension_numbers=("NCHW", "OIHW", "NCHW"),
            )
            dX += dX_k

        self._cache["dP"] = d(self.Parameters(weights=dK, biases=db))
        return cast(D[Activations], dX)

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
        if (dP := self._cache["dP"]) is None:
            raise ValueError("Gradient not set during backward pass.")
        if not self._freeze_parameters:
            self._parameters = self._parameters + dP
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
