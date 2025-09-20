from __future__ import annotations

from dataclasses import dataclass
from typing import (
    IO,
    Callable,
    Self,
)

import jax
import jax.numpy as jnp
from jax import jit
from more_itertools import one

from mo_net.constants import EPSILON
from mo_net.functions import ActivationFn, identity
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
    SupportsGradientOperations,
    d,
)


@dataclass(kw_only=True)
class Parameters(SupportsGradientOperations):
    weights: jnp.ndarray
    biases: jnp.ndarray

    def __getitem__(
        self, index: int | tuple[int, ...]
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self.weights[index], self.biases[index]

    def __add__(self, other: Self | float | int) -> Self:
        match other:
            case Parameters():
                return self.__class__(
                    weights=self.weights + other.weights,
                    biases=self.biases + other.biases,
                )
            case float() | int():
                return self.__class__(
                    weights=self.weights + other, biases=self.biases + other
                )
            case _:
                return NotImplemented

    def __radd__(self, other: Self | float | int) -> Self:
        return self.__add__(other)

    def __neg__(self) -> Self:
        return self.__class__(weights=-self.weights, biases=-self.biases)

    def __sub__(self, other: Self | float | int) -> Self:
        return self.__add__(-other)

    def __rsub__(self, other: Self | float | int) -> Self:
        return self.__sub__(other)

    def __mul__(self, other: float | int | Self) -> Self:
        match other:
            case float() | int():
                return self.__class__(
                    weights=other * self.weights, biases=other * self.biases
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
            case Parameters():
                return self.__class__(
                    weights=self.weights / (other.weights + EPSILON),
                    biases=self.biases / (other.biases + EPSILON),
                )
            case float() | int():
                return self.__mul__(1 / other)
            case _:
                return NotImplemented

    def __pow__(self, scalar: float | int) -> Self:
        return self.__class__(weights=self.weights**scalar, biases=self.biases**scalar)

    @classmethod
    def random(
        cls,
        dim_in: Dimensions,
        dim_out: Dimensions,
        key: jax.Array,
    ) -> Self:
        _dim_in = one(dim_in)
        _dim_out = one(dim_out)
        return cls(
            weights=jax.random.normal(key, (_dim_in, _dim_out)),
            biases=jnp.zeros(_dim_out),
        )

    @classmethod
    def xavier(cls, dim_in: Dimensions, dim_out: Dimensions, key: jax.Array) -> Self:
        _dim_in = one(dim_in)
        _dim_out = one(dim_out)
        return cls(
            weights=jax.random.normal(key, (_dim_in, _dim_out)) * jnp.sqrt(1 / _dim_in),
            biases=jnp.zeros(_dim_out),
        )

    @classmethod
    def he(cls, dim_in: Dimensions, dim_out: Dimensions, *, key: jax.Array) -> Self:
        _dim_in = one(dim_in)
        _dim_out = one(dim_out)
        return cls(
            weights=jax.random.normal(key, (_dim_in, _dim_out)) * jnp.sqrt(2 / _dim_in),
            biases=jnp.zeros(_dim_out),
        )

    @classmethod
    def appropriate(
        cls,
        dim_in: Dimensions,
        dim_out: Dimensions,
        *,
        activation_fn: ActivationFn,
        key: jax.Array,
    ) -> Self:
        key1, key2 = jax.random.split(key)
        # Check if it's a ReLU or LeakyReLU activation
        if hasattr(activation_fn, "__class__") and activation_fn.__class__.__name__ in (
            "ReLU",
            "LeakyReLU",
        ):
            return cls.he(dim_in, dim_out, key=key1)
        # Check if it's a Tanh or Identity activation
        elif hasattr(
            activation_fn, "__class__"
        ) and activation_fn.__class__.__name__ in ("Tanh", "Identity"):
            return cls.xavier(dim_in, dim_out, key=key2)
        # Fallback for old function-based activations
        elif activation_fn in (jax.nn.relu, jax.nn.leaky_relu):
            return cls.he(dim_in, dim_out, key=key1)
        elif activation_fn in (jax.nn.tanh, identity):
            return cls.xavier(dim_in, dim_out, key=key2)
        else:
            raise ValueError(
                f"Cannot choose appropriate initialisation for {activation_fn}"
            )

    @classmethod
    def eye(cls, dim: Dimensions) -> Self:
        _dim = one(dim)
        return cls(weights=jnp.eye(_dim), biases=jnp.zeros(_dim))

    @classmethod
    def of(cls, W: jnp.ndarray, B: jnp.ndarray) -> Self:
        return cls(weights=jnp.atleast_2d(W), biases=jnp.atleast_1d(B))

    def from_bytes(self, data: IO[bytes]) -> Self:
        W = jnp.frombuffer(
            data.read(self.weights.nbytes), dtype=self.weights.dtype
        ).reshape(self.weights.shape)
        B = jnp.frombuffer(
            data.read(self.biases.nbytes), dtype=self.biases.dtype
        ).reshape(self.biases.shape)
        return self.__class__(weights=W, biases=B)


type ParametersType = Parameters


class Cache(GradCache[ParametersType]):
    input_activations: Activations | None
    output_activations: Activations | None


type CacheType = Cache


class Linear(ParametrisedHidden[ParametersType, CacheType]):
    Parameters = Parameters
    Cache = Cache
    _parameters: ParametersType
    _cache: CacheType

    @staticmethod
    @jit
    def _clip_gradient_impl(grad: jnp.ndarray, max_norm: float) -> jnp.ndarray:
        """JIT-compiled gradient clipping implementation."""
        norm = jnp.linalg.norm(grad)
        scale = jnp.minimum(1.0, max_norm * jnp.sqrt(grad.size) / (norm + EPSILON))
        return grad * scale

    @staticmethod
    def _no_clip_gradient(grad: jnp.ndarray, max_norm: float) -> jnp.ndarray:
        """No-op gradient clipping function."""
        return grad

    @classmethod
    def of_bias(cls, dim: Dimensions, bias: float | jnp.ndarray) -> Self:
        return cls(
            input_dimensions=dim,
            output_dimensions=dim,
            parameters_init_fn=lambda dim_in, dim_out: cls.Parameters.of(
                W=jnp.zeros((one(dim_in), one(dim_out))),
                B=(jnp.ones(one(dim_out)) * bias if isinstance(bias, float) else bias),
            ),
        )

    @classmethod
    def of_eye(cls, dim: Dimensions) -> Self:
        return cls(
            input_dimensions=dim,
            output_dimensions=dim,
            parameters_init_fn=lambda _, __: cls.Parameters.eye(dim),
        )

    @dataclass(frozen=True, kw_only=True)
    class Serialized:
        layer_id: str
        input_dimensions: tuple[int, ...]
        output_dimensions: tuple[int, ...]
        parameters: Parameters

        def deserialize(
            self,
            *,
            training: bool = False,
            freeze_parameters: bool = False,
        ) -> Linear:
            del training  # unused
            return Linear(
                layer_id=self.layer_id,
                input_dimensions=self.input_dimensions,
                output_dimensions=self.output_dimensions,
                parameters_init_fn=lambda _, __: self.parameters,
                freeze_parameters=freeze_parameters,
            )

    def __init__(
        self,
        *,
        bias_max_norm: float = 1.0,
        clip_gradients: bool = True,
        freeze_parameters: bool = False,
        input_dimensions: Dimensions,
        layer_id: str | None = None,
        output_dimensions: Dimensions | None = None,
        parameters_init_fn: Callable[[Dimensions, Dimensions], ParametersType],
        store_output_activations: bool = False,  # Only used for tracing
        weight_max_norm: float = 1.0,
    ):
        if output_dimensions is None:
            output_dimensions = input_dimensions
        super().__init__(
            layer_id=layer_id,
            input_dimensions=input_dimensions,
            output_dimensions=output_dimensions,
        )
        self._freeze_parameters = freeze_parameters
        self._weight_max_norm = weight_max_norm
        self._bias_max_norm = bias_max_norm

        self._clip_gradient_fn = (
            self._clip_gradient_impl if clip_gradients else self._no_clip_gradient
        )

        self._parameters_init_fn = parameters_init_fn
        self._parameters = parameters_init_fn(input_dimensions, output_dimensions)

        if self._parameters.weights.shape != (
            one(input_dimensions),
            one(output_dimensions),
        ):
            raise ValueError(
                f"Weight matrix shape ({self._parameters.weights.shape}) "
                f"does not match input dimensions ({input_dimensions}) "
                f"and output dimensions ({output_dimensions})."
            )
        if self._parameters.biases.shape != (one(output_dimensions),):
            raise ValueError(
                f"Bias vector shape ({self._parameters.biases.shape}) "
                f"does not match output dimensions ({output_dimensions})."
            )

        self._store_output_activations = store_output_activations
        self._cache: CacheType = {
            "input_activations": None,
            "output_activations": None,
            "dP": None,
        }

    def _forward_prop(self, *, input_activations: Activations) -> Activations:
        self._cache["input_activations"] = input_activations

        @jit
        def linear_forward(x, w, b):
            return x @ w + b

        output_activations = Activations(
            linear_forward(
                input_activations, self._parameters.weights, self._parameters.biases
            )
        )
        if self._store_output_activations:
            self._cache["output_activations"] = output_activations
        return output_activations

    def _backward_prop(
        self,
        *,
        dZ: D[Activations],
    ) -> D[Activations]:
        if (input_activations := self._cache["input_activations"]) is None:
            raise ValueError("Input activations not set during forward pass.")

        @jit
        def compute_gradients(x, dz, w):
            dW = x.T @ dz
            dB = jnp.sum(dz, axis=0)
            dX = dz @ w.T
            return dW, dB, dX

        dW, dB, dX = compute_gradients(input_activations, dZ, self._parameters.weights)

        dW = self._clip_gradient_fn(dW, self._weight_max_norm)
        dB = self._clip_gradient_fn(dB, self._bias_max_norm)

        self._cache["dP"] = d(self.Parameters(weights=dW, biases=dB))
        return dX

    def empty_gradient(self) -> D[ParametersType]:
        return d(
            self.Parameters(
                weights=jnp.zeros_like(self._parameters.weights),
                biases=jnp.zeros_like(self._parameters.biases),
            )
        )

    def reinitialise(self) -> None:
        self._parameters = self._parameters_init_fn(
            self._input_dimensions, self._output_dimensions
        )

    def serialize(self) -> Linear.Serialized:
        return self.Serialized(
            layer_id=self._layer_id,
            input_dimensions=tuple(self._input_dimensions),
            output_dimensions=tuple(self._output_dimensions),
            parameters=self._parameters,
        )

    def update_parameters(self) -> None:
        if (dP := self._cache["dP"]) is None:
            raise ValueError("Gradient not set during backward pass.")
        if not self._freeze_parameters:
            self._parameters = self._parameters + dP
        self._cache["dP"] = None

    def gradient_operation(self, f: Callable[[GradLayer], None]) -> None:
        f(self)

    @property
    def cache(self) -> CacheType:
        return self._cache

    @property
    def parameters(self) -> ParametersType:
        return self._parameters

    @property
    def parameter_count(self) -> int:
        return self._parameters.weights.size + self._parameters.biases.size

    def write_serialized_parameters(self, buffer: IO[bytes]) -> None:
        self._write_header(buffer)
        if self._cache["dP"] is None:
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

    @property
    def parameter_nbytes(self) -> int:
        return self._parameters.weights.nbytes + self._parameters.biases.nbytes
