from __future__ import annotations

from dataclasses import dataclass
from typing import IO, Callable, Self, cast

import jax
import jax.numpy as jnp
from jax import jit
from more_itertools import one

from mo_net.constants import EPSILON
from mo_net.functions import ActivationFn, Tanh
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


@dataclass(kw_only=True)
class Parameters(SupportsGradientOperations):
    """RNN parameters: Wxh (input→hidden), Whh (hidden→hidden), bh (bias)."""

    Wxh: jnp.ndarray
    Whh: jnp.ndarray
    bh: jnp.ndarray

    def __add__(self, other: Self | float | int) -> Self:
        match other:
            case Parameters():
                return self.__class__(
                    Wxh=self.Wxh + other.Wxh,
                    Whh=self.Whh + other.Whh,
                    bh=self.bh + other.bh,
                )
            case float() | int():
                return self.__class__(
                    Wxh=self.Wxh + other, Whh=self.Whh + other, bh=self.bh + other
                )
            case _:
                return NotImplemented

    def __radd__(self, other: Self | float | int) -> Self:
        return self.__add__(other)

    def __neg__(self) -> Self:
        return self.__class__(Wxh=-self.Wxh, Whh=-self.Whh, bh=-self.bh)

    def __sub__(self, other: Self | float | int) -> Self:
        return self.__add__(-other)

    def __rsub__(self, other: Self | float | int) -> Self:
        return self.__sub__(other)

    def __mul__(self, other: float | int | Self) -> Self:
        match other:
            case float() | int():
                return self.__class__(
                    Wxh=other * self.Wxh, Whh=other * self.Whh, bh=other * self.bh
                )
            case self.__class__():  # type: ignore[reportGeneralTypeIssues]
                return self.__class__(
                    Wxh=self.Wxh * other.Wxh,
                    Whh=self.Whh * other.Whh,
                    bh=self.bh * other.bh,
                )
            case _:
                return NotImplemented

    def __rmul__(self, other: float | Self) -> Self:
        return self.__mul__(other)

    def __truediv__(self, other: Self | float | int) -> Self:
        match other:
            case Parameters():
                return self.__class__(
                    Wxh=self.Wxh / (other.Wxh + EPSILON),
                    Whh=self.Whh / (other.Whh + EPSILON),
                    bh=self.bh / (other.bh + EPSILON),
                )
            case float() | int():
                return self.__mul__(1 / other)
            case _:
                return NotImplemented

    def __pow__(self, scalar: float | int) -> Self:
        return self.__class__(
            Wxh=self.Wxh**scalar, Whh=self.Whh**scalar, bh=self.bh**scalar
        )

    @classmethod
    def random(cls, input_dim: int, hidden_dim: int, key: jax.Array) -> Self:
        key1, key2 = jax.random.split(key)
        return cls(
            Wxh=jax.random.normal(key1, (input_dim, hidden_dim)),
            Whh=jax.random.normal(key2, (hidden_dim, hidden_dim)),
            bh=jnp.zeros(hidden_dim),
        )

    @classmethod
    def xavier(cls, input_dim: int, hidden_dim: int, key: jax.Array) -> Self:
        key1, key2 = jax.random.split(key)
        return cls(
            Wxh=jax.random.normal(key1, (input_dim, hidden_dim))
            * jnp.sqrt(1 / input_dim),
            Whh=jax.random.normal(key2, (hidden_dim, hidden_dim))
            * jnp.sqrt(1 / hidden_dim),
            bh=jnp.zeros(hidden_dim),
        )

    @classmethod
    def he(cls, input_dim: int, hidden_dim: int, key: jax.Array) -> Self:
        key1, key2 = jax.random.split(key)
        return cls(
            Wxh=jax.random.normal(key1, (input_dim, hidden_dim))
            * jnp.sqrt(2 / input_dim),
            Whh=jax.random.normal(key2, (hidden_dim, hidden_dim))
            * jnp.sqrt(2 / hidden_dim),
            bh=jnp.zeros(hidden_dim),
        )

    @classmethod
    def of(cls, Wxh: jnp.ndarray, Whh: jnp.ndarray, bh: jnp.ndarray) -> Self:
        return cls(
            Wxh=jnp.atleast_2d(Wxh),
            Whh=jnp.atleast_2d(Whh),
            bh=jnp.atleast_1d(bh),
        )

    def from_bytes(self, data: IO[bytes]) -> Self:
        Wxh = jnp.frombuffer(data.read(self.Wxh.nbytes), dtype=self.Wxh.dtype).reshape(
            self.Wxh.shape
        )
        Whh = jnp.frombuffer(data.read(self.Whh.nbytes), dtype=self.Whh.dtype).reshape(
            self.Whh.shape
        )
        bh = jnp.frombuffer(data.read(self.bh.nbytes), dtype=self.bh.dtype).reshape(
            self.bh.shape
        )
        return self.__class__(Wxh=Wxh, Whh=Whh, bh=bh)


type ParametersType = Parameters


class Cache(GradCache[ParametersType]):
    input_sequences: Activations | None
    hidden_states: jnp.ndarray | None
    output_activations: Activations | None


type CacheType = Cache


class Recurrent(ParametrisedHidden[ParametersType, CacheType]):
    """Vanilla RNN layer with BPTT (Backpropagation Through Time).

    Processes sequences of inputs and maintains hidden state across time steps.
    Supports both many-to-one (return final hidden) and many-to-many (return all hiddens).
    """

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

    @dataclass(frozen=True, kw_only=True)
    class Serialized:
        layer_id: str
        input_dimensions: tuple[int, ...]
        output_dimensions: tuple[int, ...]
        hidden_dim: int
        return_sequences: bool
        parameters: Parameters

        def deserialize(
            self,
            *,
            training: bool = False,
            freeze_parameters: bool = False,
        ) -> Recurrent:
            del training
            return Recurrent(
                layer_id=self.layer_id,
                input_dimensions=self.input_dimensions,
                output_dimensions=self.output_dimensions,
                hidden_dim=self.hidden_dim,
                return_sequences=self.return_sequences,
                parameters=self.parameters,
                freeze_parameters=freeze_parameters,
                key=jax.random.PRNGKey(0),
                exists_ok=True,
            )

    def __init__(
        self,
        *,
        activation_fn: ActivationFn = Tanh(),
        clip_gradients: bool = True,
        exists_ok: bool = False,
        freeze_parameters: bool = False,
        gradient_max_norm: float = 1.0,
        hidden_dim: int,
        input_dimensions: Dimensions,
        key: jax.Array,
        layer_id: str | None = None,
        output_dimensions: Dimensions | None = None,
        parameters: ParametersType | None = None,
        parameters_init_fn: Callable[
            [int, int, jax.Array], ParametersType
        ] = Parameters.xavier,
        return_sequences: bool = True,
        store_output_activations: bool = False,
    ):
        input_dim = (
            one(input_dimensions)
            if len(input_dimensions) == 1
            else input_dimensions[-1]
        )

        if output_dimensions is None:
            output_dimensions = (
                (input_dimensions[0] if len(input_dimensions) > 1 else 1, hidden_dim)
                if return_sequences
                else (hidden_dim,)
            )

        super().__init__(
            layer_id=layer_id,
            input_dimensions=input_dimensions,
            output_dimensions=output_dimensions,
            exists_ok=exists_ok,
        )

        self._activation_fn = activation_fn
        self._freeze_parameters = freeze_parameters
        self._gradient_max_norm = gradient_max_norm
        self._hidden_dim = hidden_dim
        self._key = key
        self._parameters_init_fn = parameters_init_fn
        self._return_sequences = return_sequences
        self._store_output_activations = store_output_activations

        self._clip_gradient_fn = (
            self._clip_gradient_impl if clip_gradients else self._no_clip_gradient
        )

        # Initialize or validate parameters
        if parameters is not None:
            if parameters.Wxh.shape != (input_dim, hidden_dim):
                raise ValueError(
                    f"Wxh shape {parameters.Wxh.shape} does not match "
                    f"expected ({input_dim}, {hidden_dim})"
                )
            if parameters.Whh.shape != (hidden_dim, hidden_dim):
                raise ValueError(
                    f"Whh shape {parameters.Whh.shape} does not match "
                    f"expected ({hidden_dim}, {hidden_dim})"
                )
            if parameters.bh.shape != (hidden_dim,):
                raise ValueError(
                    f"bh shape {parameters.bh.shape} does not match "
                    f"expected ({hidden_dim},)"
                )
            self._parameters = parameters
        else:
            self._key, subkey = jax.random.split(self._key)
            self._parameters = self._parameters_init_fn(input_dim, hidden_dim, subkey)

        self._cache: CacheType = {
            "input_sequences": None,
            "hidden_states": None,
            "output_activations": None,
            "dP": None,
        }

    def _forward_prop(self, *, input_activations: Activations) -> Activations:
        if input_activations.ndim == 2:
            input_activations = Activations(jnp.expand_dims(input_activations, axis=1))

        self._cache["input_sequences"] = input_activations
        batch_size, seq_len, _ = input_activations.shape

        @jit
        def rnn_step(h_prev, x_t):
            return (
                h_t := self._activation_fn(
                    x_t @ self._parameters.Wxh
                    + h_prev @ self._parameters.Whh
                    + self._parameters.bh
                ),
                h_t,
            )

        _, hidden_states = jax.lax.scan(
            rnn_step,
            jnp.zeros((batch_size, self._hidden_dim)),
            jnp.transpose(input_activations, (1, 0, 2)),
        )

        hidden_states = jnp.transpose(hidden_states, (1, 0, 2))
        self._cache["hidden_states"] = hidden_states

        output_activations = Activations(
            hidden_states if self._return_sequences else hidden_states[:, -1, :]
        )

        if self._store_output_activations:
            self._cache["output_activations"] = output_activations

        return output_activations

    def _backward_prop(self, *, dZ: D[Activations]) -> D[Activations]:
        if (input_sequences := self._cache["input_sequences"]) is None:
            raise ValueError("Input sequences not cached during forward pass")
        if (hidden_states := self._cache["hidden_states"]) is None:
            raise ValueError("Hidden states not cached during forward pass")

        batch_size, seq_len, input_dim = input_sequences.shape
        dZ_array = cast(jnp.ndarray, dZ)

        if not self._return_sequences:
            dZ_array = (
                jnp.zeros((batch_size, seq_len, self._hidden_dim))
                .at[:, -1, :]
                .set(dZ_array)
            )

        dWxh = jnp.zeros_like(self._parameters.Wxh)
        dWhh = jnp.zeros_like(self._parameters.Whh)
        dbh = jnp.zeros_like(self._parameters.bh)
        dX = jnp.zeros_like(input_sequences)
        dh_next = jnp.zeros((batch_size, self._hidden_dim))

        for t in reversed(range(seq_len)):
            dh_raw = self._activation_fn.deriv(hidden_states[:, t, :]) * (
                dZ_array[:, t, :] + dh_next
            )
            h_prev = (
                hidden_states[:, t - 1, :]
                if t > 0
                else jnp.zeros((batch_size, self._hidden_dim))
            )

            dWxh += input_sequences[:, t, :].T @ dh_raw
            dWhh += h_prev.T @ dh_raw
            dbh += jnp.sum(dh_raw, axis=0)
            dX = dX.at[:, t, :].set(dh_raw @ self._parameters.Wxh.T)
            dh_next = dh_raw @ self._parameters.Whh.T

        self._cache["dP"] = d(
            self.Parameters(
                Wxh=self._clip_gradient_fn(dWxh, self._gradient_max_norm),
                Whh=self._clip_gradient_fn(dWhh, self._gradient_max_norm),
                bh=self._clip_gradient_fn(dbh, self._gradient_max_norm),
            )
        )

        return cast(D[Activations], dX)

    def empty_gradient(self) -> D[ParametersType]:
        return d(
            self.Parameters(
                Wxh=jnp.zeros_like(self._parameters.Wxh),
                Whh=jnp.zeros_like(self._parameters.Whh),
                bh=jnp.zeros_like(self._parameters.bh),
            )
        )

    def reinitialise(self) -> None:
        input_dim = self._parameters.Wxh.shape[0]
        self._key, subkey = jax.random.split(self._key)
        self._parameters = self._parameters_init_fn(input_dim, self._hidden_dim, subkey)

    def serialize(self) -> Recurrent.Serialized:
        return self.Serialized(
            layer_id=self._layer_id,
            input_dimensions=tuple(self._input_dimensions),
            output_dimensions=tuple(self._output_dimensions),
            hidden_dim=self._hidden_dim,
            return_sequences=self._return_sequences,
            parameters=self._parameters,
        )

    def update_parameters(self) -> None:
        if (dP := self._cache["dP"]) is None:
            raise ValueError("Gradient not set during backward pass")
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
        return (
            self._parameters.Wxh.size
            + self._parameters.Whh.size
            + self._parameters.bh.size
        )

    def write_serialized_parameters(self, buffer: IO[bytes]) -> None:
        self._write_header(buffer)
        if self._cache["dP"] is None:
            raise RuntimeError("Cache is not populated during serialization")
        dP = cast(ParametersType, self._cache["dP"])
        buffer.write(memoryview(dP.Wxh))
        buffer.write(memoryview(dP.Whh))
        buffer.write(memoryview(dP.bh))

    def read_serialized_parameters(self, data: IO[bytes]) -> None:
        if (layer_id := self.get_layer_id(data)) != self._layer_id:
            raise BadLayerId(f"Layer ID mismatch: {layer_id} != {self._layer_id}")
        update = self._parameters.from_bytes(data)
        if self._cache["dP"] is None:
            self._cache["dP"] = d(update)
        else:
            current_dP = cast(ParametersType, self._cache["dP"])
            self._cache["dP"] = d(current_dP + update)

    @property
    def parameter_nbytes(self) -> int:
        return (
            self._parameters.Wxh.nbytes
            + self._parameters.Whh.nbytes
            + self._parameters.bh.nbytes
        )

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def return_sequences(self) -> bool:
        return self._return_sequences
