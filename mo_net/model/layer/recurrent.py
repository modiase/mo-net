from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import IO, Self

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
    weights_ih: jnp.ndarray  # input-to-hidden
    weights_hh: jnp.ndarray  # hidden-to-hidden
    biases: jnp.ndarray

    def __getitem__(
        self, index: int | tuple[int, ...]
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self.weights_ih[index], self.weights_hh[index], self.biases[index]

    def __add__(self, other: Self | float | int) -> Self:
        match other:
            case Parameters():
                return self.__class__(
                    weights_ih=self.weights_ih + other.weights_ih,
                    weights_hh=self.weights_hh + other.weights_hh,
                    biases=self.biases + other.biases,
                )
            case float() | int():
                return self.__class__(
                    weights_ih=self.weights_ih + other,
                    weights_hh=self.weights_hh + other,
                    biases=self.biases + other,
                )
            case _:
                return NotImplemented

    def __radd__(self, other: Self | float | int) -> Self:
        return self.__add__(other)

    def __neg__(self) -> Self:
        return self.__class__(
            weights_ih=-self.weights_ih,
            weights_hh=-self.weights_hh,
            biases=-self.biases,
        )

    def __sub__(self, other: Self | float | int) -> Self:
        return self.__add__(-other)

    def __rsub__(self, other: Self | float | int) -> Self:
        return self.__sub__(other)

    def __mul__(self, other: float | int | Self) -> Self:
        match other:
            case float() | int():
                return self.__class__(
                    weights_ih=other * self.weights_ih,
                    weights_hh=other * self.weights_hh,
                    biases=other * self.biases,
                )
            case self.__class__():
                return self.__class__(
                    weights_ih=self.weights_ih * other.weights_ih,
                    weights_hh=self.weights_hh * other.weights_hh,
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
                    weights_ih=self.weights_ih / (other.weights_ih + EPSILON),
                    weights_hh=self.weights_hh / (other.weights_hh + EPSILON),
                    biases=self.biases / (other.biases + EPSILON),
                )
            case float() | int():
                return self.__mul__(1 / other)
            case _:
                return NotImplemented

    def __pow__(self, scalar: float | int) -> Self:
        return self.__class__(
            weights_ih=self.weights_ih**scalar,
            weights_hh=self.weights_hh**scalar,
            biases=self.biases**scalar,
        )

    @classmethod
    def random(
        cls,
        dim_in: Dimensions,
        dim_hidden: Dimensions,
        key: jax.Array,
    ) -> Self:
        _dim_in = one(dim_in)
        _dim_hidden = one(dim_hidden)
        key1, key2 = jax.random.split(key)
        return cls(
            weights_ih=jax.random.normal(key1, (_dim_in, _dim_hidden)),
            weights_hh=jax.random.normal(key2, (_dim_hidden, _dim_hidden)),
            biases=jnp.zeros(_dim_hidden),
        )

    @classmethod
    def xavier(cls, dim_in: Dimensions, dim_hidden: Dimensions, key: jax.Array) -> Self:
        _dim_in = one(dim_in)
        _dim_hidden = one(dim_hidden)
        key1, key2 = jax.random.split(key)
        return cls(
            weights_ih=jax.random.normal(key1, (_dim_in, _dim_hidden))
            * jnp.sqrt(1 / _dim_in),
            weights_hh=jax.random.normal(key2, (_dim_hidden, _dim_hidden))
            * jnp.sqrt(1 / _dim_hidden),
            biases=jnp.zeros(_dim_hidden),
        )

    @classmethod
    def orthogonal(
        cls, dim_in: Dimensions, dim_hidden: Dimensions, key: jax.Array
    ) -> Self:
        """Orthogonal initialization for recurrent weights (better for RNNs)."""
        _dim_in = one(dim_in)
        _dim_hidden = one(dim_hidden)
        key1, key2 = jax.random.split(key)

        # Xavier for input weights
        weights_ih = jax.random.normal(key1, (_dim_in, _dim_hidden)) * jnp.sqrt(
            1 / _dim_in
        )

        # Orthogonal for recurrent weights
        random_matrix = jax.random.normal(key2, (_dim_hidden, _dim_hidden))
        q, r = jnp.linalg.qr(random_matrix)
        weights_hh = q * jnp.sign(jnp.diag(r))

        return cls(
            weights_ih=weights_ih,
            weights_hh=weights_hh,
            biases=jnp.zeros(_dim_hidden),
        )

    @classmethod
    def appropriate(
        cls,
        dim_in: Dimensions,
        dim_hidden: Dimensions,
        *,
        activation_fn: ActivationFn,
        key: jax.Array,
    ) -> Self:
        """Choose appropriate initialization based on activation function."""
        # For RNNs, orthogonal initialization is generally preferred for recurrent weights
        # Regardless of activation function
        return cls.orthogonal(dim_in, dim_hidden, key)

    @classmethod
    def of(cls, W_ih: jnp.ndarray, W_hh: jnp.ndarray, B: jnp.ndarray) -> Self:
        return cls(
            weights_ih=jnp.atleast_2d(W_ih),
            weights_hh=jnp.atleast_2d(W_hh),
            biases=jnp.atleast_1d(B),
        )

    def from_bytes(self, data: IO[bytes]) -> Self:
        W_ih = jnp.frombuffer(
            data.read(self.weights_ih.nbytes), dtype=self.weights_ih.dtype
        ).reshape(self.weights_ih.shape)
        W_hh = jnp.frombuffer(
            data.read(self.weights_hh.nbytes), dtype=self.weights_hh.dtype
        ).reshape(self.weights_hh.shape)
        B = jnp.frombuffer(
            data.read(self.biases.nbytes), dtype=self.biases.dtype
        ).reshape(self.biases.shape)
        return self.__class__(weights_ih=W_ih, weights_hh=W_hh, biases=B)


type ParametersType = Parameters


class Cache(GradCache[ParametersType]):
    input_activations: Activations | None
    hidden_states: jnp.ndarray | None  # All hidden states for BPTT
    output_activations: Activations | None


type CacheType = Cache


class Recurrent(ParametrisedHidden[ParametersType, CacheType]):
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
        hidden_dimensions: tuple[int, ...]
        return_sequences: bool
        parameters: Parameters

        def deserialize(
            self,
            *,
            training: bool = False,
            freeze_parameters: bool = False,
        ) -> Recurrent:
            del training  # unused
            return Recurrent(
                layer_id=self.layer_id,
                input_dimensions=self.input_dimensions,
                hidden_dimensions=self.hidden_dimensions,
                return_sequences=self.return_sequences,
                parameters_init_fn=lambda _, __: self.parameters,
                freeze_parameters=freeze_parameters,
            )

    def __init__(
        self,
        *,
        activation_fn: ActivationFn = identity,
        clip_gradients: bool = True,
        freeze_parameters: bool = False,
        hidden_dimensions: Dimensions,
        input_dimensions: Dimensions,
        layer_id: str | None = None,
        parameters_init_fn: Callable[[Dimensions, Dimensions], ParametersType],
        return_sequences: bool = True,
        stateful: bool = False,
        store_output_activations: bool = False,
        weight_max_norm: float = 1.0,
    ):
        # Output dimensions depend on return_sequences
        output_dimensions = hidden_dimensions if return_sequences else hidden_dimensions
        super().__init__(
            layer_id=layer_id,
            input_dimensions=input_dimensions,
            output_dimensions=output_dimensions,
        )
        self._freeze_parameters = freeze_parameters
        self._hidden_dimensions = hidden_dimensions
        self._return_sequences = return_sequences
        self._stateful = stateful
        self._activation_fn = activation_fn
        self._weight_max_norm = weight_max_norm

        self._clip_gradient_fn = (
            self._clip_gradient_impl if clip_gradients else self._no_clip_gradient
        )

        self._parameters_init_fn = parameters_init_fn
        self._parameters = parameters_init_fn(input_dimensions, hidden_dimensions)

        # Validate parameter shapes
        _dim_in = one(input_dimensions)
        _dim_hidden = one(hidden_dimensions)
        if self._parameters.weights_ih.shape != (_dim_in, _dim_hidden):
            raise ValueError(
                f"Input weight matrix shape ({self._parameters.weights_ih.shape}) "
                f"does not match input dimensions ({input_dimensions}) "
                f"and hidden dimensions ({hidden_dimensions})."
            )
        if self._parameters.weights_hh.shape != (_dim_hidden, _dim_hidden):
            raise ValueError(
                f"Hidden weight matrix shape ({self._parameters.weights_hh.shape}) "
                f"does not match hidden dimensions ({hidden_dimensions})."
            )
        if self._parameters.biases.shape != (_dim_hidden,):
            raise ValueError(
                f"Bias vector shape ({self._parameters.biases.shape}) "
                f"does not match hidden dimensions ({hidden_dimensions})."
            )

        self._store_output_activations = store_output_activations
        self._hidden_state: jnp.ndarray | None = None  # For stateful mode
        self._cache: CacheType = {
            "input_activations": None,
            "hidden_states": None,
            "output_activations": None,
            "dP": None,
        }

    def forward_prop(self, input_activations: Activations) -> Activations:
        """
        Override base forward_prop to handle 3D sequential inputs.

        RNN layers accept (batch, seq_len, input_dim) unlike feedforward layers
        which expect (batch, input_dim).
        """
        input_activations = Activations(jnp.atleast_2d(input_activations))

        if input_activations.ndim == 2:
            if input_activations.shape[1:] != self.input_dimensions:
                raise ValueError(
                    f"Input activations shape {input_activations.shape[1:]} does not match "
                    f"input dimensions {self.input_dimensions} in layer {self}."
                )
        elif input_activations.ndim == 3:
            if input_activations.shape[2:] != self.input_dimensions:
                raise ValueError(
                    f"Input activations feature dimension {input_activations.shape[2:]} does not match "
                    f"input dimensions {self.input_dimensions} in layer {self}."
                )
        else:
            raise ValueError(
                f"RNN layer expects 2D or 3D input, got shape {input_activations.shape}"
            )

        return self._forward_prop(input_activations=input_activations)

    def _forward_prop(self, *, input_activations: Activations) -> Activations:
        """
        Forward propagation through RNN.

        Input shape: (batch, seq_len, input_dim) or (batch, input_dim) for single timestep
        Output shape: (batch, seq_len, hidden_dim) if return_sequences else (batch, hidden_dim)
        """
        # Ensure input is 3D: (batch, seq_len, input_dim)
        if input_activations.ndim == 2:
            input_activations = jnp.expand_dims(input_activations, axis=1)

        self._cache["input_activations"] = input_activations
        batch_size, seq_len, _ = input_activations.shape
        hidden_dim = one(self._hidden_dimensions)

        # Initialize hidden state
        if self._stateful and self._hidden_state is not None:
            h0 = self._hidden_state
        else:
            h0 = jnp.zeros((batch_size, hidden_dim))

        @jit
        def rnn_step(h_prev, x_t):
            """Single RNN step: h_t = activation(x_t @ W_ih + h_{t-1} @ W_hh + b)"""
            h_t = self._activation_fn(
                x_t @ self._parameters.weights_ih
                + h_prev @ self._parameters.weights_hh
                + self._parameters.biases
            )
            return h_t, h_t

        # Unroll RNN across time using scan
        # input_activations has shape (batch, seq_len, input_dim)
        # We need to transpose to (seq_len, batch, input_dim) for scan
        inputs_transposed = jnp.transpose(input_activations, (1, 0, 2))

        final_h, all_h = jax.lax.scan(rnn_step, h0, inputs_transposed)

        # all_h has shape (seq_len, batch, hidden_dim)
        # Transpose back to (batch, seq_len, hidden_dim)
        all_h = jnp.transpose(all_h, (1, 0, 2))

        # Store hidden states for backward pass
        self._cache["hidden_states"] = jnp.concatenate(
            [jnp.expand_dims(h0, axis=1), all_h], axis=1
        )  # (batch, seq_len+1, hidden_dim)

        # Update stateful hidden state
        if self._stateful:
            self._hidden_state = final_h

        # Return based on return_sequences flag
        if self._return_sequences:
            output_activations = Activations(all_h)
        else:
            output_activations = Activations(final_h)

        if self._store_output_activations:
            self._cache["output_activations"] = output_activations

        return output_activations

    def _backward_prop(
        self,
        *,
        dZ: D[Activations],
    ) -> D[Activations]:
        """
        Backward propagation through time (BPTT).

        Computes gradients w.r.t. parameters and input activations.
        """
        if (input_activations := self._cache["input_activations"]) is None:
            raise ValueError("Input activations not set during forward pass.")
        if (hidden_states := self._cache["hidden_states"]) is None:
            raise ValueError("Hidden states not set during forward pass.")

        batch_size, seq_len, input_dim = input_activations.shape
        hidden_dim = one(self._hidden_dimensions)

        # Handle gradient based on return_sequences
        if self._return_sequences:
            # dZ has shape (batch, seq_len, hidden_dim)
            dh_all = dZ
        else:
            # dZ has shape (batch, hidden_dim)
            # Create gradient array with zeros except at last timestep
            dh_all = jnp.zeros((batch_size, seq_len, hidden_dim))
            dh_all = dh_all.at[:, -1, :].set(dZ)

        # Initialize gradients
        dW_ih = jnp.zeros_like(self._parameters.weights_ih)
        dW_hh = jnp.zeros_like(self._parameters.weights_hh)
        dB = jnp.zeros_like(self._parameters.biases)
        dx_all = jnp.zeros_like(input_activations)

        @jit
        def bptt_step(carry, t):
            """Backward pass for single timestep."""
            dh_next, dW_ih_acc, dW_hh_acc, dB_acc = carry

            # Get current gradients and activations
            dh_t = (
                dh_all[:, t, :] + dh_next
            )  # Gradient from output + gradient from next timestep
            h_t = hidden_states[:, t + 1, :]  # Current hidden state
            h_prev = hidden_states[:, t, :]  # Previous hidden state
            x_t = input_activations[:, t, :]  # Current input

            # Gradient through activation function
            dh_raw = dh_t * self._activation_fn.deriv(
                x_t @ self._parameters.weights_ih
                + h_prev @ self._parameters.weights_hh
                + self._parameters.biases
            )

            # Gradients w.r.t. parameters
            dW_ih_t = x_t.T @ dh_raw
            dW_hh_t = h_prev.T @ dh_raw
            dB_t = jnp.sum(dh_raw, axis=0)

            # Gradients w.r.t. inputs
            dx_t = dh_raw @ self._parameters.weights_ih.T
            dh_prev = dh_raw @ self._parameters.weights_hh.T

            return (
                dh_prev,
                dW_ih_acc + dW_ih_t,
                dW_hh_acc + dW_hh_t,
                dB_acc + dB_t,
            ), dx_t

        # Run BPTT backward through time
        init_carry = (jnp.zeros((batch_size, hidden_dim)), dW_ih, dW_hh, dB)
        (_, dW_ih, dW_hh, dB), dx_all_transposed = jax.lax.scan(
            bptt_step, init_carry, jnp.arange(seq_len - 1, -1, -1)
        )

        # Reverse dx_all since we scanned backward
        dx_all = jnp.flip(dx_all_transposed, axis=0)
        dx_all = jnp.transpose(dx_all, (1, 0, 2))  # (batch, seq_len, input_dim)

        # Apply gradient clipping
        dW_ih = self._clip_gradient_fn(dW_ih, self._weight_max_norm)
        dW_hh = self._clip_gradient_fn(dW_hh, self._weight_max_norm)
        dB = self._clip_gradient_fn(dB, self._weight_max_norm)

        self._cache["dP"] = d(
            self.Parameters(weights_ih=dW_ih, weights_hh=dW_hh, biases=dB)
        )
        return dx_all

    def empty_gradient(self) -> D[ParametersType]:
        return d(
            self.Parameters(
                weights_ih=jnp.zeros_like(self._parameters.weights_ih),
                weights_hh=jnp.zeros_like(self._parameters.weights_hh),
                biases=jnp.zeros_like(self._parameters.biases),
            )
        )

    def reinitialise(self) -> None:
        self._parameters = self._parameters_init_fn(
            self._input_dimensions, self._hidden_dimensions
        )
        if self._stateful:
            self._hidden_state = None

    def serialize(self) -> Recurrent.Serialized:
        return self.Serialized(
            layer_id=self._layer_id,
            input_dimensions=tuple(self._input_dimensions),
            hidden_dimensions=tuple(self._hidden_dimensions),
            return_sequences=self._return_sequences,
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

    def reset_state(self) -> None:
        """Reset hidden state (useful for stateful mode)."""
        self._hidden_state = None

    @property
    def cache(self) -> CacheType:
        return self._cache

    @property
    def parameters(self) -> ParametersType:
        return self._parameters

    @property
    def parameter_count(self) -> int:
        return (
            self._parameters.weights_ih.size
            + self._parameters.weights_hh.size
            + self._parameters.biases.size
        )

    def write_serialized_parameters(self, buffer: IO[bytes]) -> None:
        self._write_header(buffer)
        if self._cache["dP"] is None:
            raise RuntimeError("Cache is not populated during serialization.")
        buffer.write(memoryview(self._cache["dP"].weights_ih))
        buffer.write(memoryview(self._cache["dP"].weights_hh))
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
        return (
            self._parameters.weights_ih.nbytes
            + self._parameters.weights_hh.nbytes
            + self._parameters.biases.nbytes
        )
