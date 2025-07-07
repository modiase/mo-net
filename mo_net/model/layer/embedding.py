from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import IO, Self

import jax.numpy as jnp
import jax.random as random
from jax import jit
from more_itertools import one

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


@dataclass(kw_only=True)
class Parameters(SupportsGradientOperations):
    embeddings: jnp.ndarray

    def __add__(self, other: Self | float | int) -> Self:
        match other:
            case Parameters():
                return self.__class__(embeddings=self.embeddings + other.embeddings)
            case float() | int():
                return self.__class__(embeddings=self.embeddings + other)
            case _:
                return NotImplemented

    def __radd__(self, other: Self | float | int) -> Self:
        return self.__add__(other)

    def __neg__(self) -> Self:
        return self.__class__(embeddings=-self.embeddings)

    def __sub__(self, other: Self | float | int) -> Self:
        return self.__add__(-other)

    def __rsub__(self, other: Self | float | int) -> Self:
        return self.__sub__(other)

    def __mul__(self, other: float | int | Self) -> Self:
        match other:
            case float() | int():
                return self.__class__(embeddings=other * self.embeddings)
            case self.__class__():
                return self.__class__(embeddings=self.embeddings * other.embeddings)
            case _:
                return NotImplemented

    def __rmul__(self, other: float | Self) -> Self:
        return self.__mul__(other)

    def __truediv__(self, other: Self | float | int) -> Self:
        match other:
            case Parameters():
                return self.__class__(
                    embeddings=self.embeddings / (other.embeddings + EPSILON)
                )
            case float() | int():
                return self.__mul__(1 / other)
            case _:
                return NotImplemented

    def __pow__(self, scalar: float | int) -> Self:
        return self.__class__(embeddings=self.embeddings**scalar)

    @classmethod
    def random(cls, vocab_size: int, embedding_dim: int) -> Self:
        key = random.PRNGKey(42)
        return cls(embeddings=random.normal(key, (vocab_size, embedding_dim)))

    @classmethod
    def xavier(cls, vocab_size: int, embedding_dim: int) -> Self:
        key = random.PRNGKey(42)
        embeddings = random.normal(key, (vocab_size, embedding_dim)) * jnp.sqrt(
            1 / vocab_size
        )
        return cls(embeddings=embeddings)

    @classmethod
    def he(cls, vocab_size: int, embedding_dim: int) -> Self:
        key = random.PRNGKey(42)
        embeddings = random.normal(key, (vocab_size, embedding_dim)) * jnp.sqrt(
            2 / vocab_size
        )
        return cls(embeddings=embeddings)

    @classmethod
    def of(cls, embeddings: jnp.ndarray) -> Self:
        return cls(embeddings=jnp.atleast_2d(embeddings))

    def from_bytes(self, data: IO[bytes]) -> Self:
        embeddings = jnp.frombuffer(
            data.read(self.embeddings.nbytes), dtype=self.embeddings.dtype
        ).reshape(self.embeddings.shape)
        return self.__class__(embeddings=embeddings)


type ParametersType = Parameters


class Cache(GradCache[ParametersType]):
    input_indices: jnp.ndarray | None
    output_activations: Activations | None


type CacheType = Cache


class Embedding(ParametrisedHidden[ParametersType, CacheType]):
    Parameters = Parameters
    Cache = Cache
    _parameters: ParametersType
    _cache: CacheType

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
        input_dimensions: tuple[int, ...]
        output_dimensions: tuple[int, ...]
        vocab_size: int
        parameters: Parameters

        def deserialize(self, *, training: bool = False) -> Embedding:
            del training
            return Embedding(
                layer_id=self.layer_id,
                input_dimensions=self.input_dimensions,
                output_dimensions=self.output_dimensions,
                vocab_size=self.vocab_size,
                parameters=self.parameters,
            )

    def __init__(
        self,
        *,
        clip_gradients: bool = True,
        freeze_parameters: bool = False,
        input_dimensions: Dimensions,
        layer_id: str | None = None,
        output_dimensions: Dimensions,
        parameters: ParametersType | None = None,
        parameters_init_fn: Callable[[int, int], ParametersType] = Parameters.xavier,
        store_output_activations: bool = False,
        vocab_size: int,
        weight_max_norm: float = 1.0,
    ):
        super().__init__(
            layer_id=layer_id,
            input_dimensions=input_dimensions,
            output_dimensions=output_dimensions,
        )
        self._parameters_init_fn = parameters_init_fn
        self._store_output_activations = store_output_activations
        self._freeze_parameters = freeze_parameters
        self._weight_max_norm = weight_max_norm

        # Assign the appropriate gradient clipping function
        self._clip_gradient_fn = (
            self._clip_gradient_impl if clip_gradients else self._no_clip_gradient
        )

        embedding_dim = (
            output_dimensions[-1]
            if len(output_dimensions) > 1
            else one(output_dimensions)
        )

        if parameters is not None and parameters.embeddings.shape != (
            vocab_size,
            embedding_dim,
        ):
            raise ValueError(
                f"Embedding matrix shape {parameters.embeddings.shape} does not match vocab_size {vocab_size} and embedding_dim {embedding_dim}"
            )

        self._vocab_size = vocab_size
        self._parameters = (
            parameters
            if parameters is not None
            else self._parameters_init_fn(vocab_size, embedding_dim)
        )
        self._cache: CacheType = {
            "input_indices": None,
            "output_activations": None,
            "dP": None,
        }

    def _forward_prop(self, *, input_activations: Activations) -> Activations:
        indices = input_activations.astype(int)
        self._cache["input_indices"] = indices
        embeddings = self._parameters.embeddings[indices]
        output_activations = Activations(embeddings)
        if self._store_output_activations:
            self._cache["output_activations"] = output_activations
        return output_activations

    def _backward_prop(self, *, dZ: D[Activations]) -> D[Activations]:
        if (indices := self._cache["input_indices"]) is None:
            raise ValueError("Input indices not set during forward pass.")

        dE = jnp.zeros_like(self._parameters.embeddings)
        dE = dE.at[indices].add(dZ)

        dE = self._clip_gradient_fn(dE, self._weight_max_norm)

        self._cache["dP"] = d(self.Parameters(embeddings=dE))
        return dZ

    def empty_gradient(self) -> D[ParametersType]:
        return d(
            self.Parameters(embeddings=jnp.zeros_like(self._parameters.embeddings))
        )

    def reinitialise(self) -> None:
        vocab_size = self._vocab_size
        embedding_dim = (
            self.output_dimensions[-1]
            if len(self.output_dimensions) > 1
            else one(self.output_dimensions)
        )
        self._parameters = self._parameters_init_fn(vocab_size, embedding_dim)

    def serialize(self) -> Embedding.Serialized:
        return self.Serialized(
            layer_id=self._layer_id,
            input_dimensions=tuple(self._input_dimensions),
            output_dimensions=tuple(self._output_dimensions),
            vocab_size=self._vocab_size,
            parameters=self._parameters,
        )

    def update_parameters(self) -> None:
        if self._cache["dP"] is None:
            raise ValueError("Gradient not set during backward pass.")
        if not self._freeze_parameters:
            self._parameters = self._parameters + self._cache["dP"]
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
        return self._parameters.embeddings.size

    def write_serialized_parameters(self, buffer: IO[bytes]) -> None:
        self._write_header(buffer)
        if self._cache["dP"] is None:
            raise RuntimeError("Cache is not populated during serialization.")
        buffer.write(memoryview(self._cache["dP"].embeddings))

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
        return self._parameters.embeddings.nbytes

    @property
    def vocab_size(self) -> int:
        return self._vocab_size
