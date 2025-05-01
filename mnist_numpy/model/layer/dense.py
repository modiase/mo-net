from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Self, TypedDict, cast

import numpy as np

from mnist_numpy.functions import Identity, LeakyReLU, ReLU, Tanh
from mnist_numpy.model.layer.base import (
    _Hidden,
)
from mnist_numpy.types import (
    ActivationFn,
    Activations,
    D,
    GradLayer,
    SupportsGradientOperations,
    d,
    d_op,
)


@dataclass(kw_only=True, frozen=True)
class Parameters(SupportsGradientOperations):
    _W: np.ndarray
    _B: np.ndarray

    def __getitem__(
        self, index: int | tuple[int, ...]
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._W[index], self._B[index]

    def __add__(self, other: Self | float | int) -> Self:
        match other:
            case self.__class__():
                return self.__class__(_W=self._W + other._W, _B=self._B + other._B)
            case float() | int():
                return self.__class__(_W=self._W + other, _B=self._B + other)
            case _:
                return NotImplemented

    def __radd__(self, other: Self | float | int) -> Self:
        return self.__add__(other)

    def __neg__(self) -> Self:
        return self.__class__(_W=-self._W, _B=-self._B)

    def __sub__(self, other: Self | float | int) -> Self:
        return self.__add__(-other)

    def __rsub__(self, other: Self | float | int) -> Self:
        return self.__sub__(other)

    def __mul__(self, other: float | int) -> Self:
        match other:
            case float() | int():
                return self.__class__(_W=other * self._W, _B=other * self._B)
            case _:
                return NotImplemented

    def __rmul__(self, other: float | int) -> Self:
        return self.__mul__(other)

    def __truediv__(self, other: Self | float | int) -> Self:
        match other:
            case self.__class__():
                return self.__class__(
                    _W=self._W / other._W,
                    _B=self._B / other._B,
                )
            case float() | int():
                return self.__mul__(1 / other)
            case _:
                return NotImplemented

    def __pow__(self, scalar: float | int) -> Self:
        return self.__class__(_W=self._W**scalar, _B=self._B**scalar)

    @classmethod
    def random(cls, dim_in: int, dim_out: int) -> Self:
        return cls(_W=np.random.randn(dim_in, dim_out), _B=np.zeros(dim_out))

    @classmethod
    def xavier(cls, dim_in: int, dim_out: int) -> Self:
        return cls(
            _W=np.random.randn(dim_in, dim_out) * np.sqrt(1 / dim_in),
            _B=np.zeros(dim_out),
        )

    @classmethod
    def he(cls, dim_in: int, dim_out: int) -> Self:
        return cls(
            _W=np.random.normal(0, np.sqrt(2 / dim_in), (dim_in, dim_out)),
            _B=np.zeros(dim_out),
        )

    @classmethod
    def appropriate(
        cls, dim_in: int, dim_out: int, activation_fn: ActivationFn
    ) -> Self:
        if activation_fn == ReLU or activation_fn == LeakyReLU:
            return cls.he(dim_in, dim_out)
        elif activation_fn == Tanh or activation_fn == Identity:
            return cls.xavier(dim_in, dim_out)
        else:
            raise ValueError(
                f"Cannot choose appropriate initialisation for {activation_fn}"
            )

    @classmethod
    def eye(cls, dim: int) -> Self:
        return cls(_W=np.eye(dim), _B=np.zeros(dim))

    @classmethod
    def of(cls, W: np.ndarray, B: np.ndarray) -> Self:
        return cls(_W=np.atleast_2d(W), _B=np.atleast_1d(B))


type ParametersType = Parameters


class Dense(_Hidden):
    Parameters = Parameters
    _parameters: ParametersType
    _cache: Dense.Cache

    class Cache(TypedDict):
        input_activations: Activations | None
        output_activations: Activations | None
        dP: D[ParametersType] | None

    @dataclass(frozen=True, kw_only=True)
    class Serialized:
        input_dimensions: int
        output_dimensions: int
        parameters: Parameters

        def deserialize(
            self,
            *,
            training: bool = False,
        ) -> Dense:
            del training  # unused
            return Dense(
                input_dimensions=self.input_dimensions,
                output_dimensions=self.output_dimensions,
                parameters=self.parameters,
            )

    def __init__(
        self,
        *,
        input_dimensions: int,
        output_dimensions: int,
        parameters: ParametersType | None = None,
        parameters_init_fn: Callable[[int, int], ParametersType] = Parameters.xavier,
        store_output_activations: bool = False,  # Only used for tracing
    ):
        super().__init__(
            input_dimensions=input_dimensions,
            output_dimensions=output_dimensions,
        )
        self._parameters_init_fn = parameters_init_fn
        if parameters is not None:
            if parameters._W.shape != (input_dimensions, output_dimensions):
                raise ValueError(
                    f"Weight matrix shape ({parameters._W.shape}) "
                    f"does not match input dimensions ({input_dimensions}) "
                    f"and output dimensions ({output_dimensions})."
                )
            if parameters._B.shape != (output_dimensions,):
                raise ValueError(
                    f"Bias vector shape ({parameters._B.shape}) "
                    f"does not match output dimensions ({output_dimensions})."
                )

        self._parameters = (
            parameters
            if parameters is not None
            else self._parameters_init_fn(input_dimensions, output_dimensions)
        )
        self._store_output_activations = store_output_activations
        self._cache: Dense.Cache = {
            "input_activations": None,
            "output_activations": None,
            "dP": None,
        }

    def _forward_prop(self, *, input_activations: Activations) -> Activations:
        self._cache["input_activations"] = input_activations
        output_activations = (
            input_activations @ self._parameters._W + self._parameters._B
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
        self._cache["dP"] = cast(
            D[Parameters],
            self.Parameters.of(
                W=input_activations.T @ dZ,
                B=d_op(dZ, np.sum),
            ),
        )
        return dZ @ self._parameters._W.T

    def empty_gradient(self) -> D[ParametersType]:
        return d(
            self.Parameters(
                _W=np.zeros_like(self._parameters._W),
                _B=np.zeros_like(self._parameters._B),
            )
        )

    def reinitialise(self) -> None:
        self._parameters = self._parameters_init_fn(
            self.input_dimensions, self.output_dimensions
        )

    def serialize(self) -> Dense.Serialized:
        return self.Serialized(
            input_dimensions=self._input_dimensions,
            output_dimensions=self._output_dimensions,
            parameters=self._parameters,
        )

    def update_parameters(self) -> None:
        if self._cache["dP"] is None:
            raise ValueError("Gradient not set during backward pass.")
        self._parameters = self._parameters + self._cache["dP"]
        self._cache["dP"] = None

    def gradient_operation(self, f: Callable[[GradLayer], None]) -> None:
        f(self)

    @property
    def cache(self) -> Dense.Cache:
        return self._cache

    @property
    def parameters(self) -> ParametersType:
        return self._parameters

    @property
    def parameter_count(self) -> int:
        return self._parameters._W.size + self._parameters._B.size
