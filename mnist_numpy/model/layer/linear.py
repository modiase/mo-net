from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Final, Self, TypedDict, cast

from more_itertools import one
import numpy as np

from mnist_numpy.functions import Identity, LeakyReLU, ReLU, Tanh
from mnist_numpy.model.layer.base import (
    Hidden,
)
from mnist_numpy.protos import (
    ActivationFn,
    Activations,
    D,
    Dimensions,
    GradLayer,
    SupportsGradientOperations,
    d,
    d_op,
)

EPSILON: Final[float] = 1e-8


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
            case Parameters():
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

    def __mul__(self, other: float | int | Self) -> Self:
        match other:
            case float() | int():
                return self.__class__(_W=other * self._W, _B=other * self._B)
            case self.__class__():
                return self.__class__(_W=self._W * other._W, _B=self._B * other._B)
            case _:
                return NotImplemented

    def __rmul__(self, other: float | Self) -> Self:
        return self.__mul__(other)

    def __truediv__(self, other: Self | float | int) -> Self:
        match other:
            case Parameters():
                return self.__class__(
                    _W=self._W / (other._W + EPSILON),
                    _B=self._B / (other._B + EPSILON),
                )
            case float() | int():
                return self.__mul__(1 / other)
            case _:
                return NotImplemented

    def __pow__(self, scalar: float | int) -> Self:
        return self.__class__(_W=self._W**scalar, _B=self._B**scalar)

    @classmethod
    def random(cls, dim_in: Dimensions, dim_out: Dimensions) -> Self:
        _dim_in = one(dim_in)
        _dim_out = one(dim_out)
        return cls(_W=np.random.randn(_dim_in, _dim_out), _B=np.zeros(_dim_out))

    @classmethod
    def xavier(cls, dim_in: Dimensions, dim_out: Dimensions) -> Self:
        _dim_in = one(dim_in)
        _dim_out = one(dim_out)
        return cls(
            _W=np.random.randn(_dim_in, _dim_out) * np.sqrt(1 / _dim_in),
            _B=np.zeros(_dim_out),
        )

    @classmethod
    def he(cls, dim_in: Dimensions, dim_out: Dimensions) -> Self:
        _dim_in = one(dim_in)
        _dim_out = one(dim_out)
        return cls(
            _W=np.random.normal(0, np.sqrt(2 / _dim_in), (_dim_in, _dim_out)),
            _B=np.zeros(_dim_out),
        )

    @classmethod
    def appropriate(
        cls, dim_in: Dimensions, dim_out: Dimensions, activation_fn: ActivationFn
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
    def eye(cls, dim: Dimensions) -> Self:
        _dim = one(dim)
        return cls(_W=np.eye(_dim), _B=np.zeros(_dim))

    @classmethod
    def of(cls, W: np.ndarray, B: np.ndarray) -> Self:
        return cls(_W=np.atleast_2d(W), _B=np.atleast_1d(B))


type ParametersType = Parameters


class Linear(Hidden):
    Parameters = Parameters
    _parameters: ParametersType
    _cache: Linear.Cache

    @classmethod
    def of_bias(cls, dim: Dimensions, bias: float | np.ndarray) -> Self:
        return cls(
            input_dimensions=dim,
            output_dimensions=dim,
            parameters=cls.Parameters.of(
                W=np.zeros((one(dim), one(dim))),
                B=(np.ones(one(dim)) * bias if isinstance(bias, float) else bias),
            ),
        )

    @classmethod
    def of_eye(cls, dim: Dimensions) -> Self:
        return cls(
            input_dimensions=dim,
            output_dimensions=dim,
            parameters=cls.Parameters.eye(dim),
        )

    class Cache(TypedDict):
        input_activations: Activations | None
        output_activations: Activations | None
        dP: D[ParametersType] | None

    @dataclass(frozen=True, kw_only=True)
    class Serialized:
        input_dimensions: tuple[int, ...]
        output_dimensions: tuple[int, ...]
        parameters: Parameters

        def deserialize(
            self,
            *,
            training: bool = False,
        ) -> Linear:
            del training  # unused
            return Linear(
                input_dimensions=self.input_dimensions,
                output_dimensions=self.output_dimensions,
                parameters=self.parameters,
            )

    def __init__(
        self,
        *,
        input_dimensions: Dimensions,
        output_dimensions: Dimensions | None = None,
        parameters: ParametersType | None = None,
        parameters_init_fn: Callable[[Dimensions, Dimensions], ParametersType] = (
            Parameters.xavier
        ),
        store_output_activations: bool = False,  # Only used for tracing
        freeze_parameters: bool = False,
        clip_gradients: bool = True,
        weight_max_norm: float = 1.0,
        bias_max_norm: float = 1.0,
    ):
        if output_dimensions is None:
            output_dimensions = input_dimensions
        super().__init__(
            input_dimensions=input_dimensions,
            output_dimensions=output_dimensions,
        )
        self._parameters_init_fn = parameters_init_fn
        self._freeze_parameters = freeze_parameters
        self._clip_gradients = clip_gradients
        self._weight_max_norm = weight_max_norm
        self._bias_max_norm = bias_max_norm
        if parameters is not None:
            if parameters._W.shape != (one(input_dimensions), one(output_dimensions)):
                raise ValueError(
                    f"Weight matrix shape ({parameters._W.shape}) "
                    f"does not match input dimensions ({input_dimensions}) "
                    f"and output dimensions ({output_dimensions})."
                )
            if parameters._B.shape != (one(output_dimensions),):
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
        self._cache: Linear.Cache = {
            "input_activations": None,
            "output_activations": None,
            "dP": None,
        }

    def _forward_prop(self, *, input_activations: Activations) -> Activations:
        self._cache["input_activations"] = input_activations
        output_activations = Activations(
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

        dW = input_activations.T @ dZ
        dB = d_op(dZ, np.sum)

        if self._clip_gradients:
            W_norm = np.linalg.norm(dW)
            if W_norm > self._weight_max_norm:
                dW = dW * (self._weight_max_norm / W_norm)

            B_norm = np.linalg.norm(dB)
            if B_norm > self._bias_max_norm:
                dB = dB * (self._bias_max_norm / B_norm)

        self._cache["dP"] = cast(
            D[Parameters],
            self.Parameters.of(W=dW, B=dB),
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

    def serialize(self) -> Linear.Serialized:
        return self.Serialized(
            input_dimensions=tuple(self._input_dimensions),
            output_dimensions=tuple(self._output_dimensions),
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
    def cache(self) -> Linear.Cache:
        return self._cache

    @property
    def parameters(self) -> ParametersType:
        return self._parameters

    @property
    def parameter_count(self) -> int:
        return self._parameters._W.size + self._parameters._B.size
