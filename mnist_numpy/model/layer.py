from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Self, TypeVar, cast

import numpy as np

from mnist_numpy.functions import eye, softmax
from mnist_numpy.types import ActivationFn, Activations, D, PreActivations

_ParamType = TypeVar("_ParamType")


class _LayerBase(ABC, Generic[_ParamType]):
    Parameters: type[_ParamType]
    _parameters: _ParamType
    _previous_layer: Layer

    def __init__(
        self,
        *,
        neurons: int,
        activation_fn: ActivationFn,
    ):
        self._activation_fn = activation_fn
        self._neurons = neurons

    @abstractmethod
    def _forward_prop(
        self, *, As: Activations
    ) -> tuple[PreActivations, Activations]: ...

    def _update_parameters(self, params: D[_ParamType]) -> None:
        self._parameters = params + self._parameters

    @property
    @abstractmethod
    def neurons(self) -> int: ...

    @abstractmethod
    def empty_parameters(self) -> _ParamType: ...


@dataclass(kw_only=True, frozen=True)
class DenseParameters:
    _W: np.ndarray
    _B: np.ndarray

    def __add__(self, other: Self | float) -> Self:
        match other:
            case self.__class__():
                return self.__class__(_W=self._W + other._W, _B=self._B + other._B)
            case float():
                return self.__class__(_W=self._W + other, _B=self._B + other)
            case _:
                return NotImplemented

    def __radd__(self, other: Self | float) -> Self:
        return self.__add__(other)

    def __mul__(self, other: float) -> Self:
        if not isinstance(other, float):
            return NotImplemented
        return self.__class__(_W=other * self._W, _B=other * self._B)

    def __rmul__(self, other: float) -> Self:
        if not isinstance(other, float):
            return NotImplemented
        return self.__mul__(other)

    def __truediv__(self, other: Self | float) -> Self:
        match other:
            case self.__class__():
                return self.__class__(
                    _W=self._W / other._W,
                    _B=self._B / other._B,
                )
            case float():
                return self.__mul__(1 / other)
            case _:
                return NotImplemented

    def __pow__(self, scalar: float) -> Self:
        return self.__class__(_W=self._W**scalar, _B=self._B**scalar)

    @classmethod
    def empty(cls, *, dim_in: int = 0, dim_out: int = 0) -> Self:
        return cls(_W=np.zeros((dim_in, dim_out)), _B=np.zeros(dim_out))

    @classmethod
    def random(cls, dim_in: int, dim_out: int) -> Self:
        return cls(_W=np.random.randn(dim_in, dim_out), _B=np.zeros(dim_out))

    @classmethod
    def eye(cls, dim: int) -> Self:
        return cls(_W=np.eye(dim), _B=np.zeros(dim))

    @classmethod
    def of(cls, W: np.ndarray, B: np.ndarray) -> Self:
        return cls(_W=np.atleast_2d(W), _B=np.atleast_1d(B))


class HiddenLayerBase(
    _LayerBase[_ParamType]
):  # TODO: Consider merging this with OutputLayerBase
    @abstractmethod
    def _backward_prop(
        self,
        *,
        As_prev: Activations,
        Zs_prev: PreActivations,
        dZ: D[PreActivations],
    ) -> tuple[D[DenseParameters], D[PreActivations]]: ...


class DenseLayer(HiddenLayerBase[DenseParameters]):
    Parameters = DenseParameters
    _parameters: DenseParameters

    def __init__(
        self,
        *,
        neurons: int,
        activation_fn: ActivationFn,
        parameters: DenseParameters | None = None,
        previous_layer: Layer,
    ):
        super().__init__(neurons=neurons, activation_fn=activation_fn)
        self._previous_layer = previous_layer
        self._parameters = (
            parameters
            if parameters is not None
            else self.Parameters.random(previous_layer.neurons, self._neurons)
        )

    def _forward_prop(self, As: Activations) -> tuple[PreActivations, Activations]:
        preactivations = As @ self._parameters._W + self._parameters._B
        return preactivations, self._activation_fn(preactivations)

    def _backward_prop(
        self,
        *,
        As_prev: Activations,
        Zs_prev: PreActivations,
        dZ: D[PreActivations],
    ) -> tuple[D[DenseParameters], D[PreActivations]]:
        return cast(  # TODO: fix-types
            D[DenseParameters],
            self.Parameters.of(
                W=As_prev.T @ dZ,
                B=np.sum(
                    cast(  # TODO: fix-types
                        float | np.ndarray, dZ
                    ),
                    axis=0,
                ),
            ),
        ), dZ @ self._parameters._W.T * self._activation_fn.deriv(Zs_prev)

    @property
    def neurons(self) -> int:
        return self._neurons

    def empty_parameters(self) -> DenseParameters:
        return self.Parameters.empty(
            dim_in=self._previous_layer.neurons, dim_out=self._neurons
        )

    @property
    def parameters(self) -> DenseParameters:
        return self._parameters


class OutputLayerBase(_LayerBase[_ParamType]):
    @abstractmethod
    def _backward_prop(
        self,
        *,
        Y_pred: Activations,
        Y_true: np.ndarray,
        As_prev: Activations,
        Zs_prev: PreActivations,
    ) -> tuple[D[DenseParameters], D[PreActivations]]: ...


class SoftmaxOutputLayer(OutputLayerBase[DenseParameters]):
    Parameters = DenseParameters

    def __init__(
        self,
        *,
        neurons: int,
        parameters: DenseParameters | None = None,
        previous_layer: Layer,
    ):
        super().__init__(neurons=neurons, activation_fn=softmax)
        self._previous_layer = previous_layer
        self._parameters = (
            parameters
            if parameters is not None
            else self.Parameters.random(previous_layer.neurons, self._neurons)
        )

    def _backward_prop(
        self,
        *,
        Y_pred: Activations,
        Y_true: np.ndarray,
        As_prev: Activations,
        Zs_prev: PreActivations,
    ) -> tuple[D[DenseParameters], D[PreActivations]]:
        dZ = np.atleast_1d(Y_pred - Y_true)
        return cast(  # TODO: fix-types
            D[DenseParameters],
            self.Parameters.of(
                W=As_prev.T @ dZ,
                B=np.sum(
                    cast(  # TODO: fix-types
                        float | np.ndarray, dZ
                    ),
                    axis=0,
                ),
            ),
        ), dZ @ self._parameters._W.T * self._activation_fn.deriv(Zs_prev)

    def _forward_prop(self, As: Activations) -> tuple[PreActivations, Activations]:
        As = As @ self._parameters._W + self._parameters._B
        return PreActivations(As), self._activation_fn(As)

    @property
    def neurons(self) -> int:
        return self._neurons

    def empty_parameters(self) -> DenseParameters:
        return self.Parameters.empty(
            dim_in=self._previous_layer.neurons, dim_out=self._neurons
        )

    @property
    def parameters(self) -> DenseParameters:
        return self._parameters


class RawOutputLayer(SoftmaxOutputLayer):
    def __init__(
        self,
        *,
        neurons: int,
        previous_layer: Layer,
    ):
        super().__init__(neurons=neurons, previous_layer=previous_layer)
        self._activation_fn = eye


class InputLayer:
    def __init__(self, *, neurons: int):
        # TODO: fix-types
        self._neurons = neurons
        self._activation_fn = eye

    def _forward_prop(self, As: Activations) -> tuple[PreActivations, Activations]:
        return cast(tuple[PreActivations, Activations], (As, As))

    @property
    def neurons(self) -> int:
        return self._neurons


Layer = InputLayer | _LayerBase
NonInputLayer = _LayerBase
