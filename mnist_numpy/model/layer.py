from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Self, TypeVar, cast

import numpy as np

from mnist_numpy.functions import eye
from mnist_numpy.types import ActivationFn, Activations, D, PreActivations

_ParamType = TypeVar("_ParamType")
_PreviousLayerType = TypeVar("_PreviousLayerType", "LayerBase", None)
_NextLayerType = TypeVar("_NextLayerType", "LayerBase", None)


class LayerBase(ABC, Generic[_ParamType, _PreviousLayerType, _NextLayerType]):
    Parameters: type[_ParamType]
    _parameters: _ParamType
    _previous_layer: _PreviousLayerType
    _next_layer: _NextLayerType

    def __init__(
        self,
        neurons: int,
        activation_fn: ActivationFn,
    ):
        self._activation_fn = activation_fn
        self._neurons = neurons

    @abstractmethod
    def _forward_prop(self, As: Activations) -> tuple[PreActivations, Activations]: ...

    @abstractmethod
    def _backward_prop(
        self, As: Activations, Zs: PreActivations, dZ: D[PreActivations]
    ) -> tuple[D[DenseParameters], D[PreActivations]]: ...

    def _update_parameters(self, params: D[DenseParameters]) -> None:
        self._parameters = params + self._parameters

    @abstractmethod
    def _init(
        self, previous_layer: LayerBase | None, next_layer: LayerBase | None
    ) -> None: ...

    @property
    @abstractmethod
    def neurons(self) -> int: ...

    @property
    def parameters(self) -> _ParamType:
        return self._parameters


@dataclass(kw_only=True, frozen=True)
class DenseParameters:
    _W: np.ndarray
    _B: np.ndarray

    def __add__(self, other: Self) -> Self:
        return self.__class__(_W=self._W + other._W, _B=self._B + other._B)

    def __radd__(self, other: Self) -> Self:
        return self.__add__(other)

    def __mul__(self, other: float) -> Self:
        if not isinstance(other, float):
            return NotImplemented
        return self.__class__(_W=other * self._W, _B=other * self._B)

    def __rmul__(self, other: float) -> Self:
        if not isinstance(other, float):
            return NotImplemented
        return self.__mul__(other)

    @classmethod
    def empty(cls) -> Self:
        return cls(_W=np.array([]), _B=np.array([]))

    @classmethod
    def random(cls, dim_in: int, dim_out: int) -> Self:
        return cls(_W=np.random.randn(dim_in, dim_out), _B=np.zeros(dim_out))

    @classmethod
    def eye(cls, dim: int) -> Self:
        return cls(_W=np.eye(dim), _B=np.zeros(dim))

    @classmethod
    def of(cls, W: np.ndarray, B: np.ndarray) -> Self:
        return cls(_W=np.atleast_2d(W), _B=np.atleast_1d(B))


class DenseLayer(LayerBase[DenseParameters, LayerBase, LayerBase]):
    Parameters = DenseParameters
    _parameters: DenseParameters

    def __init__(
        self,
        neurons: int,
        activation_fn: ActivationFn,
    ):
        super().__init__(neurons, activation_fn)
        self._parameters = self.Parameters.empty()

    def _init(self, previous_layer: LayerBase | None, next_layer: LayerBase | None):
        if previous_layer is None or next_layer is None:
            raise ValueError("DenseLayerBase must have a previous and next layer")
        self._parameters = self.Parameters.random(previous_layer.neurons, self._neurons)

    def _forward_prop(self, As: Activations) -> tuple[PreActivations, Activations]:
        preactivations = As @ self._parameters._W + self._parameters._B
        return preactivations, self._activation_fn(preactivations)

    def _backward_prop(
        self, As: Activations, Zs: PreActivations, dZ: D[PreActivations]
    ) -> tuple[D[DenseParameters], D[PreActivations]]:
        return cast(  # TODO: fix-types
            D[DenseParameters],
            self.Parameters.of(
                W=As.T @ dZ,
                B=np.sum(
                    cast(  # TODO: fix-types
                        float | np.ndarray, dZ
                    )
                ),
            ),
        ), dZ @ self._parameters._W.T * self._activation_fn.deriv(Zs)

    @property
    def neurons(self) -> int:
        return self._neurons


class OutputLayer(LayerBase[DenseParameters, LayerBase, None]):
    Parameters = DenseParameters

    def __init__(
        self,
        neurons: int,
        activation_fn: ActivationFn,
    ):
        super().__init__(neurons, activation_fn)
        self._parameters = self.Parameters.empty()

    def _init(self, previous_layer: LayerBase | None, next_layer: LayerBase | None):
        if previous_layer is None:
            raise ValueError("OutputLayer must have a previous layer")
        self._previous_layer = previous_layer
        if next_layer is not None:
            raise ValueError("OutputLayer must have a next layer")
        self._parameters = DenseParameters.random(previous_layer.neurons, self._neurons)

    def _backward_prop(
        self, As: Activations, Zs: PreActivations, dZ: D[PreActivations]
    ) -> tuple[D[DenseParameters], D[PreActivations]]:
        return cast(  # TODO: fix-types
            D[DenseParameters],
            self.Parameters.of(
                W=As.T @ dZ,
                B=np.sum(
                    cast(  # TODO: fix-types
                        float | np.ndarray, dZ
                    )
                ),
            ),
        ), dZ @ self._parameters._W.T * self._activation_fn.deriv(Zs)

    def _forward_prop(self, As: Activations) -> tuple[PreActivations, Activations]:
        As = As @ self._parameters._W + self._parameters._B
        return As, self._activation_fn(As)

    @property
    def neurons(self) -> int:
        return self._neurons


class InputLayer(LayerBase[None, None, LayerBase]):
    def __init__(self, neurons: int):
        # TODO: fix-types
        super().__init__(neurons, eye)  # type: ignore
        self._parameters = None

    def _init(self, previous_layer: LayerBase | None, next_layer: LayerBase | None):
        if previous_layer is not None:
            raise ValueError("InputLayer must not have a previous layer")
        self._previous_layer = None

        if next_layer is None:
            raise ValueError("InputLayer must have a next layer")
        self._next_layer = next_layer

        self._parameters = None

    def _forward_prop(self, As: Activations) -> tuple[PreActivations, Activations]:
        return cast(tuple[PreActivations, Activations], (As, As))

    def _backward_prop(
        self, As: Activations, Zs: PreActivations, dZ: D[PreActivations]
    ) -> tuple[D[DenseParameters], D[PreActivations]]:
        del As, Zs, dZ  # unused
        raise NotImplementedError("InputLayer cannot backpropagate")

    def _update_parameters(self, params: D[DenseParameters]) -> None:
        del params  # unused
        raise NotImplementedError("InputLayer cannot update parameters")

    @property
    def neurons(self) -> int:
        return self._neurons
