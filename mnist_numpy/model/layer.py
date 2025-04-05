from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Self, TypeVar, cast

import numpy as np

from mnist_numpy.types import ActivationFn, Activations, D, PreActivations

_ParamType = TypeVar("_ParamType")


class LayerBase(ABC, Generic[_ParamType]):
    Parameters: type[_ParamType]
    _parameters: _ParamType

    @abstractmethod
    def forward_prop(self, As: Activations) -> Activations: ...

    @abstractmethod
    def backward_prop(
        self,
        As: Activations,
        Zs: PreActivations,
        dZ: D[PreActivations],
    ) -> tuple[D[_ParamType], D[PreActivations]]: ...

    def _update_parameters(self, params: D[_ParamType]) -> None: ...


@dataclass(kw_only=True, frozen=True)
class DenseParameters:
    _W: np.ndarray
    _B: np.ndarray

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
        return cls(_W=np.random.randn(dim_in, dim_out), _B=np.random.randn(dim_out))


class DenseLayer(LayerBase[DenseParameters]):
    Parameters = DenseParameters
    _parameters: DenseParameters

    def __init__(
        self,
        neurons: int,
        activation_fn: ActivationFn,
    ):
        self._activation_fn = activation_fn
        self._neurons = neurons
        self._parameters = DenseParameters.empty()

    def _init(self, dim_in: int, dim_out: int):
        self._parameters = self.Parameters.random(dim_in, dim_out)

    def _forward_prop(self, As: Activations) -> Activations:
        return self._activation_fn(As @ self._parameters._W + self._parameters._B)

    def _backward_prop(
        self, As: Activations, Zs: PreActivations, dZ: D[PreActivations]
    ) -> tuple[D[DenseParameters], D[PreActivations]]:
        return cast(  # TODO: fix-types
            D[DenseParameters],
            self.Parameters(
                _W=As.T @ dZ,
                _B=np.sum(
                    cast(  # TODO: fix-types
                        float | np.ndarray, dZ
                    )
                ),
            ),
        ), dZ @ self._parameters._W.T * self._activation_fn.deriv(Zs)
