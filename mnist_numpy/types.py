from collections.abc import Callable
from typing import Generic, NewType, Protocol, TypeAlias, TypeVar

import numpy as np


class HasWeights(Protocol):
    _W: np.ndarray


class HasBiases(Protocol):
    _B: np.ndarray


class HasWeightsAndBiases(HasWeights, HasBiases):
    def __init__(self, _W: np.ndarray, _B: np.ndarray): ...


_Quantity = TypeVar("_Quantity")
_ParamType = TypeVar("_ParamType", bound=HasWeightsAndBiases)


class D(Protocol, Generic[_Quantity]):
    def __add__(self, other: _Quantity) -> _Quantity: ...


Activations = NewType("Activations", np.ndarray)


_X = TypeVar("_X", bound=np.ndarray | float)


class ActivationFn(Protocol):
    def __call__(self, x: _X) -> _X: ...

    def deriv(self, x: _X) -> _X: ...

    @property
    def name(self) -> str: ...


class ForwardStepHandler(Protocol):
    def forward(self, x: Activations) -> Activations: ...


class TrainingStepHandler(ForwardStepHandler, Protocol):  # TODO: rationalise protocols
    def pre_backward(self, dZ: D[Activations]) -> D[Activations]: ...

    def post_backward(
        self, dP: D[_ParamType], dZ: D[Activations]
    ) -> tuple[D[_ParamType], D[Activations]]: ...


LossContributor: TypeAlias = Callable[[], float]
