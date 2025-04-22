from collections.abc import Callable
from typing import Generic, NewType, Protocol, TypeAlias, TypeVar

import numpy as np

Activations = NewType("Activations", np.ndarray)

_X = TypeVar("_X", bound=np.ndarray | float)


class ActivationFn(Protocol):
    def __call__(self, x: _X) -> _X: ...

    def deriv(self, x: _X) -> _X: ...

    @property
    def name(self) -> str: ...


_Quantity = TypeVar("_Quantity")


class D(Protocol, Generic[_Quantity]):
    def __add__(self, other: _Quantity) -> _Quantity: ...


class EventLike(Protocol):
    def clear(self) -> None: ...
    def is_set(self) -> bool: ...
    def set(self) -> None: ...
    def wait(self, timeout: float = ...) -> bool: ...


class HasBiases(Protocol):
    _B: np.ndarray


class HasWeights(Protocol):
    _W: np.ndarray


class HasWeightsAndBiases(HasWeights, HasBiases):
    def __init__(self, _W: np.ndarray, _B: np.ndarray): ...


LossContributor: TypeAlias = Callable[[], float]


_ParamType = TypeVar("_ParamType", bound=HasWeightsAndBiases)


class TrainingStepHandler:
    def pre_forward(self, As_prev: Activations) -> Activations:
        return As_prev

    def post_forward(self, As: Activations) -> Activations:
        return As

    def pre_backward(self, dZ: D[Activations]) -> D[Activations]:
        return dZ

    def post_backward(
        self, dP: D[_ParamType], dZ: D[Activations]
    ) -> tuple[D[_ParamType], D[Activations]]:
        return dP, dZ
