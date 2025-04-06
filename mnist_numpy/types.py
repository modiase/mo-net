from typing import Generic, NewType, Protocol, TypeVar

import numpy as np

_Quantity = TypeVar("_Quantity")


class D(Protocol, Generic[_Quantity]):
    def __add__(self, other: _Quantity) -> _Quantity: ...


Activations = NewType("Activations", np.ndarray)
PreActivations = NewType("PreActivations", np.ndarray)


_X = TypeVar("_X", bound=np.ndarray | float)


class ActivationFn(Protocol):
    def __call__(self, x: _X) -> _X: ...

    def deriv(self, x: _X) -> _X: ...
