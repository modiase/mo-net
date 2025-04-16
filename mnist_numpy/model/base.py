from __future__ import annotations

from abc import ABC, abstractmethod
from typing import IO, Generic, Self, TypeVar

import numpy as np

from mnist_numpy.types import Activations

_GradientT = TypeVar("_GradientT")


class ModelBase(ABC, Generic[_GradientT]):
    Gradient: type[_GradientT]

    @abstractmethod
    def update_parameters(self, update: _GradientT) -> None: ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def get_name(cls) -> str: ...

    @classmethod
    @abstractmethod
    def get_description(cls) -> str: ...

    @classmethod
    @abstractmethod
    def initialize(cls, *dims: int) -> Self: ...

    @abstractmethod
    def dump(self, io: IO[bytes]) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, io: IO[bytes]) -> Self: ...

    @abstractmethod
    def forward_prop(self, X: np.ndarray) -> Activations: ...

    @abstractmethod
    def backward_prop(
        self,
        *,
        Y_true: np.ndarray,
    ) -> _GradientT: ...

    @abstractmethod
    def empty_gradient(self) -> _GradientT: ...

    """
    Returns an empty gradient. Used to initialize optimizers.
    """

    @abstractmethod
    def compute_loss(self, X: np.ndarray, Y_true: np.ndarray) -> float: ...


ModelT = TypeVar("ModelT", bound=ModelBase)
