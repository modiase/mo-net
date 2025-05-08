from __future__ import annotations

from abc import ABC, abstractmethod
from typing import IO, Self, TypeVar

import numpy as np
from protos import Activations


class ModelBase(ABC):
    @abstractmethod
    def update_parameters(self) -> None: ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def get_name(cls) -> str: ...

    @classmethod
    @abstractmethod
    def get_description(cls) -> str: ...

    @abstractmethod
    def dump(self, io: IO[bytes]) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, io: IO[bytes]) -> Self: ...

    @abstractmethod
    def forward_prop(self, X: np.ndarray) -> Activations: ...

    @abstractmethod
    def backward_prop(self, *, Y_true: np.ndarray) -> None: ...

    @abstractmethod
    def compute_loss(self, X: np.ndarray, Y_true: np.ndarray) -> float: ...


ModelT = TypeVar("ModelT", bound=ModelBase)
