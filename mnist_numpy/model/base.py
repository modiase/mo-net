from __future__ import annotations

from abc import ABC, abstractmethod
from typing import IO, Self, Sequence

import numpy as np


class ModelBase(ABC):
    _W: Sequence[np.ndarray]
    _b: Sequence[np.ndarray]

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
    def _forward_prop(
        self, X: np.ndarray
    ) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]: ...

    @abstractmethod
    def _update_weights(
        self,
        dWs: Sequence[np.ndarray],
        dbs: Sequence[np.ndarray],
        learning_rate: float,
        momentum_parameter: float,
    ) -> None: ...

    @abstractmethod
    def _undo_update(self) -> None: ...

    @abstractmethod
    def _backward_prop(
        self,
        X: np.ndarray,
        Y_true: np.ndarray,
        Z: tuple[np.ndarray, ...],
        A: tuple[np.ndarray, ...],
    ) -> tuple[Sequence[np.ndarray], Sequence[np.ndarray]]: ...
