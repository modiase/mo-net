from __future__ import annotations

from abc import ABC, abstractmethod
from typing import IO, Generic, Self, TypeVar

import numpy as np

_WeightsT = TypeVar("_WeightsT")
_GradientT = TypeVar("_GradientT")


class ModelBase(ABC, Generic[_WeightsT, _GradientT]):
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
    def update_weights(
        self,
        weights: _WeightsT,
    ) -> None: ...

    @abstractmethod
    def _backward_prop(
        self,
        X: np.ndarray,
        Y_true: np.ndarray,
        Z: tuple[np.ndarray, ...],
        A: tuple[np.ndarray, ...],
    ) -> _GradientT: ...

    @abstractmethod
    def empty_weights(self) -> _WeightsT: ...
