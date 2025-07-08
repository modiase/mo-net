from __future__ import annotations

from abc import ABC, abstractmethod
from typing import IO, Self, TypeVar

import jax.numpy as jnp

from mo_net.functions import LossFn
from mo_net.protos import Activations


class ModelBase(ABC):
    @abstractmethod
    def update_parameters(self) -> None: ...

    @abstractmethod
    def predict(self, X: jnp.ndarray) -> jnp.ndarray: ...

    @classmethod
    @abstractmethod
    def get_name(cls) -> str: ...

    @classmethod
    @abstractmethod
    def get_description(cls) -> str: ...

    @abstractmethod
    def dump(self, io: IO[bytes]) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, source: IO[bytes], training: bool = False) -> Self: ...

    @abstractmethod
    def forward_prop(self, X: jnp.ndarray) -> Activations: ...

    @abstractmethod
    def backward_prop(self, *, Y_true: jnp.ndarray) -> None: ...

    @abstractmethod
    def compute_loss(
        self, X: jnp.ndarray, Y_true: jnp.ndarray, loss_fn: LossFn
    ) -> float: ...


ModelT = TypeVar("ModelT", bound=ModelBase)
