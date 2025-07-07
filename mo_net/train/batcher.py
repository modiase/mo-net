from collections.abc import Callable, Iterator
from typing import Self

import jax.numpy as jnp
import jax.random as random

from mo_net.functions import identity


class Batcher:
    def __init__(
        self,
        *,
        X: jnp.ndarray,
        Y: jnp.ndarray,
        batch_size: int,
        transform: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    ):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.train_set_size = X.shape[0]
        self._transform = transform if transform is not None else identity
        self._shuffle()

        num_batches = (self.train_set_size + self.batch_size - 1) // self.batch_size
        self._internal_iterator = zip(
            iter(jnp.array_split(self.X, num_batches)),
            iter(jnp.array_split(self.Y, num_batches)),
            strict=True,
        )

    def _shuffle(self) -> None:
        key = random.PRNGKey(0)  # You might want to make this configurable
        permutation = random.permutation(key, self.train_set_size)
        self.X = self.X[permutation]
        self.Y = self.Y[permutation]

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        try:
            X, Y = next(self._internal_iterator)
            X = self._transform(X)
            return X, Y
        except StopIteration:
            self._shuffle()
            self._internal_iterator = zip(
                iter(jnp.array_split(self.X, self.train_set_size / self.batch_size)),
                iter(jnp.array_split(self.Y, self.train_set_size / self.batch_size)),
                strict=True,
            )
            return next(self._internal_iterator)


class IndexBatcher:
    def __init__(self, *, train_set_size: int, batch_size: int):
        self.batch_size = batch_size
        self.train_set_size = train_set_size
        key = random.PRNGKey(0)  # You might want to make this configurable
        self._internal_iterator: Iterator[jnp.ndarray] = iter(
            jnp.array_split(
                random.permutation(key, self.train_set_size),
                self.train_set_size // self.batch_size,
            )
        )

    def _shuffle(self) -> None:
        key = random.PRNGKey(0)  # You might want to make this configurable
        self._internal_iterator = iter(
            jnp.array_split(
                random.permutation(key, self.train_set_size),
                self.train_set_size // self.batch_size,
            )
        )

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> jnp.ndarray:
        try:
            return next(self._internal_iterator)
        except StopIteration:
            self._shuffle()
            self._internal_iterator = iter(
                jnp.array_split(
                    random.permutation(random.PRNGKey(0), self.train_set_size),
                    self.train_set_size // self.batch_size,
                )
            )
            return next(self._internal_iterator)
