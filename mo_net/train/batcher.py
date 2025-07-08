from collections.abc import Iterator
from typing import Self

import jax.numpy as jnp
import jax.random as random

from mo_net.functions import TransformFn


def noop_transform(X: jnp.ndarray, key: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    return X, key


class Batcher:
    def __init__(
        self,
        *,
        X: jnp.ndarray,
        Y: jnp.ndarray,
        batch_size: int,
        key: jnp.ndarray,
        transform: TransformFn = noop_transform,
    ):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.train_set_size = X.shape[0]
        self._key = key
        self._transform = transform if transform is not None else noop_transform
        self._shuffle()

        num_batches = (self.train_set_size + self.batch_size - 1) // self.batch_size
        self._internal_iterator = zip(
            iter(jnp.array_split(self.X, num_batches)),
            iter(jnp.array_split(self.Y, num_batches)),
            strict=True,
        )

    def _shuffle(self) -> None:
        self._key, subkey = random.split(self._key)
        permutation = random.permutation(subkey, self.train_set_size)
        self.X = self.X[permutation]
        self.Y = self.Y[permutation]

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        try:
            X, Y = next(self._internal_iterator)
            X, self._key = self._transform(X, self._key)
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
    def __init__(self, *, train_set_size: int, batch_size: int, key: jnp.ndarray):
        self.batch_size = batch_size
        self.train_set_size = train_set_size
        self._key = key
        self._internal_iterator: Iterator[jnp.ndarray] = iter(
            jnp.array_split(
                random.permutation(key, self.train_set_size),
                self.train_set_size // self.batch_size,
            )
        )

    def _shuffle(self) -> None:
        self._key, subkey = random.split(self._key)
        self._internal_iterator = iter(
            jnp.array_split(
                random.permutation(subkey, self.train_set_size),
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
            self._key, subkey = random.split(self._key)
            self._internal_iterator = iter(
                jnp.array_split(
                    random.permutation(subkey, self.train_set_size),
                    self.train_set_size // self.batch_size,
                )
            )
            return next(self._internal_iterator)
