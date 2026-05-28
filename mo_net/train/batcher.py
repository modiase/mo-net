from collections.abc import Iterator
from typing import Self

import jax.numpy as jnp
import jax.random as random
import numpy as np

from mo_net.functions import TransformFn


def _np_permutation_chunks(
    train_set_size: int, batch_size: int, seed: int
) -> Iterator[np.ndarray]:
    """Permute on CPU with numpy then split into batches.

    Done on host because jax.random.permutation + jnp.array_split allocate
    the full permuted index array on-device, which OOMs the GPU once the
    training set crosses a few tens of millions of rows. The per-batch slices
    are small (4096 ints by default) and convert to JAX cheaply at the model
    boundary.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(train_set_size)
    num_batches = max(1, train_set_size // batch_size)
    return iter(np.array_split(perm, num_batches))


def noop_transform(X: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
    del key  # unused
    return X


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
        # Seed the numpy permutation off the JAX key so behaviour is still
        # reproducible from the caller's PRNG.
        self._key = key
        self._internal_iterator = self._new_epoch()

    def _new_epoch(self) -> Iterator[np.ndarray]:
        self._key, subkey = random.split(self._key)
        seed = int(subkey[0])
        return _np_permutation_chunks(self.train_set_size, self.batch_size, seed)

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> np.ndarray:
        try:
            return next(self._internal_iterator)
        except StopIteration:
            self._internal_iterator = self._new_epoch()
            return next(self._internal_iterator)
