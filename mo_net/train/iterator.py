from __future__ import annotations

from typing import Protocol

import jax.numpy as jnp


class TrainSetIterator(Protocol):
    """Protocol for training data iterators."""

    def __iter__(self) -> TrainSetIterator:
        """Return iterator for the training set."""
        ...

    def __next__(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return next batch as (X, Y) tuple."""
        ...


class ArrayTrainSetIterator:
    """Iterator for fixed arrays (backward compatibility)."""

    def __init__(
        self,
        X: jnp.ndarray,
        Y: jnp.ndarray,
        batch_size: int,
        key: jnp.ndarray,
    ):
        self._X = X
        self._Y = Y
        self._batch_size = batch_size
        self._key = key
        self._current_idx = 0
        self._total_samples = len(X)
        self._num_batches = (self._total_samples + batch_size - 1) // batch_size

    def __iter__(self) -> ArrayTrainSetIterator:
        self._current_idx = 0
        return self

    def __next__(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        if self._current_idx >= self._num_batches:
            raise StopIteration

        start_idx = self._current_idx * self._batch_size
        end_idx = min(start_idx + self._batch_size, self._total_samples)

        X_batch = self._X[start_idx:end_idx]
        Y_batch = self._Y[start_idx:end_idx]

        self._current_idx += 1
        return X_batch, Y_batch

    @property
    def num_batches(self) -> int:
        return self._num_batches

    @property
    def total_samples(self) -> int:
        return self._total_samples
