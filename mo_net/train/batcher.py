from collections.abc import Callable, Iterator
from typing import Self

import numpy as np


class Batcher:
    def __init__(
        self,
        *,
        X: np.ndarray,
        Y: np.ndarray,
        batch_size: int,
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.train_set_size = X.shape[0]
        self._transform = transform
        self._shuffle()

        # Fix: Use integer division and handle remainder properly
        num_batches = (self.train_set_size + self.batch_size - 1) // self.batch_size
        self._internal_iterator = zip(
            iter(np.array_split(self.X, num_batches)),
            iter(np.array_split(self.Y, num_batches)),
        )

    def _shuffle(self) -> None:
        permutation = np.random.permutation(self.train_set_size)
        self.X = self.X[permutation]
        self.Y = self.Y[permutation]

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> tuple[np.ndarray, np.ndarray]:
        try:
            X, Y = next(self._internal_iterator)
            if self._transform is not None:
                X = self._transform(X)
            return X, Y
        except StopIteration:
            self._shuffle()
            self._internal_iterator = zip(
                iter(np.array_split(self.X, self.train_set_size / self.batch_size)),
                iter(np.array_split(self.Y, self.train_set_size / self.batch_size)),
            )
            return next(self._internal_iterator)


class IndexBatcher:
    def __init__(self, *, train_set_size: int, batch_size: int):
        self.batch_size = batch_size
        self.train_set_size = train_set_size
        self._internal_iterator: Iterator[np.ndarray] = iter(
            np.array_split(
                np.random.permutation(self.train_set_size),
                self.train_set_size // self.batch_size,
            )
        )

    def _shuffle(self) -> None:
        self._internal_iterator = iter(
            np.array_split(
                np.random.permutation(self.train_set_size),
                self.train_set_size // self.batch_size,
            )
        )

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> np.ndarray:
        try:
            return next(self._internal_iterator)
        except StopIteration:
            self._shuffle()
            self._internal_iterator = iter(
                np.array_split(
                    np.random.permutation(self.train_set_size),
                    self.train_set_size // self.batch_size,
                )
            )
            return next(self._internal_iterator)
