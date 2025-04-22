import multiprocessing as mp
from collections.abc import Collection, Iterator, Sequence
from queue import Empty
from typing import Self

import numpy as np

from mnist_numpy.model.mlp import MultiLayerPerceptron
from mnist_numpy.types import EventLike


class Batcher:
    def __init__(self, *, X: np.ndarray, Y: np.ndarray, batch_size: int):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.train_set_size = X.shape[0]
        self._shuffle()
        self._internal_iterator = zip(
            iter(np.array_split(self.X, self.train_set_size / self.batch_size)),
            iter(np.array_split(self.Y, self.train_set_size / self.batch_size)),
        )

    def _shuffle(self) -> None:
        permutation = np.random.permutation(self.train_set_size)
        self.X = self.X[permutation]
        self.Y = self.Y[permutation]

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> tuple[np.ndarray, np.ndarray]:
        try:
            return next(self._internal_iterator)
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


class SharedBatcher:
    """A batcher that can be shared across multiple processes."""

    def __init__(
        self,
        *,
        batch_size: int,
        batch_queue: mp.Queue,
        result_queue: mp.Queue,
        train_set_size: int,
        update_queue: mp.Queue,
        update_ready: EventLike,
        worker_count: int,
    ):
        self.batcher = IndexBatcher(
            train_set_size=train_set_size, batch_size=batch_size // worker_count
        )
        self.batch_queue = batch_queue
        self.result_queue = result_queue
        self.update_ready = update_ready
        self.update_queue = update_queue
        self.worker_count = worker_count

    def _clear_queue(self, queue: mp.Queue) -> None:
        while not queue.empty():
            try:
                queue.get_nowait()
            except Empty:
                break

    def leader_prepare_batches(self) -> None:
        self._clear_queue(self.batch_queue)

        for _ in range(self.worker_count):
            indices = next(self.batcher)
            self.batch_queue.put(indices)

    def worker_get_batch(self, timeout_seconds: float = 1.0) -> Sequence[int]:
        return self.batch_queue.get(timeout=timeout_seconds)

    def worker_put_result(self, update: MultiLayerPerceptron.Gradient) -> None:
        self.result_queue.put(update)

    def leader_get_all_results(self) -> Collection[MultiLayerPerceptron.Gradient]:
        results = []
        while not self.result_queue.empty():
            try:
                update = self.result_queue.get_nowait()
                results.append(update)
            except Empty:
                break
        return results

    def leader_send_update(self, update: MultiLayerPerceptron.Gradient) -> None:
        self._clear_queue(self.update_queue)
        for _ in range(self.worker_count):
            self.update_queue.put(update)
        self.update_ready.set()

    def worker_wait_for_update(self) -> MultiLayerPerceptron.Gradient:
        """Wait for an update from the main process (called by worker)"""
        self.update_ready.wait()  # type: ignore[attr-defined]
        update = self.update_queue.get()
        self.update_ready.clear()
        return update
