import contextlib
import multiprocessing as mp
import operator
import pickle
import zlib
from collections.abc import Callable, Iterator, Sequence
from multiprocessing.shared_memory import SharedMemory
from queue import Empty
from typing import Final, Self

import numpy as np

from mnist_numpy.types import EventLike, SupportsGradientOperations, UpdateGradientType

DATA_BYTES_LEN_OFFSET: Final[int] = 4


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


class SharedBatcher:
    """A batcher that can be shared across multiple processes."""

    def __init__(
        self,
        *,
        batch_size: int,
        batch_queue: mp.Queue,
        result_queue: mp.Queue,
        train_set_size: int,
        update_shared_memory: SharedMemory,
        update_ready: EventLike,
        worker_count: int,
    ):
        self.worker_count = worker_count
        self._batch_size = batch_size
        self._train_set_size = train_set_size
        self.batcher = IndexBatcher(
            train_set_size=train_set_size, batch_size=self.worker_batch_size
        )
        self.batch_queue = batch_queue
        self.result_queue = result_queue
        self.update_ready = update_ready
        self._update_shared_memory = update_shared_memory

    @property
    def worker_batch_size(self) -> int:
        return self._batch_size // self.worker_count

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

    def worker_put_result(self, update: tuple[SupportsGradientOperations, ...]) -> None:
        self.result_queue.put(update)

    def leader_get_aggregated_results(
        self,
    ) -> Sequence[SupportsGradientOperations]:
        result_count = 0
        aggregated: tuple[SupportsGradientOperations, ...] | None = None
        while result_count < self.worker_count:
            with contextlib.suppress(Empty):
                update = self.result_queue.get(timeout=1.0)
                if aggregated is None:
                    aggregated = update
                else:
                    aggregated = tuple(map(operator.add, aggregated, update))
                result_count += 1

        if aggregated is None:
            raise RuntimeError("No results received from workers.")
        return aggregated

    def leader_send_update(self, update: Sequence[SupportsGradientOperations]) -> None:
        data_bytes = zlib.compress(pickle.dumps(update), level=zlib.Z_BEST_COMPRESSION)
        data_bytes_len = len(data_bytes)
        self._update_shared_memory.buf[0:DATA_BYTES_LEN_OFFSET] = (
            data_bytes_len.to_bytes(DATA_BYTES_LEN_OFFSET, byteorder="little")
        )
        self._update_shared_memory.buf[
            DATA_BYTES_LEN_OFFSET : DATA_BYTES_LEN_OFFSET + data_bytes_len
        ] = data_bytes
        self.update_ready.set()

    def worker_wait_for_update(self) -> UpdateGradientType | None:
        """Wait for an update from the main process (called by worker)"""
        if not self.update_ready.wait(timeout=10):  # type: ignore[attr-defined]
            return None
        data_bytes_len = int.from_bytes(
            self._update_shared_memory.buf[0:DATA_BYTES_LEN_OFFSET],
            byteorder="little",
        )
        data = bytes(
            self._update_shared_memory.buf[
                DATA_BYTES_LEN_OFFSET : DATA_BYTES_LEN_OFFSET + data_bytes_len
            ]
        )
        update = pickle.loads(zlib.decompress(data))
        self.update_ready.clear()
        return update
