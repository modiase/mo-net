import contextlib
import multiprocessing as mp
import operator
import pickle
import time
import zlib
from collections.abc import Collection, Iterator, Sequence
from contextlib import contextmanager
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from queue import Empty
from typing import ContextManager, Final

import numpy as np
from loguru import logger

from mnist_numpy.augment import affine_transform
from mnist_numpy.model.model import Model
from mnist_numpy.protos import EventLike, SupportsGradientOperations, UpdateGradientType
from mnist_numpy.regulariser.weight_decay import attach_weight_decay_regulariser
from mnist_numpy.train.trainer.trainer import (
    BasicTrainer,
    TrainingResult,
    TrainingSuccessful,
)

_64_BIT_FLOAT_BYTES_SIZE: Final[int] = 8
_PADDING_FACTOR: Final[float] = 1.2
_DATA_BYTES_LEN_OFFSET: Final[int] = 4


class Manager:
    def __init__(
        self,
        *,
        result_queue: mp.Queue,
        update_shared_memory: SharedMemory,
        update_ready: EventLike,
        worker_ready_events: Collection[EventLike],
        worker_count: int,
    ):
        self.worker_count = worker_count
        self.result_queue = result_queue
        self.update_ready = update_ready
        self._update_shared_memory = update_shared_memory
        self._worker_ready_events = worker_ready_events

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
        self._update_shared_memory.buf[0:_DATA_BYTES_LEN_OFFSET] = (
            data_bytes_len.to_bytes(_DATA_BYTES_LEN_OFFSET, byteorder="little")
        )
        self._update_shared_memory.buf[
            _DATA_BYTES_LEN_OFFSET : _DATA_BYTES_LEN_OFFSET + data_bytes_len
        ] = data_bytes
        while not all(event.is_set() for event in self._worker_ready_events):
            time.sleep(0.1)
        self.update_ready.set()

    def worker_wait_for_update(self) -> UpdateGradientType | None:
        """Wait for an update from the main process (called by worker)"""
        if not self.update_ready.wait(timeout=10):
            return None
        data_bytes_len = int.from_bytes(
            self._update_shared_memory.buf[0:_DATA_BYTES_LEN_OFFSET],
            byteorder="little",
        )
        data = bytes(
            self._update_shared_memory.buf[
                _DATA_BYTES_LEN_OFFSET : _DATA_BYTES_LEN_OFFSET + data_bytes_len
            ]
        )
        update = pickle.loads(zlib.decompress(data))
        self.update_ready.clear()
        return update


def worker_process(
    *,
    model_checkpoint_path: str,
    regulariser_lambda: float,
    reload_event: EventLike,
    manager: Manager,
    stop_event: EventLike,
    worker_id: int,
    worker_ready_event: EventLike,
    X_shared_memory_dtype: np.dtype,
    X_shared_memory_name: str,
    X_shared_memory_shape: Sequence[int],
    Y_shared_memory_dtype: np.dtype,
    Y_shared_memory_name: str,
    Y_shared_memory_shape: Sequence[int],
) -> None:
    """Worker process that trains on batches and submits updates"""

    with open(model_checkpoint_path, "rb") as f:
        model = Model.load(f, training=True)

    worker_ready_event.set()
    X_shared_memory = mp.shared_memory.SharedMemory(X_shared_memory_name)
    Y_shared_memory = mp.shared_memory.SharedMemory(Y_shared_memory_name)

    X_train: np.ndarray = np.ndarray(
        shape=X_shared_memory_shape,
        dtype=X_shared_memory_dtype,
        buffer=X_shared_memory.buf,
    )
    Y_train: np.ndarray = np.ndarray(
        shape=Y_shared_memory_shape,
        dtype=Y_shared_memory_dtype,
        buffer=Y_shared_memory.buf,
    )

    while not stop_event.is_set():
        try:
            if reload_event.is_set():
                with open(model_checkpoint_path, "rb") as f:
                    model = Model.load(f, training=True)
                reload_event.clear()
                worker_ready_event.set()

            x_size = y_size = int(
                np.sqrt(X_train.shape[1])
            )  # TODO: Fix: handle non-square images
            X_batch = affine_transform(X=X_train, x_size=x_size, y_size=y_size)

            model.forward_prop(X=X_batch)
            model.backward_prop(Y_true=Y_train)

            manager.worker_put_result(
                update=tuple(layer.cache["dP"] for layer in model.grad_layers)
            )

            worker_ready_event.set()
            for _ in range(10):
                # TODO: Arbitrary number of retries is not ideal.
                # This whole update mechanism needs a good refactor.
                aggregated_update = manager.worker_wait_for_update()
                if aggregated_update is not None or reload_event.is_set():
                    break
            else:
                raise RuntimeError("Failed to receive update from main process.")

            if not reload_event.is_set():
                if aggregated_update is None:
                    raise RuntimeError("Failed to receive update from main process.")
                model.populate_caches(aggregated_update)
                model.update_parameters()

        except Exception as e:
            logger.exception(f"Worker {worker_id} error: {e}")
            break


class ParallelTrainer(BasicTrainer):
    """Implements parallel training using multiple processes."""

    def resume(
        self,
        start_epoch: int,
        model_checkpoint_path: Path,
    ) -> TrainingResult:
        logger.info(f"Resuming training from epoch {start_epoch}.")

        self._start_epoch = start_epoch
        self._model = Model.load(open(model_checkpoint_path, "rb"), training=True)
        self._optimizer.set_model(self._model)
        self._optimizer.restore()
        if self._monitor is not None:
            self._monitor.reset(restore_history=True)
        for event in self._worker_ready_events:
            event.clear()
        for event in self._reload_events:
            event.set()
        for event in self._worker_ready_events:
            event.wait()

        with self._create_training_loop_context():
            self._training_loop()

        return TrainingSuccessful(
            model_checkpoint_path=model_checkpoint_path,
        )

    @staticmethod
    def create_worker_process(
        *,
        model_checkpoint_path: Path,
        regulariser_lambda: float,
        reload_event: EventLike,
        manager: Manager,
        stop_event: EventLike,
        worker_id: int,
        worker_ready_event: EventLike,
        X: np.ndarray,
        X_shared_memory_name: str,
        Y: np.ndarray,
        Y_shared_memory_name: str,
    ) -> mp.Process:
        p = mp.Process(
            target=worker_process,
            kwargs={
                "model_checkpoint_path": str(model_checkpoint_path),
                "regulariser_lambda": regulariser_lambda,
                "reload_event": reload_event,
                "manager": manager,
                "stop_event": stop_event,
                "worker_id": worker_id,
                "worker_ready_event": worker_ready_event,
                "X_shared_memory_dtype": X.dtype,
                "X_shared_memory_name": X_shared_memory_name,
                "X_shared_memory_shape": X.shape,
                "Y_shared_memory_dtype": Y.dtype,
                "Y_shared_memory_name": Y_shared_memory_name,
                "Y_shared_memory_shape": Y.shape,
            },
        )
        p.daemon = True
        p.start()
        return p

    def _before_training_loop(self) -> None:
        if self._training_parameters.regulariser_lambda > 0:
            attach_weight_decay_regulariser(
                lambda_=self._training_parameters.regulariser_lambda,
                batch_size=self._X_train.shape[0],
                model=self._model,
                optimizer=self._optimizer,
            )
        self._X_shared_memory = mp.shared_memory.SharedMemory(
            create=True, size=self._X_train.nbytes
        )
        self._Y_shared_memory = mp.shared_memory.SharedMemory(
            create=True, size=self._Y_train.nbytes
        )
        self._update_shared_memory = mp.shared_memory.SharedMemory(
            create=True,
            size=int(
                self._model.parameter_count * _64_BIT_FLOAT_BYTES_SIZE * _PADDING_FACTOR
            ),
        )
        X_shared: np.ndarray = np.ndarray(
            self._X_train.shape,
            dtype=self._X_train.dtype,
            buffer=self._X_shared_memory.buf,
        )
        Y_shared: np.ndarray = np.ndarray(
            self._Y_train.shape,
            dtype=self._Y_train.dtype,
            buffer=self._Y_shared_memory.buf,
        )
        np.copyto(X_shared, self._X_train)
        np.copyto(Y_shared, self._Y_train)

        stop_event = mp.Event()
        self._worker_ready_events = tuple(
            mp.Event() for _ in range(self._training_parameters.workers)
        )
        self._reload_events = tuple(
            mp.Event() for _ in range(self._training_parameters.workers)
        )
        self._manager: Manager = Manager(
            result_queue=mp.Queue(),
            update_shared_memory=self._update_shared_memory,
            update_ready=mp.Event(),
            worker_count=self._training_parameters.workers,
            worker_ready_events=self._worker_ready_events,
        )

        self._processes = tuple(
            ParallelTrainer.create_worker_process(
                model_checkpoint_path=self._model_checkpoint_path,
                reload_event=self._reload_events[i],
                regulariser_lambda=self._training_parameters.regulariser_lambda,
                manager=self._manager,
                stop_event=stop_event,
                worker_id=i,
                worker_ready_event=self._worker_ready_events[i],
                X=self._X_train,
                X_shared_memory_name=self._X_shared_memory.name,
                Y=self._Y_train,
                Y_shared_memory_name=self._Y_shared_memory.name,
            )
            for i in range(self._training_parameters.workers)
        )
        self._ready_all_workers()

    def _create_training_loop_context(self) -> ContextManager[None]:
        @contextmanager
        def _training_loop_context() -> Iterator[None]:
            try:
                yield
            finally:
                if not self._disable_shutdown:
                    self.shutdown()

        return _training_loop_context()

    def _ready_all_workers(self) -> None:
        for event in self._worker_ready_events:
            event.wait()

    def _training_step(
        self,
    ) -> tuple[
        Sequence[SupportsGradientOperations],
        Sequence[SupportsGradientOperations],
    ]:
        with self._create_training_step_context():
            self._ready_all_workers()
            gradient = self._manager.leader_get_aggregated_results()
            self._model.populate_caches(gradient)
            self._optimizer.compute_update()
            self._manager.leader_send_update(
                update := self._model.get_gradient_caches()
            )
            self._model.update_parameters()
            return gradient, update

    def shutdown(self) -> None:
        for p in self._processes:
            p.terminate()
        self._X_shared_memory.close()
        self._X_shared_memory.unlink()
        self._Y_shared_memory.close()
        self._Y_shared_memory.unlink()
        self._update_shared_memory.close()
        self._update_shared_memory.unlink()
