import io
import multiprocessing as mp
import operator
from collections.abc import Collection, Iterator
from contextlib import contextmanager
from functools import reduce
from typing import ContextManager

import numpy as np
from loguru import logger

from mnist_numpy.model.base import ModelT
from mnist_numpy.model.mlp import MultiLayerPerceptron
from mnist_numpy.optimizer.base import NoOptimizer, OptimizerBase
from mnist_numpy.train.batcher import SharedBatcher
from mnist_numpy.train.trainer.trainer import BasicTrainer
from mnist_numpy.types import EventLike


def worker_process(
    *,
    batch_ready_event: EventLike,
    initial_model_data,
    shared_batcher: SharedBatcher,
    stop_event: EventLike,
    worker_id: int,
    worker_ready_event: EventLike,
    X_shared_memory_dtype: np.dtype,
    X_shared_memory_name: str,
    X_shared_memory_shape: tuple[int, ...],
    Y_shared_memory_dtype: np.dtype,
    Y_shared_memory_name: str,
    Y_shared_memory_shape: tuple[int, ...],
) -> None:
    """Worker process that trains on batches and submits updates"""

    buffer = io.BytesIO(initial_model_data)
    model = MultiLayerPerceptron.load(buffer, training=True)

    optimizer: OptimizerBase = NoOptimizer(
        config=NoOptimizer.Config(learning_rate=0.0)
    )  # learning rate is unused

    worker_ready_event.set()  # type: ignore[attr-defined]
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
            batch_ready_event.wait()  # type: ignore[attr-defined]
            batch_ready_event.clear()  # type: ignore[attr-defined]
            indices = shared_batcher.worker_get_batch()
            X_batch = X_train[indices]
            Y_batch = Y_train[indices]

            gradient = optimizer.training_step(
                model=model,
                X_train_batch=X_batch,
                Y_train_batch=Y_batch,
            )

            shared_batcher.worker_put_result(gradient)
            worker_ready_event.set()

            aggregated_update = shared_batcher.worker_wait_for_update()
            model.update_parameters(aggregated_update)

        except Exception as e:
            logger.exception(f"Worker {worker_id} error: {e}")
            break


class ParallelTrainer(BasicTrainer):
    """Implements parallel training using multiple processes."""

    @staticmethod
    def apply_updates(
        model: ModelT,
        updates: Collection[MultiLayerPerceptron.Gradient],
    ) -> None:
        if not updates:
            return

        for update in updates:
            model.update_parameters(update)

    @staticmethod
    def create_worker_process(
        *,
        batch_ready_event: EventLike,
        initial_model_data: bytes,
        shared_batcher: SharedBatcher,
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
                "batch_ready_event": batch_ready_event,
                "initial_model_data": initial_model_data,
                "shared_batcher": shared_batcher,
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
        self._manager = mp.Manager()
        self._X_shared_memory = mp.shared_memory.SharedMemory(
            create=True, size=self._X_train.nbytes
        )
        self._Y_shared_memory = mp.shared_memory.SharedMemory(
            create=True, size=self._Y_train.nbytes
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
        self._shared_batcher = SharedBatcher(
            batch_queue=self._manager.Queue(),  # type: ignore[arg-type]
            batch_size=self._training_parameters.batch_size,
            result_queue=self._manager.Queue(),  # type: ignore[arg-type]
            train_set_size=self._X_train.shape[0],
            update_queue=self._manager.Queue(),  # type: ignore[arg-type]
            update_ready=mp.Event(),
            worker_count=self._training_parameters.workers,
        )

        buffer = io.BytesIO()
        self._model.dump(buffer)
        buffer.seek(0)
        initial_model_data = buffer.getvalue()

        stop_event = mp.Event()
        self._worker_ready_events = tuple(
            mp.Event() for _ in range(self._training_parameters.workers)
        )
        self._batch_ready_events = tuple(
            mp.Event() for _ in range(self._training_parameters.workers)
        )
        self._processes = tuple(
            ParallelTrainer.create_worker_process(
                batch_ready_event=self._batch_ready_events[i],
                initial_model_data=initial_model_data,
                shared_batcher=self._shared_batcher,
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
                for p in self._processes:
                    p.join(timeout=1.0)  # type: ignore[attr-defined]
                    if p.is_alive():  # type: ignore[attr-defined]
                        p.terminate()  # type: ignore[attr-defined]
                self._X_shared_memory.close()
                self._Y_shared_memory.close()
                self._X_shared_memory.unlink()
                self._Y_shared_memory.unlink()

        return _training_loop_context()

    def _ready_all_workers(self) -> None:
        for event in self._worker_ready_events:
            event.wait()  # type: ignore[attr-defined]
            event.clear()  # type: ignore[attr-defined]

    def _prepare_batches(self) -> None:
        self._shared_batcher.leader_prepare_batches()
        for event in self._batch_ready_events:
            event.set()  # type: ignore[attr-defined]

    def _training_step(
        self,
    ) -> tuple[MultiLayerPerceptron.Gradient, MultiLayerPerceptron.Gradient]:
        self._prepare_batches()
        self._ready_all_workers()
        gradient = reduce(operator.add, self._shared_batcher.leader_get_all_results())
        update = self._optimizer.compute_update(gradient)
        self._model.update_parameters(update)
        self._shared_batcher.leader_send_update(update)
        return gradient, update
