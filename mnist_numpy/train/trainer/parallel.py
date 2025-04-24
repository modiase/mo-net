import multiprocessing as mp
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import ContextManager, Final

import numpy as np
from loguru import logger

from mnist_numpy.model.mlp import MultiLayerPerceptron
from mnist_numpy.regulariser.ridge import attach_l2_regulariser
from mnist_numpy.train.batcher import SharedBatcher
from mnist_numpy.train.trainer.trainer import BasicTrainer
from mnist_numpy.types import EventLike, SupportsGradientOperations

_64_BIT_FLOAT_BYTES_SIZE: Final[int] = 8
_PADDING_FACTOR: Final[float] = 1.2


def worker_process(
    *,
    batch_ready_event: EventLike,
    initial_model_data_path: str,
    regulariser_lambda: float,
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

    with open(initial_model_data_path, "rb") as f:
        model = MultiLayerPerceptron.load(f, training=True)

    if regulariser_lambda > 0:
        attach_l2_regulariser(
            lambda_=regulariser_lambda,
            batch_size=shared_batcher.worker_batch_size,
            model=model,
        )

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
            batch_ready_event.wait()
            batch_ready_event.clear()
            indices = shared_batcher.worker_get_batch()
            X_batch = X_train[indices]
            Y_batch = Y_train[indices]

            model.forward_prop(X=X_batch)
            model.backward_prop(Y_true=Y_batch)

            shared_batcher.worker_put_result(
                update=tuple(layer.cache["dP"] for layer in model.grad_layers)
            )
            worker_ready_event.set()

            aggregated_update = shared_batcher.worker_wait_for_update()
            model.populate_caches(aggregated_update)
            model.update_parameters()

        except Exception as e:
            logger.exception(f"Worker {worker_id} error: {e}")
            break


class ParallelTrainer(BasicTrainer):
    """Implements parallel training using multiple processes."""

    @staticmethod
    def create_worker_process(
        *,
        batch_ready_event: EventLike,
        initial_model_data_path: Path,
        regulariser_lambda: float,
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
                "initial_model_data_path": str(initial_model_data_path),
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
                "regulariser_lambda": regulariser_lambda,
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
        self._shared_batcher = SharedBatcher(
            batch_queue=mp.Queue(),
            batch_size=self._training_parameters.batch_size,
            result_queue=mp.Queue(),
            train_set_size=self._X_train.shape[0],
            update_shared_memory=self._update_shared_memory,
            update_ready=mp.Event(),
            worker_count=self._training_parameters.workers,
        )

        self._initial_model_data_path = Path(
            tempfile.NamedTemporaryFile(delete=False).name
        )
        with open(self._initial_model_data_path, "wb") as f:
            self._model.dump(f)

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
                initial_model_data_path=self._initial_model_data_path,
                regulariser_lambda=self._training_parameters.regulariser_lambda,
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
                for p in filter(lambda p: p.is_alive(), self._processes):
                    p.terminate()
                self._X_shared_memory.close()
                self._X_shared_memory.unlink()
                self._Y_shared_memory.close()
                self._Y_shared_memory.unlink()
                self._update_shared_memory.close()
                self._update_shared_memory.unlink()
                self._initial_model_data_path.unlink(missing_ok=True)

        return _training_loop_context()

    def _ready_all_workers(self) -> None:
        for event in self._worker_ready_events:
            event.wait()
            event.clear()

    def _prepare_batches(self) -> None:
        self._shared_batcher.leader_prepare_batches()
        for event in self._batch_ready_events:
            event.set()  # type: ignore[attr-defined]

    def _training_step(
        self,
    ) -> tuple[
        Sequence[SupportsGradientOperations], Sequence[SupportsGradientOperations]
    ]:
        self._prepare_batches()
        self._ready_all_workers()
        gradient = self._shared_batcher.leader_get_aggregated_results()
        self._model.populate_caches(gradient)
        self._optimizer.compute_update()
        self._shared_batcher.leader_send_update(
            update := self._model.get_gradient_caches()
        )
        self._model.update_parameters()
        return gradient, update
