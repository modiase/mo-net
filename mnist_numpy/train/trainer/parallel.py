import io
import multiprocessing as mp
import operator
from collections.abc import Collection, Iterator
from contextlib import contextmanager
from functools import reduce

from loguru import logger

from mnist_numpy.model.base import ModelT
from mnist_numpy.model.mlp import MultiLayerPerceptron
from mnist_numpy.optimizer.base import NoOptimizer
from mnist_numpy.train.batcher import SharedBatcher
from mnist_numpy.train.trainer.trainer import BasicTrainer, TrainingResult


def worker_process(
    *,
    worker_id: int,
    initial_model_data,
    shared_batcher: SharedBatcher,
    worker_ready_event: mp.Event,
    batch_ready_event: mp.Event,
    stop_event: mp.Event,
) -> None:
    """Worker process that trains on batches and submits updates"""

    buffer = io.BytesIO(initial_model_data)
    model = MultiLayerPerceptron.load(buffer, training=True)

    optimizer = NoOptimizer(
        config=NoOptimizer.Config(learning_rate=0.0)
    )  # learning rate is unused

    worker_ready_event.set()  # type: ignore[attr-defined]

    while not stop_event.is_set():
        try:
            batch_ready_event.wait()  # type: ignore[attr-defined]
            batch_ready_event.clear()  # type: ignore[attr-defined]
            X_batch, Y_batch = shared_batcher.worker_get_batch()

            gradient, _ = optimizer.training_step(
                model=model,
                X_train_batch=X_batch,
                Y_train_batch=Y_batch,
                do_update=False,
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
    def create_worker_processes(
        *,
        worker_id: int,
        initial_model_data: bytes,
        shared_batcher: SharedBatcher,
        worker_ready_event: mp.Event,
        batch_ready_event: mp.Event,
        stop_event: mp.Event,
    ) -> tuple[mp.Process, ...]:
        p = mp.Process(
            target=worker_process,
            kwargs={
                "worker_id": worker_id,
                "initial_model_data": initial_model_data,
                "shared_batcher": shared_batcher,
                "worker_ready_event": worker_ready_event,
                "batch_ready_event": batch_ready_event,
                "stop_event": stop_event,
            },
        )
        p.daemon = True
        p.start()
        return p

    def _before_training_loop(self) -> None:
        self._manager = mp.Manager()
        self._shared_batcher = SharedBatcher(
            X=self._X_train,
            Y=self._Y_train,
            batch_size=self._training_parameters.batch_size,
            batch_queue=self._manager.Queue(),  # type: ignore[attr-defined]
            result_queue=self._manager.Queue(),  # type: ignore[attr-defined]
            update_ready=mp.Event(),
            update_queue=self._manager.Queue(),  # type: ignore[attr-defined]
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
        # Start worker processes
        self._processes = tuple(
            ParallelTrainer.create_worker_processes(
                worker_id=i,
                initial_model_data=initial_model_data,
                shared_batcher=self._shared_batcher,
                worker_ready_event=self._worker_ready_events[i],
                batch_ready_event=self._batch_ready_events[i],
                stop_event=stop_event,
            )
            for i in range(self._training_parameters.workers)
        )
        self._ready_all_workers()

    @contextmanager
    def _training_context(self) -> Iterator[None]:
        try:
            yield

        finally:
            self._stop_event.set()
            for p in self._processes:
                p.join(timeout=1.0)  # type: ignore[attr-defined]
                if p.is_alive():  # type: ignore[attr-defined]
                    p.terminate()  # type: ignore[attr-defined]

        return TrainingResult(
            model_checkpoint_path=self._model_checkpoint_path,
        )

    def _ready_all_workers(self) -> None:
        for event in self._worker_ready_events:
            event.wait()  # type: ignore[attr-defined]
            event.clear()  # type: ignore[attr-defined]

    def _prepare_batches(self) -> None:
        self._shared_batcher.leader_prepare_batches()
        for event in self._batch_ready_events:
            event.set()  # type: ignore[attr-defined]

    def _training_step(self) -> None:
        self._prepare_batches()
        self._ready_all_workers()
        gradient = reduce(operator.add, self._shared_batcher.leader_get_all_results())
        update = self._optimizer.compute_update(gradient)
        self._model.update_parameters(update)
        self._shared_batcher.leader_send_update(update)
