import functools
import multiprocessing as mp
import pickle
import struct
import sys
import time
import typing
from collections.abc import Buffer, Callable, Iterable, Iterator, Sequence
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Barrier
from pathlib import Path
from typing import IO as IO_Type
from typing import Final, ParamSpec, TypeVar, cast

import jax.numpy as jnp
import jax.random as random
from loguru import logger

from mo_net.log import log_time
from mo_net.model.layer.base import ParametrisedHidden, _global_registry
from mo_net.model.model import Model
from mo_net.protos import EventLike, SupportsGradientOperations, UpdateGradientType
from mo_net.regulariser.weight_decay import WeightDecayRegulariser
from mo_net.train.trainer.trainer import (
    BasicTrainer,
    TrainingResult,
    TrainingSuccessful,
)

_DATA_BYTES_LEN_OFFSET: Final[int] = 4


class SharedMemoryManager:
    """Manages shared memory for efficient gradient aggregation using barrier synchronization"""

    class IO(IO_Type[bytes]):
        """Wrapper that makes shared memory buffer look like IO[bytes] for reading and writing"""

        def __init__(self, shared_memory_buffer: memoryview, max_size: int):
            self._buffer = shared_memory_buffer
            self._position = 0
            self._max_size = max_size

        def read(self, size: int = -1) -> bytes:
            """Read data from shared memory buffer"""
            if size == -1:
                size = self._max_size - self._position

            if self._position >= self._max_size:
                return b""

            end_pos = min(self._position + size, self._max_size)
            data = bytes(self._buffer[self._position : end_pos])
            self._position = end_pos
            return data

        def write(self, s: bytes | Buffer, /) -> int:
            """Write data directly to shared memory buffer"""
            if not isinstance(s, bytes):
                # memoryview
                _s = s.tobytes()  # type: ignore[attr-defined]
            else:
                _s = s

            data_len = len(_s)
            if self._position + data_len > self._max_size:
                available_space = self._max_size - self._position
                if available_space <= 0:
                    return 0
                _s = _s[:available_space]
                data_len = len(_s)

            self._buffer[self._position : self._position + data_len] = _s
            self._position += data_len
            return data_len

        def writelines(self, lines: Iterable[bytes | Buffer]) -> None:
            """Write a list of byte strings to the buffer"""
            for line in lines:
                self.write(line)

        def tell(self) -> int:
            """Return current position"""
            return self._position

        def seek(self, pos: int, whence: int = 0, /) -> int:
            """Seek to position"""
            self._position = max(0, min(pos, self._max_size))
            return self._position

        def close(self) -> None:
            """No-op for compatibility"""
            pass

        def __enter__(self) -> "SharedMemoryManager.IO":
            """Context manager entry"""
            return self

        def __exit__(self, type, value, traceback) -> None:
            """Context manager exit"""
            self.close()

        # Additional IO[bytes] protocol methods for completeness
        @property
        def mode(self) -> str:
            return "rb+"

        @property
        def name(self) -> str:
            return "<shared_memory>"

        @property
        def closed(self) -> bool:
            return False

        def fileno(self) -> int:
            raise OSError("SharedMemory buffer has no file descriptor")

        def flush(self) -> None:
            """No-op for shared memory"""
            pass

        def isatty(self) -> bool:
            return False

        def readable(self) -> bool:
            return True

        def readline(self, limit: int = -1) -> bytes:
            """Read until newline or limit"""
            start_pos = self._position
            if limit == -1:
                limit = self._max_size - start_pos

            end_pos = min(start_pos + limit, self._max_size)

            for i in range(start_pos, end_pos):
                if self._buffer[i] == ord(b"\n"):
                    self._position = i + 1
                    return bytes(self._buffer[start_pos : i + 1])

            self._position = end_pos
            return bytes(self._buffer[start_pos:end_pos])

        def readlines(self, hint: int = -1) -> list[bytes]:
            """Read lines from buffer"""
            lines = []
            while self._position < self._max_size:
                line = self.readline()
                if not line:
                    break
                lines.append(line)
                if hint > 0 and sum(len(l) for l in lines) >= hint:
                    break
            return lines

        def seekable(self) -> bool:
            return True

        def truncate(self, size: typing.Optional[int] = None) -> int:
            """Truncate buffer (no-op for shared memory)"""
            if size is None:
                size = self._position
            return size

        def writable(self) -> bool:
            return True

        def __iter__(self) -> Iterator[bytes]:
            """Iterate over the buffer"""
            return iter(self.readlines())

        def __next__(self) -> bytes:
            """Next item in the buffer"""
            return self.readline()

    def __init__(
        self,
        *,
        worker_count: int,
        gradient_n_bytes: int,
        update_shared_memory: SharedMemory,
        gradient_barrier: Barrier,
        update_barrier: Barrier,
    ):
        logger.trace(
            f"Initializing SharedMemoryManager: {worker_count} workers, {gradient_n_bytes} gradient size"
        )

        self.worker_count = worker_count
        self._gradient_size_bytes = int(gradient_n_bytes * 1.2)  # 20% padding
        self._update_shared_memory = update_shared_memory
        self._gradient_barrier = gradient_barrier
        self._update_barrier = update_barrier

        self._gradient_shared_memories = []
        total_gradient_memory = 0

        for i in range(worker_count):
            gradient_memory = mp.shared_memory.SharedMemory(
                create=True, size=self._gradient_size_bytes
            )
            self._gradient_shared_memories.append(gradient_memory)

            total_gradient_memory += self._gradient_size_bytes
            logger.trace(
                f"Created gradient shared memory for worker {i}: {self._gradient_size_bytes} bytes ({gradient_memory.name})"
            )

        logger.trace(
            f"SharedMemoryManager initialized: {total_gradient_memory} bytes total gradient memory, "
            f"{self._update_shared_memory.size} bytes update memory"
        )

    def worker_put_result(
        self, worker_id: int, grad_layers: Sequence[ParametrisedHidden]
    ) -> None:
        """Worker writes layer parameters to shared memory using ParametrisedHidden interface"""
        with log_time(
            f"Worker {worker_id} parameter submission: {{time_taken:.4f}}s total"
        ):
            logger.trace(
                f"Worker {worker_id} starting parameter write to shared memory"
            )

            writer = self.IO(
                self._gradient_shared_memories[worker_id].buf, self._gradient_size_bytes
            )

            for i, layer in enumerate(grad_layers):
                with log_time(
                    f"Worker {worker_id} serialized layer {i} ({type(layer).__name__}): {{time_taken:.4f}}s"
                ):
                    layer.write_serialized_parameters(writer)

            bytes_written = writer.tell()

            if bytes_written < self._gradient_size_bytes:
                remaining_space = self._gradient_size_bytes - bytes_written
                for i in range(remaining_space):
                    self._gradient_shared_memories[worker_id].buf[bytes_written + i] = 0
                logger.trace(
                    f"Worker {worker_id} wrote {bytes_written} bytes, zero-padded {remaining_space} bytes"
                )

            logger.trace(f"Worker {worker_id} now waiting at gradient barrier")

            with log_time(
                f"Worker {worker_id} gradient barrier wait: {{time_taken:.4f}}s"
            ):
                self._gradient_barrier.wait()

    def leader_get_aggregated_results(self, model: Model) -> None:
        """Leader waits at barrier then aggregates gradients from all workers' shared memory"""
        with log_time("Leader gradient processing: {time_taken:.4f}s total"):
            logger.trace("Leader waiting at gradient barrier for all workers")

            with log_time("Leader gradient barrier wait: {time_taken:.4f}s"):
                self._gradient_barrier.wait()

            logger.trace("Leader starting gradient aggregation")

            with log_time("Leader gradient aggregation: {time_taken:.4f}s"):
                for worker_id in range(self.worker_count):
                    with log_time(
                        f"Leader processing worker {worker_id}: {{time_taken:.4f}}s"
                    ):
                        logger.trace(f"Leader processing worker {worker_id}")

                        reader = self.IO(
                            self._gradient_shared_memories[worker_id].buf,
                            self._gradient_size_bytes,
                        )

                        layers_processed = 0
                        while reader.tell() < self._gradient_size_bytes:
                            try:
                                layer_id = ParametrisedHidden.get_layer_id(
                                    reader, peek=True
                                )

                                if not layer_id:
                                    logger.trace(
                                        f"Leader hit padding for worker {worker_id} (empty layer_id) at position {reader.tell()}"
                                    )
                                    break

                                logger.trace(
                                    f"Leader found layer {layer_id} from worker {worker_id}"
                                )

                                layer = model.get_parametrised_layer(layer_id)
                                layer.read_serialized_parameters(reader)

                                layers_processed += 1
                                logger.trace(
                                    f"Leader deserialized layer {layer_id} from worker {worker_id}"
                                )

                            except (
                                struct.error,
                                UnicodeDecodeError,
                                ValueError,
                                IndexError,
                            ):
                                logger.trace(
                                    f"Leader hit malformed data/end for worker {worker_id} at position {reader.tell()}"
                                )
                                break

                        logger.trace(
                            f"Leader processed {layers_processed} layers from worker {worker_id}"
                        )

                for layer in model.grad_layers:
                    if layer.cache["dP"] is not None:
                        layer.cache["dP"] = layer.cache["dP"] / self.worker_count

    def leader_send_update(self, update: Sequence[SupportsGradientOperations]) -> None:
        """Send parameter updates to workers via shared memory and barrier synchronization"""
        with log_time("Leader update broadcast: {time_taken:.4f}s total"):
            logger.trace("Leader starting parameter update broadcast")

            with log_time("Leader update serialization: {time_taken:.4f}s"):
                data_bytes = pickle.dumps(update)
                data_bytes_len = len(data_bytes)

            if (
                _DATA_BYTES_LEN_OFFSET + data_bytes_len
                > self._update_shared_memory.size
            ):
                raise RuntimeError(
                    f"Update data too large for shared memory: {_DATA_BYTES_LEN_OFFSET + data_bytes_len} bytes > {self._update_shared_memory.size} bytes"
                )

            writer = self.IO(
                self._update_shared_memory.buf, self._update_shared_memory.size
            )
            writer.write(
                data_bytes_len.to_bytes(_DATA_BYTES_LEN_OFFSET, byteorder="little")
            )

            if writer.write(data_bytes) != data_bytes_len:
                raise RuntimeError(
                    f"Failed to write complete update data: wrote {writer.tell() - _DATA_BYTES_LEN_OFFSET} of {data_bytes_len} bytes"
                )

            logger.trace(
                f"Leader serialized and wrote {data_bytes_len} bytes of updates, now waiting at update barrier"
            )

            with log_time("Leader update barrier wait: {time_taken:.4f}s"):
                self._update_barrier.wait()

    def worker_wait_for_update(self) -> UpdateGradientType | None:
        """Worker waits at barrier for parameter updates"""
        with log_time("Worker parameter update receive: {time_taken:.4f}s total"):
            logger.trace("Worker waiting at update barrier for parameter updates")

            with log_time("Worker update barrier wait: {time_taken:.4f}s"):
                self._update_barrier.wait()

            logger.trace("Worker reading updates")

            data_bytes_len = int.from_bytes(
                self._update_shared_memory.buf[0:_DATA_BYTES_LEN_OFFSET],
                byteorder="little",
            )

            if (
                data_bytes_len <= 0
                or data_bytes_len
                > self._update_shared_memory.size - _DATA_BYTES_LEN_OFFSET
            ):
                raise RuntimeError(
                    f"Invalid data length received: {data_bytes_len} bytes"
                )

            data = bytes(
                self._update_shared_memory.buf[
                    _DATA_BYTES_LEN_OFFSET : _DATA_BYTES_LEN_OFFSET + data_bytes_len
                ]
            )

            if len(data) != data_bytes_len:
                raise RuntimeError(
                    f"Truncated data received: got {len(data)} bytes, expected {data_bytes_len} bytes"
                )

            with log_time("Worker update deserialization: {time_taken:.4f}s"):
                try:
                    result = pickle.loads(data)
                except Exception as e:
                    raise RuntimeError(f"Failed to unpickle data: {e}") from e

            logger.trace(f"Worker received parameter update: {data_bytes_len} bytes")
            return result

    def cleanup(self):
        """Clean up shared memory resources"""
        logger.trace(
            f"Cleaning up SharedMemoryManager: {len(self._gradient_shared_memories)} gradient memories"
        )

        cleanup_errors = []
        for i, memory in enumerate(self._gradient_shared_memories):
            try:
                memory.close()
                memory.unlink()
                logger.trace(f"Cleaned up gradient memory {i}: {memory.name}")
            except Exception as e:
                cleanup_errors.append(f"Worker {i} memory cleanup error: {e}")
                logger.warning(f"Failed to cleanup gradient memory {i}: {e}")

        if cleanup_errors:
            logger.warning(
                f"SharedMemoryManager cleanup completed with {len(cleanup_errors)} errors"
            )
        else:
            logger.trace("SharedMemoryManager cleanup completed successfully")


P = ParamSpec("P")
R = TypeVar("R")


def worker_decorator(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator for worker processes"""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        worker_id = kwargs.pop("worker_id")
        log_level = cast(str, kwargs.pop("log_level"))
        logger.configure(extra={"worker_id": worker_id})
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | Worker {extra[worker_id]} | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=log_level,
        )
        logger.trace(
            f"Worker {worker_id} decorator initialized with log level {log_level}"
        )
        return func(*args, **kwargs, worker_id=worker_id, log_level=log_level)  # type: ignore[arg-type]

    return wrapper


@worker_decorator
def worker_process(
    *,
    model_checkpoint_path: str,
    regulariser_lambda: float,
    reload_event: EventLike,
    shared_memory_manager: SharedMemoryManager,
    stop_event: EventLike,
    worker_id: int,
    log_level: str,
    batch_size: int,
    worker_ready_event: EventLike,
    X_shared_memory_dtype: jnp.dtype,
    X_shared_memory_name: str,
    X_shared_memory_shape: Sequence[int],
    Y_shared_memory_dtype: jnp.dtype,
    Y_shared_memory_name: str,
    Y_shared_memory_shape: Sequence[int],
) -> None:
    """Worker process that trains on batches and submits updates via barrier synchronization"""
    del log_level  # unused
    del regulariser_lambda  # unused
    _global_registry.reset_for_new_process()

    with log_time(
        f"Worker {worker_id} process startup: {{time_taken:.4f}}s total"
    ) as process_start:
        logger.trace(
            f"Worker {worker_id} process starting with PID {mp.current_process().pid}"
        )

        with log_time(f"Worker {worker_id} model loading: {{time_taken:.4f}}s"):
            with open(model_checkpoint_path, "rb") as f:
                model = Model.load(f, training=True)

        worker_ready_event.set()
        logger.trace(f"Worker {worker_id} signaled ready, connecting to shared memory")

        with log_time(f"Worker {worker_id} shared memory setup: {{time_taken:.4f}}s"):
            X_shared_memory = mp.shared_memory.SharedMemory(X_shared_memory_name)
            Y_shared_memory = mp.shared_memory.SharedMemory(Y_shared_memory_name)

            X_train: jnp.ndarray = jnp.ndarray(
                shape=X_shared_memory_shape,
                dtype=X_shared_memory_dtype,
                buffer=X_shared_memory.buf,
            )
            Y_train: jnp.ndarray = jnp.ndarray(
                shape=Y_shared_memory_shape,
                dtype=Y_shared_memory_dtype,
                buffer=Y_shared_memory.buf,
            )

        logger.trace(
            f"Worker {worker_id} initialization completed. "
            f"X_train: {X_train.shape} {X_train.dtype}, Y_train: {Y_train.shape} {Y_train.dtype}"
        )

    iteration_count = 0.0
    total_forward_time = 0.0
    total_backward_time = 0.0
    total_gradient_time = 0.0
    total_update_time = 0.0

    while not stop_event.is_set():
        try:
            with log_time(
                f"Worker {worker_id} iteration {iteration_count + 1}: {{time_taken:.4f}}s total"
            ):
                iteration_count += 1
                logger.trace(f"Worker {worker_id} starting iteration {iteration_count}")

                if reload_event.is_set():
                    with log_time(
                        f"Worker {worker_id} model reload: {{time_taken:.4f}}s"
                    ):
                        with open(model_checkpoint_path, "rb") as f:
                            model = Model.load(f, training=True)
                        reload_event.clear()
                        worker_ready_event.set()

                indices = random.choice(
                    random.PRNGKey(worker_id),
                    X_train.shape[0],
                    shape=(batch_size,),
                    replace=False,
                )
                X_batch = X_train[indices]
                Y_batch = Y_train[indices]

                with log_time(
                    f"Worker {worker_id} forward pass: {{time_taken:.4f}}s"
                ) as forward_start:
                    model.forward_prop(X=X_batch)
                forward_time = time.perf_counter() - forward_start
                total_forward_time += forward_time

                with log_time(
                    f"Worker {worker_id} backward pass: {{time_taken:.4f}}s"
                ) as backward_start:
                    model.backward_prop(Y_true=Y_batch)
                backward_time = time.perf_counter() - backward_start
                total_backward_time += backward_time

                with log_time(
                    f"Worker {worker_id} gradient submission: {{time_taken:.4f}}s"
                ) as gradient_start:
                    shared_memory_manager.worker_put_result(
                        worker_id, model.grad_layers
                    )
                gradient_time = time.perf_counter() - gradient_start
                total_gradient_time += gradient_time

                if not reload_event.is_set():
                    with log_time(
                        f"Worker {worker_id} parameter update: {{time_taken:.4f}}s"
                    ) as update_start:
                        aggregated_update = (
                            shared_memory_manager.worker_wait_for_update()
                        )
                        if aggregated_update is None:
                            raise RuntimeError("No update received from leader")

                        model.populate_caches(aggregated_update)
                        model.update_parameters()
                    update_time = time.perf_counter() - update_start
                    total_update_time += update_time

                    if iteration_count % 10 == 0:
                        avg_forward = total_forward_time / iteration_count
                        avg_backward = total_backward_time / iteration_count
                        avg_gradient = total_gradient_time / iteration_count
                        avg_update = total_update_time / iteration_count
                        logger.trace(
                            f"Worker {worker_id} performance summary after {iteration_count} iterations: "
                            f"avg forward {avg_forward:.4f}s, backward {avg_backward:.4f}s, "
                            f"gradient {avg_gradient:.4f}s, update {avg_update:.4f}s"
                        )

        except Exception as e:
            logger.exception(
                f"Worker {worker_id} error in iteration {iteration_count}: {e}"
            )
            break

    with log_time(f"Worker {worker_id} shutdown cleanup: {{time_taken:.4f}}s"):
        try:
            X_shared_memory.close()
            Y_shared_memory.close()
        except Exception as e:
            logger.warning(f"Worker {worker_id} shared memory cleanup error: {e}")

    total_runtime = time.perf_counter() - process_start
    logger.trace(
        f"Worker {worker_id} shutting down after {iteration_count} iterations, "
        f"{total_runtime:.4f}s total runtime"
    )


class ParallelTrainer(BasicTrainer):
    """Implements parallel training using multiple processes with barrier synchronization."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        logger.trace("ParallelTrainer initializing")
        self._X_shared_memory: SharedMemory | None = None
        self._Y_shared_memory: SharedMemory | None = None
        self._update_shared_memory: SharedMemory | None = None
        self._processes: tuple[mp.Process, ...] = ()
        self._shared_memory_manager: SharedMemoryManager | None = None
        self._gradient_shapes: list = []
        self._gradient_barrier: Barrier | None = None
        self._update_barrier: Barrier | None = None
        logger.trace("ParallelTrainer initialization completed")

    def resume(
        self,
        *,
        start_epoch: int,
        model_checkpoint_path: Path,
    ) -> TrainingResult:
        logger.info(f"Resuming training from epoch {start_epoch}.")

        with log_time("Training resume setup: {time_taken:.4f}s"):
            self._start_epoch = start_epoch
            self._model = Model.load(open(model_checkpoint_path, "rb"), training=True)
            self._optimiser.set_model(self._model)
            self._optimiser.restore()
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
        X: jnp.ndarray,
        X_shared_memory_name: str,
        Y: jnp.ndarray,
        Y_shared_memory_name: str,
        batch_size: int,
        log_level: str,
        model_checkpoint_path: Path,
        regulariser_lambda: float,
        reload_event: EventLike,
        shared_memory_manager: SharedMemoryManager,
        stop_event: EventLike,
        worker_id: int,
        worker_ready_event: EventLike,
    ) -> mp.Process:
        logger.trace(
            f"Creating worker process {worker_id} with X shape {X.shape}, Y shape {Y.shape}"
        )

        p = mp.Process(
            target=worker_process,
            kwargs={
                "model_checkpoint_path": str(model_checkpoint_path),
                "regulariser_lambda": regulariser_lambda,
                "reload_event": reload_event,
                "shared_memory_manager": shared_memory_manager,
                "stop_event": stop_event,
                "worker_id": worker_id,
                "log_level": log_level,
                "batch_size": batch_size,
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
        logger.trace(f"Worker process {worker_id} started with PID {p.pid}")
        return p

    def _before_training_loop(self) -> None:
        with log_time("Parallel training setup: {time_taken:.4f}s total"):
            logger.trace("Starting parallel training setup")

            if self._training_parameters.regulariser_lambda > 0:
                with log_time(
                    f"Weight decay regulariser attachment (λ={self._training_parameters.regulariser_lambda}): {{time_taken:.4f}}s"
                ):
                    WeightDecayRegulariser.attach(
                        lambda_=self._training_parameters.regulariser_lambda,
                        batch_size=self._X_train.shape[0],
                        model=self._model,
                        optimiser=self._optimiser,
                    )

            with log_time("Shared memory creation: {time_taken:.4f}s"):
                self._X_shared_memory = mp.shared_memory.SharedMemory(
                    create=True, size=self._X_train.nbytes
                )
                self._Y_shared_memory = mp.shared_memory.SharedMemory(
                    create=True, size=self._Y_train.nbytes
                )
                self._update_shared_memory = mp.shared_memory.SharedMemory(
                    create=True,
                    size=int(
                        self._model.parameter_n_bytes * 1.2
                    ),  # extra space for headroom
                )

                total_memory_bytes = (
                    self._X_train.nbytes
                    + self._Y_train.nbytes
                    + self._model.parameter_n_bytes
                )
                logger.trace(
                    f"Created shared memories: X {self._X_train.nbytes} bytes, Y {self._Y_train.nbytes} bytes, "
                    f"updates {self._model.parameter_n_bytes} bytes = {total_memory_bytes} bytes total"
                )

            with log_time("Training data copy to shared memory: {time_taken:.4f}s"):
                X_shared: jnp.ndarray = jnp.ndarray(
                    self._X_train.shape,
                    dtype=self._X_train.dtype,
                    buffer=self._X_shared_memory.buf,
                )
                Y_shared: jnp.ndarray = jnp.ndarray(
                    self._Y_train.shape,
                    dtype=self._Y_train.dtype,
                    buffer=self._Y_shared_memory.buf,
                )
                X_shared[:] = self._X_train
                Y_shared[:] = self._Y_train

            barrier_parties = self._training_parameters.workers + 1
            self._gradient_barrier = mp.Barrier(barrier_parties)
            self._update_barrier = mp.Barrier(barrier_parties)
            logger.trace(
                f"Created barriers for {barrier_parties} parties ({self._training_parameters.workers} workers + 1 leader)"
            )

            stop_event = mp.Event()
            self._worker_ready_events = tuple(
                mp.Event() for _ in range(self._training_parameters.workers)
            )
            self._reload_events = tuple(
                mp.Event() for _ in range(self._training_parameters.workers)
            )

            with log_time("SharedMemoryManager initialization: {time_taken:.4f}s"):
                self._shared_memory_manager = SharedMemoryManager(
                    worker_count=self._training_parameters.workers,
                    gradient_n_bytes=self._model.parameter_n_bytes,
                    update_shared_memory=self._update_shared_memory,
                    gradient_barrier=self._gradient_barrier,
                    update_barrier=self._update_barrier,
                )

            with log_time(
                f"Worker process creation ({self._training_parameters.workers} workers): {{time_taken:.4f}}s"
            ):
                self._processes = tuple(
                    ParallelTrainer.create_worker_process(
                        batch_size=self._training_parameters.batch_size
                        // self._training_parameters.workers,
                        model_checkpoint_path=self._model_checkpoint_path,
                        reload_event=self._reload_events[i],
                        regulariser_lambda=self._training_parameters.regulariser_lambda,
                        shared_memory_manager=self._shared_memory_manager,
                        stop_event=stop_event,
                        worker_id=i,
                        log_level=self._training_parameters.log_level,
                        worker_ready_event=self._worker_ready_events[i],
                        X=self._X_train,
                        X_shared_memory_name=self._X_shared_memory.name,
                        Y=self._Y_train,
                        Y_shared_memory_name=self._Y_shared_memory.name,
                    )
                    for i in range(self._training_parameters.workers)
                )

            with log_time(
                f"Worker readiness wait ({len(self._worker_ready_events)} workers): {{time_taken:.4f}}s"
            ):
                self._ready_all_workers()

    def _ready_all_workers(self) -> None:
        logger.trace(
            f"Waiting for {len(self._worker_ready_events)} workers to be ready"
        )

        for i, event in enumerate(self._worker_ready_events):
            with log_time(f"Worker {i} ready wait: {{time_taken:.4f}}s"):
                event.wait()

    def _training_step(
        self,
        X_train_batch: jnp.ndarray,
        Y_train_batch: jnp.ndarray,
    ) -> tuple[
        Sequence[SupportsGradientOperations],
        Sequence[SupportsGradientOperations],
    ]:
        del X_train_batch, Y_train_batch  # unused

        with self._create_training_step_context():
            with log_time("Leader training step: {time_taken:.4f}s total"):
                logger.trace("Leader starting training step, waiting for gradients")

                with log_time("Leader gradient aggregation: {time_taken:.4f}s"):
                    if self._shared_memory_manager is None:
                        raise RuntimeError("Shared memory manager not initialized.")
                    self._shared_memory_manager.leader_get_aggregated_results(
                        self._model
                    )

                with log_time("Leader optimiser compute: {time_taken:.4f}s"):
                    self._optimiser.compute_update()

                with log_time("Leader update broadcast: {time_taken:.4f}s"):
                    update = self._model.get_gradient_caches()
                    self._shared_memory_manager.leader_send_update(update)

                with log_time("Leader parameter update: {time_taken:.4f}s"):
                    self._model.update_parameters()

                return (
                    update,
                    update,
                )  # TODO: return layer_id->gradient+update mapping if requested.

    def shutdown(self) -> None:
        logger.trace("Starting ParallelTrainer shutdown")

        super().shutdown()

        barrier_errors = []

        if self._gradient_barrier is not None:
            try:
                self._gradient_barrier.abort()
                logger.trace("Aborted gradient barrier")
            except Exception as e:
                barrier_errors.append(f"gradient barrier: {e}")
                logger.warning(f"Failed to abort gradient barrier: {e}")

        if self._update_barrier is not None:
            try:
                self._update_barrier.abort()
                logger.trace("Aborted update barrier")
            except Exception as e:
                barrier_errors.append(f"update barrier: {e}")
                logger.warning(f"Failed to abort update barrier: {e}")

        for i, p in enumerate(self._processes):
            try:
                p.terminate()
                logger.trace(f"Terminated worker process {i} (PID {p.pid})")
            except Exception as e:
                logger.warning(f"Error terminating worker process {i}: {e}")

        memory_errors = []

        if self._X_shared_memory is not None:
            try:
                self._X_shared_memory.close()
                self._X_shared_memory.unlink()
                logger.trace("Cleaned up X shared memory")
            except Exception as e:
                memory_errors.append(f"X memory: {e}")
                logger.warning(f"X shared memory cleanup error: {e}")

        if self._Y_shared_memory is not None:
            try:
                self._Y_shared_memory.close()
                self._Y_shared_memory.unlink()
                logger.trace("Cleaned up Y shared memory")
            except Exception as e:
                memory_errors.append(f"Y memory: {e}")
                logger.warning(f"Y shared memory cleanup error: {e}")

        if self._update_shared_memory is not None:
            try:
                self._update_shared_memory.close()
                self._update_shared_memory.unlink()
                logger.trace("Cleaned up update shared memory")
            except Exception as e:
                memory_errors.append(f"update memory: {e}")
                logger.warning(f"Update shared memory cleanup error: {e}")

        if self._shared_memory_manager is not None:
            try:
                self._shared_memory_manager.cleanup()
                logger.trace("Cleaned up shared memory manager")
            except Exception as e:
                memory_errors.append(f"manager: {e}")
                logger.warning(f"Shared memory manager cleanup error: {e}")

        if barrier_errors:
            logger.warning(f"Barrier errors: {barrier_errors}")
        if memory_errors:
            logger.warning(f"Memory errors: {memory_errors}")
