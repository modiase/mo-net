import functools
import multiprocessing as mp
import pickle
import sys
import time
from collections.abc import Callable, Sequence
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Final, ParamSpec, TypeVar

import numpy as np
from loguru import logger

from mo_net.model.model import Model
from mo_net.protos import EventLike, SupportsGradientOperations, UpdateGradientType
from mo_net.regulariser.weight_decay import attach_weight_decay_regulariser
from mo_net.train.trainer.trainer import (
    BasicTrainer,
    TrainingResult,
    TrainingSuccessful,
)

_64_BIT_FLOAT_BYTES_SIZE: Final[int] = 8
_PADDING_FACTOR: Final[float] = 1.2
_DATA_BYTES_LEN_OFFSET: Final[int] = 4


class GradientSerializer:
    """Handles serialization/deserialization of gradients to/from numpy arrays"""

    @staticmethod
    def serialize_gradients(
        gradients: tuple[SupportsGradientOperations, ...],
    ) -> np.ndarray:
        """Convert gradient objects to a single flat numpy array"""
        start_time = time.perf_counter()
        logger.trace(
            f"Starting gradient serialization for {len(gradients)} gradient objects"
        )

        arrays = []
        total_params = 0

        for i, grad in enumerate(gradients):
            # Handle different layer types
            if hasattr(grad, "_W") and hasattr(grad, "_B"):  # Linear layer
                arrays.extend([grad._W.flatten(), grad._B.flatten()])
                params = grad._W.size + grad._B.size
                logger.trace(
                    f"Serialized linear layer {i}: {grad._W.shape} + {grad._B.shape} = {params} params"
                )
                total_params += params
            elif hasattr(grad, "weights") and hasattr(grad, "biases"):  # Conv layer
                arrays.extend([grad.weights.flatten(), grad.biases.flatten()])
                params = grad.weights.size + grad.biases.size
                logger.trace(
                    f"Serialized conv layer {i}: {grad.weights.shape} + {grad.biases.shape} = {params} params"
                )
                total_params += params
            elif hasattr(grad, "_gamma") and hasattr(grad, "_beta"):  # BatchNorm layer
                arrays.extend([grad._gamma.flatten(), grad._beta.flatten()])
                params = grad._gamma.size + grad._beta.size
                logger.trace(
                    f"Serialized batch_norm layer {i}: {grad._gamma.shape} + {grad._beta.shape} = {params} params"
                )
                total_params += params
            else:
                raise ValueError(f"Unknown gradient type: {type(grad)}")

        result = np.concatenate(arrays) if arrays else np.array([])
        duration = time.perf_counter() - start_time
        logger.trace(
            f"Gradient serialization completed: {total_params} total params -> {result.size} array elements in {duration:.4f}s"
        )
        return result

    @staticmethod
    def deserialize_gradients(
        flat_array: np.ndarray,
        gradient_shapes: list[tuple[str, tuple[tuple[int, ...], tuple[int, ...]]]],
    ) -> tuple[SupportsGradientOperations, ...]:
        """Convert flat numpy array back to gradient objects"""
        start_time = time.perf_counter()
        logger.trace(
            f"Starting gradient deserialization: {flat_array.size} elements -> {len(gradient_shapes)} gradient objects"
        )

        gradients = []
        offset = 0

        for i, (layer_type, (shape1, shape2)) in enumerate(gradient_shapes):
            size1 = np.prod(shape1)
            size2 = np.prod(shape2)

            param1 = flat_array[offset : offset + size1].reshape(shape1)
            param2 = flat_array[offset + size1 : offset + size1 + size2].reshape(shape2)

            if layer_type == "linear":
                from mnist_numpy.model.layer.linear import Linear

                gradients.append(Linear.Parameters(_W=param1, _B=param2))
            elif layer_type == "conv":
                from mnist_numpy.model.layer.convolution import Convolution2D

                gradients.append(
                    Convolution2D.Parameters(weights=param1, biases=param2)
                )
            elif layer_type == "batch_norm":
                from mnist_numpy.model.layer.batch_norm import BatchNorm

                gradients.append(BatchNorm.Parameters(_gamma=param1, _beta=param2))

            logger.trace(
                f"Deserialized {layer_type} layer {i}: {shape1} + {shape2} = {size1 + size2} params"
            )
            offset += size1 + size2

        duration = time.perf_counter() - start_time
        logger.trace(
            f"Gradient deserialization completed: {len(gradients)} objects in {duration:.4f}s"
        )
        return tuple(gradients)

    @staticmethod
    def get_gradient_shapes(
        model: Model,
    ) -> list[tuple[str, tuple[tuple[int, ...], tuple[int, ...]]]]:
        """Extract shape information for gradient reconstruction"""
        logger.trace("Extracting gradient shapes from model")
        shapes = []
        total_params = 0

        for i, layer in enumerate(model.grad_layers):
            if hasattr(layer.parameters, "_W"):  # Linear layer
                shape_info = (
                    "linear",
                    (layer.parameters._W.shape, layer.parameters._B.shape),
                )
                shapes.append(shape_info)
                params = layer.parameters._W.size + layer.parameters._B.size
                logger.trace(
                    f"Layer {i} (linear): {layer.parameters._W.shape} + {layer.parameters._B.shape} = {params} params"
                )
                total_params += params
            elif hasattr(layer.parameters, "weights"):  # Conv layer
                shape_info = (
                    "conv",
                    (layer.parameters.weights.shape, layer.parameters.biases.shape),
                )
                shapes.append(shape_info)
                params = layer.parameters.weights.size + layer.parameters.biases.size
                logger.trace(
                    f"Layer {i} (conv): {layer.parameters.weights.shape} + {layer.parameters.biases.shape} = {params} params"
                )
                total_params += params
            elif hasattr(layer.parameters, "_gamma"):  # BatchNorm layer
                shape_info = (
                    "batch_norm",
                    (layer.parameters._gamma.shape, layer.parameters._beta.shape),
                )
                shapes.append(shape_info)
                params = layer.parameters._gamma.size + layer.parameters._beta.size
                logger.trace(
                    f"Layer {i} (batch_norm): {layer.parameters._gamma.shape} + {layer.parameters._beta.shape} = {params} params"
                )
                total_params += params

        logger.trace(
            f"Gradient shape extraction completed: {len(shapes)} layers, {total_params} total parameters"
        )
        return shapes


class SharedMemoryManager:
    """Manages shared memory for efficient gradient aggregation using barrier synchronization"""

    def __init__(
        self,
        *,
        worker_count: int,
        gradient_size: int,
        update_shared_memory: SharedMemory,
        gradient_barrier: mp.Barrier,
        update_barrier: mp.Barrier,
    ):
        logger.trace(
            f"Initializing SharedMemoryManager: {worker_count} workers, {gradient_size} gradient size"
        )

        self.worker_count = worker_count
        self.gradient_size = gradient_size
        self._update_shared_memory = update_shared_memory
        self._gradient_barrier = gradient_barrier
        self._update_barrier = update_barrier

        # Create shared memory for each worker's gradients
        self._gradient_shared_memories = []
        total_gradient_memory = 0

        for i in range(worker_count):
            # Shared memory for gradients
            memory_size = gradient_size * _64_BIT_FLOAT_BYTES_SIZE
            gradient_memory = mp.shared_memory.SharedMemory(
                create=True, size=memory_size
            )
            self._gradient_shared_memories.append(gradient_memory)
            total_gradient_memory += memory_size
            logger.trace(
                f"Created gradient shared memory for worker {i}: {memory_size} bytes ({gradient_memory.name})"
            )

        logger.trace(
            f"SharedMemoryManager initialized: {total_gradient_memory} bytes total gradient memory, "
            f"{self._update_shared_memory.size} bytes update memory"
        )

    def get_worker_gradient_memory_name(self, worker_id: int) -> str:
        """Get shared memory name for worker's gradients"""
        memory_name = self._gradient_shared_memories[worker_id].name
        logger.trace(
            f"Retrieved gradient memory name for worker {worker_id}: {memory_name}"
        )
        return memory_name

    def worker_put_result(
        self, worker_id: int, gradients: tuple[SupportsGradientOperations, ...]
    ) -> None:
        """Worker writes gradients to shared memory and waits at barrier"""
        start_time = time.perf_counter()
        logger.trace(f"Worker {worker_id} starting gradient write to shared memory")

        flat_gradients = GradientSerializer.serialize_gradients(gradients)
        serialize_time = time.perf_counter()

        # Write to shared memory
        gradient_buffer = np.ndarray(
            shape=(self.gradient_size,),
            dtype=np.float64,
            buffer=self._gradient_shared_memories[worker_id].buf,
        )

        # Pad or truncate if necessary
        if len(flat_gradients) <= self.gradient_size:
            gradient_buffer[: len(flat_gradients)] = flat_gradients
            gradient_buffer[len(flat_gradients) :] = 0  # Zero-pad
            logger.trace(
                f"Worker {worker_id} wrote {len(flat_gradients)} gradients, zero-padded to {self.gradient_size}"
            )
        else:
            gradient_buffer[:] = flat_gradients[: self.gradient_size]
            logger.warning(
                f"Worker {worker_id}: Gradient size {len(flat_gradients)} exceeds buffer size {self.gradient_size}, truncating"
            )

        write_time = time.perf_counter()
        logger.trace(
            f"Worker {worker_id} wrote gradients to shared memory in {write_time - serialize_time:.4f}s, "
            f"now waiting at gradient barrier"
        )

        # Wait at barrier for all workers to finish writing gradients
        barrier_start = time.perf_counter()
        self._gradient_barrier.wait()
        barrier_time = time.perf_counter() - barrier_start

        total_time = time.perf_counter() - start_time
        logger.trace(
            f"Worker {worker_id} passed gradient barrier after {barrier_time:.4f}s wait "
            f"(total gradient submission: {total_time:.4f}s)"
        )

    def leader_get_aggregated_results(
        self, gradient_shapes
    ) -> Sequence[SupportsGradientOperations]:
        """Leader waits at barrier then aggregates gradients from all workers' shared memory"""
        start_time = time.perf_counter()
        logger.trace("Leader waiting at gradient barrier for all workers")

        # Wait at barrier for all workers to finish writing gradients
        barrier_start = time.perf_counter()
        self._gradient_barrier.wait()
        barrier_time = time.perf_counter() - barrier_start
        logger.trace(
            f"Leader passed gradient barrier after {barrier_time:.4f}s, starting gradient aggregation"
        )

        # Aggregate gradients
        aggregated_gradients = np.zeros(self.gradient_size, dtype=np.float64)
        aggregation_start = time.perf_counter()

        for i in range(self.worker_count):
            worker_buffer = np.ndarray(
                shape=(self.gradient_size,),
                dtype=np.float64,
                buffer=self._gradient_shared_memories[i].buf,
            )
            aggregated_gradients += worker_buffer
            logger.trace(f"Leader aggregated gradients from worker {i}")

        aggregated_gradients /= self.worker_count

        aggregation_time = time.perf_counter() - aggregation_start
        logger.trace(
            f"Leader completed gradient aggregation from {self.worker_count} workers in {aggregation_time:.4f}s"
        )

        # Convert back to gradient objects
        deserialize_start = time.perf_counter()
        result = GradientSerializer.deserialize_gradients(
            aggregated_gradients, gradient_shapes
        )
        deserialize_time = time.perf_counter() - deserialize_start

        total_time = time.perf_counter() - start_time
        logger.trace(
            f"Leader gradient aggregation completed: barrier {barrier_time:.4f}s + "
            f"aggregation {aggregation_time:.4f}s + deserialization {deserialize_time:.4f}s = {total_time:.4f}s total"
        )
        return result

    def leader_send_update(self, update: Sequence[SupportsGradientOperations]) -> None:
        """Send parameter updates to workers via shared memory and barrier synchronization"""
        start_time = time.perf_counter()
        logger.trace("Leader starting parameter update broadcast")

        # Serialize parameter updates to shared memory
        serialize_start = time.perf_counter()
        data_bytes = pickle.dumps(update)
        data_bytes_len = len(data_bytes)
        serialize_time = time.perf_counter() - serialize_start

        self._update_shared_memory.buf[0:_DATA_BYTES_LEN_OFFSET] = (
            data_bytes_len.to_bytes(_DATA_BYTES_LEN_OFFSET, byteorder="little")
        )
        self._update_shared_memory.buf[
            _DATA_BYTES_LEN_OFFSET : _DATA_BYTES_LEN_OFFSET + data_bytes_len
        ] = data_bytes

        write_time = time.perf_counter()
        logger.trace(
            f"Leader serialized and wrote {data_bytes_len} bytes of updates in "
            f"{serialize_time:.4f}s + {write_time - serialize_start - serialize_time:.4f}s, "
            f"now waiting at update barrier"
        )

        # Wait at barrier for all workers to be ready for update
        barrier_start = time.perf_counter()
        self._update_barrier.wait()
        barrier_time = time.perf_counter() - barrier_start

        total_time = time.perf_counter() - start_time
        logger.trace(
            f"Leader completed update broadcast: serialization {serialize_time:.4f}s + "
            f"barrier {barrier_time:.4f}s = {total_time:.4f}s total"
        )

    def worker_wait_for_update(self) -> UpdateGradientType | None:
        """Worker waits at barrier for parameter updates"""
        start_time = time.perf_counter()
        logger.trace("Worker waiting at update barrier for parameter updates")

        # Wait at barrier for leader to write update
        barrier_start = time.perf_counter()
        self._update_barrier.wait()
        barrier_time = time.perf_counter() - barrier_start
        logger.trace(
            f"Worker passed update barrier after {barrier_time:.4f}s, reading updates"
        )

        # Read update from shared memory
        read_start = time.perf_counter()
        data_bytes_len = int.from_bytes(
            self._update_shared_memory.buf[0:_DATA_BYTES_LEN_OFFSET],
            byteorder="little",
        )
        data = bytes(
            self._update_shared_memory.buf[
                _DATA_BYTES_LEN_OFFSET : _DATA_BYTES_LEN_OFFSET + data_bytes_len
            ]
        )

        deserialize_start = time.perf_counter()
        result = pickle.loads(data)
        deserialize_time = time.perf_counter() - deserialize_start

        total_time = time.perf_counter() - start_time
        logger.trace(
            f"Worker received parameter update: {data_bytes_len} bytes, "
            f"barrier {barrier_time:.4f}s + read {deserialize_start - read_start:.4f}s + "
            f"deserialize {deserialize_time:.4f}s = {total_time:.4f}s total"
        )
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
    def wrapper(*args: P.args, worker_id: int, log_level: str, **kwargs: P.kwargs) -> R:
        logger.configure(extra={"worker_id": worker_id})
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | Worker {extra[worker_id]} | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=log_level,
        )
        logger.trace(
            f"Worker {worker_id} decorator initialized with log level {log_level}"
        )
        return func(*args, worker_id=worker_id, log_level=log_level, **kwargs)

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
    X_shared_memory_dtype: np.dtype,
    X_shared_memory_name: str,
    X_shared_memory_shape: Sequence[int],
    Y_shared_memory_dtype: np.dtype,
    Y_shared_memory_name: str,
    Y_shared_memory_shape: Sequence[int],
) -> None:
    """Worker process that trains on batches and submits updates via barrier synchronization"""
    del log_level  # unused
    del regulariser_lambda  # unused

    process_start = time.perf_counter()
    logger.trace(
        f"Worker {worker_id} process starting with PID {mp.current_process().pid}"
    )

    # Load model
    model_start = time.perf_counter()
    with open(model_checkpoint_path, "rb") as f:
        model = Model.load(f, training=True)
    model_time = time.perf_counter() - model_start
    logger.trace(
        f"Worker {worker_id} loaded model in {model_time:.4f}s from {model_checkpoint_path}"
    )

    # Signal ready and connect to shared memory
    worker_ready_event.set()
    logger.trace(f"Worker {worker_id} signaled ready, connecting to shared memory")

    memory_start = time.perf_counter()
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
    memory_time = time.perf_counter() - memory_start

    startup_time = time.perf_counter() - process_start
    logger.trace(
        f"Worker {worker_id} initialization completed in {startup_time:.4f}s "
        f"(model: {model_time:.4f}s, memory: {memory_time:.4f}s). "
        f"X_train: {X_train.shape} {X_train.dtype}, Y_train: {Y_train.shape} {Y_train.dtype}"
    )

    iteration_count = 0
    total_forward_time = 0
    total_backward_time = 0
    total_gradient_time = 0
    total_update_time = 0

    while not stop_event.is_set():
        try:
            iteration_start = time.perf_counter()
            iteration_count += 1
            logger.trace(f"Worker {worker_id} starting iteration {iteration_count}")

            if reload_event.is_set():
                reload_start = time.perf_counter()
                with open(model_checkpoint_path, "rb") as f:
                    model = Model.load(f, training=True)
                reload_time = time.perf_counter() - reload_start
                reload_event.clear()
                worker_ready_event.set()
                logger.trace(f"Worker {worker_id} reloaded model in {reload_time:.4f}s")

            # Generate random batch
            indices = np.random.choice(X_train.shape[0], size=batch_size, replace=False)
            X_batch = X_train[indices]
            Y_batch = Y_train[indices]

            # Forward pass
            forward_start = time.perf_counter()
            model.forward_prop(X=X_batch)
            forward_time = time.perf_counter() - forward_start
            total_forward_time += forward_time

            # Backward pass
            backward_start = time.perf_counter()
            model.backward_prop(Y_true=Y_batch)
            backward_time = time.perf_counter() - backward_start
            total_backward_time += backward_time

            logger.trace(
                f"Worker {worker_id} completed forward/backward pass: "
                f"forward {forward_time:.4f}s, backward {backward_time:.4f}s"
            )

            # Submit gradients via barrier synchronization
            gradient_start = time.perf_counter()
            gradients = tuple(layer.cache["dP"] for layer in model.grad_layers)
            shared_memory_manager.worker_put_result(worker_id, gradients)
            gradient_time = time.perf_counter() - gradient_start
            total_gradient_time += gradient_time

            # Receive parameter updates
            if not reload_event.is_set():
                update_start = time.perf_counter()
                aggregated_update = shared_memory_manager.worker_wait_for_update()
                model.populate_caches(aggregated_update)
                model.update_parameters()
                update_time = time.perf_counter() - update_start
                total_update_time += update_time

                iteration_time = time.perf_counter() - iteration_start
                logger.trace(
                    f"Worker {worker_id} completed iteration {iteration_count} in {iteration_time:.4f}s "
                    f"(gradient: {gradient_time:.4f}s, update: {update_time:.4f}s)"
                )

                # Log performance summary every 10 iterations
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

        except mp.BrokenBarrierError:
            logger.warning(
                f"Worker {worker_id} detected broken barrier after {iteration_count} iterations, stopping gracefully"
            )
            break
        except Exception as e:
            logger.exception(
                f"Worker {worker_id} error in iteration {iteration_count}: {e}"
            )
            break

    # Cleanup
    cleanup_start = time.perf_counter()
    try:
        X_shared_memory.close()
        Y_shared_memory.close()
    except Exception as e:
        logger.warning(f"Worker {worker_id} shared memory cleanup error: {e}")

    cleanup_time = time.perf_counter() - cleanup_start
    total_runtime = time.perf_counter() - process_start
    logger.trace(
        f"Worker {worker_id} shutting down after {iteration_count} iterations, "
        f"{total_runtime:.4f}s total runtime, cleanup {cleanup_time:.4f}s"
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
        self._gradient_barrier: mp.Barrier | None = None
        self._update_barrier: mp.Barrier | None = None
        logger.trace("ParallelTrainer initialization completed")

    def resume(
        self,
        *,
        start_epoch: int,
        model_checkpoint_path: Path,
    ) -> TrainingResult:
        logger.info(f"Resuming training from epoch {start_epoch}.")
        resume_start = time.perf_counter()

        self._start_epoch = start_epoch
        self._model = Model.load(open(model_checkpoint_path, "rb"), training=True)
        self._optimizer.set_model(self._model)
        self._optimizer.restore()
        if self._monitor is not None:
            self._monitor.reset(restore_history=True)

        # Reset worker events
        for event in self._worker_ready_events:
            event.clear()
        for event in self._reload_events:
            event.set()
        for event in self._worker_ready_events:
            event.wait()

        resume_time = time.perf_counter() - resume_start
        logger.trace(f"Training resume setup completed in {resume_time:.4f}s")

        with self._create_training_loop_context():
            self._training_loop()

        return TrainingSuccessful(
            model_checkpoint_path=model_checkpoint_path,
        )

    @staticmethod
    def create_worker_process(
        *,
        X: np.ndarray,
        X_shared_memory_name: str,
        Y: np.ndarray,
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
        setup_start = time.perf_counter()
        logger.trace("Starting parallel training setup")

        if self._training_parameters.regulariser_lambda > 0:
            regulariser_start = time.perf_counter()
            attach_weight_decay_regulariser(
                lambda_=self._training_parameters.regulariser_lambda,
                batch_size=self._X_train.shape[0],
                model=self._model,
                optimizer=self._optimizer,
            )
            regulariser_time = time.perf_counter() - regulariser_start
            logger.trace(
                f"Attached weight decay regulariser (λ={self._training_parameters.regulariser_lambda}) in {regulariser_time:.4f}s"
            )

        # Calculate gradient size
        gradient_calc_start = time.perf_counter()
        gradient_size = sum(
            param.size
            for layer in self._model.grad_layers
            for param in [
                layer.parameters._W
                if hasattr(layer.parameters, "_W")
                else layer.parameters.weights
                if hasattr(layer.parameters, "weights")
                else layer.parameters._gamma,
                layer.parameters._B
                if hasattr(layer.parameters, "_B")
                else layer.parameters.biases
                if hasattr(layer.parameters, "biases")
                else layer.parameters._beta,
            ]
        )
        gradient_calc_time = time.perf_counter() - gradient_calc_start

        # Store gradient shapes for reconstruction
        self._gradient_shapes = GradientSerializer.get_gradient_shapes(self._model)
        logger.trace(
            f"Calculated gradient size: {gradient_size} parameters across {len(self._gradient_shapes)} layers in {gradient_calc_time:.4f}s"
        )

        # Create shared memories
        memory_start = time.perf_counter()
        self._X_shared_memory = mp.shared_memory.SharedMemory(
            create=True, size=self._X_train.nbytes
        )
        self._Y_shared_memory = mp.shared_memory.SharedMemory(
            create=True, size=self._Y_train.nbytes
        )
        update_memory_size = int(
            self._model.parameter_count * _64_BIT_FLOAT_BYTES_SIZE * _PADDING_FACTOR
        )
        self._update_shared_memory = mp.shared_memory.SharedMemory(
            create=True,
            size=update_memory_size,
        )

        total_memory = self._X_train.nbytes + self._Y_train.nbytes + update_memory_size
        logger.trace(
            f"Created shared memories: X {self._X_train.nbytes} bytes, Y {self._Y_train.nbytes} bytes, "
            f"updates {update_memory_size} bytes = {total_memory} bytes total"
        )

        # Copy training data to shared memory
        copy_start = time.perf_counter()
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
        copy_time = time.perf_counter() - copy_start
        logger.trace(f"Copied training data to shared memory in {copy_time:.4f}s")

        # Create barriers for synchronization (workers + 1 leader)
        barrier_parties = self._training_parameters.workers + 1
        self._gradient_barrier = mp.Barrier(barrier_parties)
        self._update_barrier = mp.Barrier(barrier_parties)
        logger.trace(
            f"Created barriers for {barrier_parties} parties ({self._training_parameters.workers} workers + 1 leader)"
        )

        # Set up synchronization
        sync_start = time.perf_counter()
        stop_event = mp.Event()
        self._worker_ready_events = tuple(
            mp.Event() for _ in range(self._training_parameters.workers)
        )
        self._reload_events = tuple(
            mp.Event() for _ in range(self._training_parameters.workers)
        )
        sync_time = time.perf_counter() - sync_start

        # Create shared memory manager with barriers
        manager_start = time.perf_counter()
        self._shared_memory_manager = SharedMemoryManager(
            worker_count=self._training_parameters.workers,
            gradient_size=gradient_size,
            update_shared_memory=self._update_shared_memory,
            gradient_barrier=self._gradient_barrier,
            update_barrier=self._update_barrier,
        )
        manager_time = time.perf_counter() - manager_start

        # Create worker processes
        processes_start = time.perf_counter()
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
        processes_time = time.perf_counter() - processes_start

        logger.trace(
            f"Created {len(self._processes)} worker processes in {processes_time:.4f}s"
        )

        ready_start = time.perf_counter()
        self._ready_all_workers()
        ready_time = time.perf_counter() - ready_start

        total_setup_time = time.perf_counter() - setup_start
        logger.trace(
            f"Parallel training setup completed in {total_setup_time:.4f}s "
            f"(memory: {copy_time + memory_start - gradient_calc_time:.4f}s, "
            f"sync: {sync_time:.4f}s, manager: {manager_time:.4f}s, "
            f"processes: {processes_time:.4f}s, ready: {ready_time:.4f}s)"
        )

    def _ready_all_workers(self) -> None:
        logger.trace(
            f"Waiting for {len(self._worker_ready_events)} workers to be ready"
        )
        ready_start = time.perf_counter()

        for i, event in enumerate(self._worker_ready_events):
            event_start = time.perf_counter()
            event.wait()
            event_time = time.perf_counter() - event_start
            logger.trace(f"Worker {i} ready after {event_time:.4f}s")

        total_ready_time = time.perf_counter() - ready_start
        logger.trace(
            f"All {len(self._worker_ready_events)} workers ready in {total_ready_time:.4f}s"
        )

    def _training_step(
        self,
        X_train_batch: np.ndarray,
        Y_train_batch: np.ndarray,
    ) -> tuple[
        Sequence[SupportsGradientOperations],
        Sequence[SupportsGradientOperations],
    ]:
        del X_train_batch, Y_train_batch  # unused
        step_start = time.perf_counter()

        with self._create_training_step_context():
            logger.trace("Leader starting training step, waiting for gradients")

            gradient_start = time.perf_counter()
            gradient = self._shared_memory_manager.leader_get_aggregated_results(
                self._gradient_shapes
            )
            gradient_time = time.perf_counter() - gradient_start

            cache_start = time.perf_counter()
            self._model.populate_caches(gradient)
            cache_time = time.perf_counter() - cache_start

            compute_start = time.perf_counter()
            self._optimizer.compute_update()
            compute_time = time.perf_counter() - compute_start

            update_start = time.perf_counter()
            update = self._model.get_gradient_caches()
            self._shared_memory_manager.leader_send_update(update)
            update_time = time.perf_counter() - update_start

            param_start = time.perf_counter()
            self._model.update_parameters()
            param_time = time.perf_counter() - param_start

            total_step_time = time.perf_counter() - step_start
            logger.trace(
                f"Leader training step completed in {total_step_time:.4f}s "
                f"(gradients: {gradient_time:.4f}s, cache: {cache_time:.4f}s, "
                f"compute: {compute_time:.4f}s, update: {update_time:.4f}s, params: {param_time:.4f}s)"
            )

            return gradient, update

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
