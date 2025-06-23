import struct
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
from loguru import logger

from mo_net.model.layer.base import ParametrisedHidden
from mo_net.model.layer.batch_norm import BatchNorm
from mo_net.model.layer.convolution import Convolution2D
from mo_net.model.layer.linear import Linear
from mo_net.model.model import Model
from mo_net.train.trainer.parallel import SharedMemoryManager


class MockedSharedMemoryManager(SharedMemoryManager):
    """Mocked version of SharedMemoryManager that uses BufferIOAdapter instead of shared memory"""

    def __init__(self, worker_count: int, gradient_n_bytes: int) -> None:
        self.worker_count = worker_count
        self._gradient_size_bytes = int(
            gradient_n_bytes * 1.2
        )  # extra space for headroom
        self._gradient_barrier = Mock()
        self._update_barrier = Mock()
        self._gradient_buffers = [
            BytesIO(b"\x00" * self._gradient_size_bytes) for _ in range(worker_count)
        ]

    def worker_put_result(
        self, worker_id: int, grad_layers: Sequence[ParametrisedHidden]
    ) -> None:
        """Simplified worker_put_result without barrier synchronization"""
        writer = self._gradient_buffers[worker_id]
        writer.seek(0)

        for layer in grad_layers:
            layer.write_serialized_parameters(writer)

        writer.write(b"\x00" * (self._gradient_size_bytes - writer.tell()))

    def leader_get_aggregated_results(self, model: Model) -> None:
        """Simplified leader_get_aggregated_results without barrier synchronization"""
        for worker_id in range(self.worker_count):
            reader = self._gradient_buffers[worker_id]
            reader.seek(0)

            while reader.tell() < self._gradient_size_bytes:
                logger.debug(
                    f"Reading from worker {worker_id} at position {reader.tell()}"
                )
                try:
                    if not ParametrisedHidden.get_layer_id(reader, peek=True):
                        logger.debug(
                            f"Hit padding for worker {worker_id} (empty layer_id)"
                        )
                        break

                    model.get_parametrised_layer(
                        ParametrisedHidden.get_layer_id(reader, peek=True)
                    ).read_serialized_parameters(reader)

                except (struct.error, UnicodeDecodeError, ValueError, IndexError):
                    logger.debug(f"Hit malformed data/end for worker {worker_id}")
                    break

        for layer in model.grad_layers:
            if layer.cache["dP"] is not None:
                layer.cache["dP"] = layer.cache["dP"] / self.worker_count


@dataclass(frozen=True)
class GradientTransferTestCase:
    name: str
    model: Model
    forward_input: np.ndarray
    backward_input: np.ndarray
    expected_w_shape: tuple[int, ...]
    expected_b_shape: tuple[int, ...]


@dataclass(frozen=True)
class GradientAggregationTestCase:
    name: str
    model: Model
    gradient1: Any
    gradient2: Any


@pytest.mark.parametrize(
    "test_case",
    [
        GradientTransferTestCase(
            name="linear",
            model=Model(
                input_dimensions=(3,),
                hidden=[
                    Linear(
                        input_dimensions=(3,),
                        output_dimensions=(2,),
                        parameters=Linear.Parameters.xavier(dim_in=(3,), dim_out=(2,)),
                    )
                ],
            ),
            forward_input=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            backward_input=np.array([[1.0, 0.0], [0.0, 1.0]]),
            expected_w_shape=(3, 2),
            expected_b_shape=(2,),
        ),
        GradientTransferTestCase(
            name="convolution",
            model=Model(
                input_dimensions=(1, 4, 4),
                hidden=[
                    Convolution2D(
                        input_dimensions=(1, 4, 4),
                        n_kernels=2,
                        kernel_size=3,
                        stride=1,
                        kernel_init_fn=Convolution2D.Parameters.xavier,
                    )
                ],
            ),
            forward_input=np.random.rand(2, 1, 4, 4),
            backward_input=np.random.rand(2, 2, 2, 2),
            expected_w_shape=(2, 1, 3, 3),
            expected_b_shape=(2,),
        ),
        GradientTransferTestCase(
            name="batch_norm",
            model=Model(
                input_dimensions=(4,),
                hidden=[BatchNorm(input_dimensions=(4,), training=True)],
            ),
            forward_input=np.random.rand(3, 4),
            backward_input=np.random.rand(3, 4),
            expected_w_shape=(4,),
            expected_b_shape=(4,),
        ),
    ],
)
@pytest.mark.parametrize("write_count", [1, 2])
def test_gradient_transfer(
    test_case: GradientTransferTestCase, write_count: int
) -> None:
    test_case.model.forward_prop(test_case.forward_input)
    test_case.model.backward_prop(test_case.backward_input)

    original_gradient = test_case.model.grad_layers[0].cache["dP"]
    assert original_gradient is not None

    manager = MockedSharedMemoryManager(
        worker_count=write_count, gradient_n_bytes=test_case.model.parameter_n_bytes
    )

    for layer in test_case.model.grad_layers:
        layer.cache["dP"] = None

    for worker_id in range(write_count):
        test_case.model.grad_layers[0].cache["dP"] = original_gradient
        manager.worker_put_result(worker_id, test_case.model.grad_layers)

    for layer in test_case.model.grad_layers:
        layer.cache["dP"] = None

    manager.leader_get_aggregated_results(test_case.model)

    final_gradient = test_case.model.grad_layers[0].cache["dP"]
    assert final_gradient is not None

    np.testing.assert_allclose(
        final_gradient.weights,
        original_gradient.weights,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        final_gradient.biases,
        original_gradient.biases,
        rtol=1e-10,
    )

    assert final_gradient.weights.shape == test_case.expected_w_shape
    assert final_gradient.biases.shape == test_case.expected_b_shape


@pytest.mark.parametrize(
    "test_case",
    [
        GradientAggregationTestCase(
            name="linear",
            model=Model(
                input_dimensions=(2,),
                hidden=[
                    Linear(
                        input_dimensions=(2,),
                        output_dimensions=(1,),
                        parameters=Linear.Parameters.xavier(dim_in=(2,), dim_out=(1,)),
                    )
                ],
            ),
            gradient1=Linear.Parameters(
                weights=np.array([[1.0], [2.0]], dtype=np.float32),
                biases=np.array([3.0], dtype=np.float32),
            ),
            gradient2=Linear.Parameters(
                weights=np.array([[4.0], [5.0]], dtype=np.float32),
                biases=np.array([6.0], dtype=np.float32),
            ),
        ),
        GradientAggregationTestCase(
            name="convolution",
            model=Model(
                input_dimensions=(1, 3, 3),
                hidden=[
                    Convolution2D(
                        input_dimensions=(1, 3, 3),
                        n_kernels=1,
                        kernel_size=2,
                        stride=1,
                        kernel_init_fn=Convolution2D.Parameters.xavier,
                    )
                ],
            ),
            gradient1=Convolution2D.Parameters(
                weights=np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32),
                biases=np.array([5.0], dtype=np.float32),
            ),
            gradient2=Convolution2D.Parameters(
                weights=np.array([[[[6.0, 7.0], [8.0, 9.0]]]], dtype=np.float32),
                biases=np.array([10.0], dtype=np.float32),
            ),
        ),
        GradientAggregationTestCase(
            name="batch_norm",
            model=Model(
                input_dimensions=(3,),
                hidden=[BatchNorm(input_dimensions=(3,), training=True)],
            ),
            gradient1=BatchNorm.Parameters(
                weights=np.array([1.0, 2.0, 3.0], dtype=np.float32),
                biases=np.array([4.0, 5.0, 6.0], dtype=np.float32),
            ),
            gradient2=BatchNorm.Parameters(
                weights=np.array([7.0, 8.0, 9.0], dtype=np.float32),
                biases=np.array([10.0, 11.0, 12.0], dtype=np.float32),
            ),
        ),
    ],
)
def test_gradient_transfer_aggregation(test_case: GradientAggregationTestCase) -> None:
    manager = MockedSharedMemoryManager(
        worker_count=2, gradient_n_bytes=test_case.model.parameter_n_bytes
    )

    test_case.model.grad_layers[0].cache["dP"] = test_case.gradient1
    manager.worker_put_result(0, test_case.model.grad_layers)

    test_case.model.grad_layers[0].cache["dP"] = test_case.gradient2
    manager.worker_put_result(1, test_case.model.grad_layers)

    for layer in test_case.model.grad_layers:
        layer.cache["dP"] = None

    manager.leader_get_aggregated_results(test_case.model)

    final_gradient = test_case.model.grad_layers[0].cache["dP"]
    np.testing.assert_allclose(
        final_gradient.weights,
        (test_case.gradient1.weights + test_case.gradient2.weights) / 2,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        final_gradient.biases,
        (test_case.gradient1.biases + test_case.gradient2.biases) / 2,
        rtol=1e-10,
    )
