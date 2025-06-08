import struct
from io import BytesIO
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

    def __init__(self, worker_count: int, gradient_n_bytes: int):
        # Don't call super().__init__ to avoid creating real shared memory
        self.worker_count = worker_count
        self._gradient_size_bytes = int(gradient_n_bytes * 1.2)  # 20% padding

        # Create mock barriers
        self._gradient_barrier = Mock()
        self._update_barrier = Mock()

        # Create BufferIOAdapter buffers instead of shared memory
        self._gradient_buffers = []

        for i in range(worker_count):
            # Create BufferIOAdapter buffer filled with zeros
            buffer = BytesIO(b"\x00" * self._gradient_size_bytes)
            self._gradient_buffers.append(buffer)

    def worker_put_result(self, worker_id: int, grad_layers) -> None:
        """Simplified worker_put_result without barrier synchronization"""
        writer = self._gradient_buffers[worker_id]
        writer.seek(0)  # Reset to beginning

        # Serialize each layer's parameters directly to buffer
        for layer in grad_layers:
            layer.serialize_parameters(writer)

        # Zero-pad remaining space
        bytes_written = writer.tell()
        if bytes_written < self._gradient_size_bytes:
            remaining_space = self._gradient_size_bytes - bytes_written
            writer.write(b"\x00" * remaining_space)

    def leader_get_aggregated_results(self, model: Model) -> None:
        """Simplified leader_get_aggregated_results without barrier synchronization"""
        for worker_id in range(self.worker_count):
            reader = self._gradient_buffers[worker_id]
            reader.seek(0)  # Reset to beginning

            # Read and deserialize all layers from this worker
            while reader.tell() < self._gradient_size_bytes:
                logger.debug(
                    f"Reading from worker {worker_id} at position {reader.tell()}"
                )
                try:
                    # Use peek to check if there's a valid layer header
                    layer_id = ParametrisedHidden.get_layer_id(reader, peek=True)

                    # Check if we hit padding (empty layer_id indicates padding)
                    if not layer_id:
                        logger.debug(
                            f"Hit padding for worker {worker_id} (empty layer_id)"
                        )
                        break

                    logger.debug(f"Found layer {layer_id} from worker {worker_id}")

                    layer = model.get_layer(layer_id)
                    layer.deserialize_parameters(reader)

                except (struct.error, UnicodeDecodeError, ValueError, IndexError):
                    logger.debug(f"Hit malformed data/end for worker {worker_id}")
                    break

        # Average the gradients across workers
        for layer in model.grad_layers:
            if layer.cache["dP"] is not None:
                layer.cache["dP"] = layer.cache["dP"] / self.worker_count


@pytest.mark.parametrize("write_count", [1, 2])
def test_linear_gradient_transfer(write_count: int):
    """Test linear layer gradient serialization and deserialization with mocked shared memory"""

    # Create a simple model with one Linear layer
    model = Model(
        input_dimensions=(3,),
        hidden=[
            Linear(
                input_dimensions=(3,),
                output_dimensions=(2,),
                parameters=Linear.Parameters.xavier(dim_in=(3,), dim_out=(2,)),
            )
        ],
    )

    # Create some test data
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 2 samples, 3 features
    Y = np.array([[1.0, 0.0], [0.0, 1.0]])  # 2 samples, 2 classes

    # Do forward and backward pass to populate gradients
    model.forward_prop(X)
    model.backward_prop(Y)
    logger.debug(f"Model gradients: {model.grad_layers[0].cache['dP']}")

    # Store the original gradient for comparison
    original_gradient = model.grad_layers[0].cache["dP"]
    assert original_gradient is not None, (
        "Gradient should be populated after backward pass"
    )

    # Create mocked shared memory manager
    gradient_n_bytes = model.parameter_n_bytes
    manager = MockedSharedMemoryManager(
        worker_count=write_count, gradient_n_bytes=gradient_n_bytes
    )

    # Clear the model's gradients to test deserialization
    for layer in model.grad_layers:
        layer.cache["dP"] = None

    # Simulate workers writing gradients
    for worker_id in range(write_count):
        # Reset gradient to original value for each worker
        model.grad_layers[0].cache["dP"] = original_gradient
        manager.worker_put_result(worker_id, model.grad_layers)

    # Clear gradients again before aggregation
    for layer in model.grad_layers:
        layer.cache["dP"] = None

    # Simulate leader aggregating results
    manager.leader_get_aggregated_results(model)

    # Verify the gradient was correctly deserialized and aggregated
    final_gradient = model.grad_layers[0].cache["dP"]
    assert final_gradient is not None, "Gradient should be populated after aggregation"

    # The final gradient should be the original gradient (since we average across workers)
    # For write_count=1: final = original/1 = original
    # For write_count=2: final = (original + original)/2 = original
    np.testing.assert_allclose(final_gradient._W, original_gradient._W, rtol=1e-10)
    np.testing.assert_allclose(final_gradient._B, original_gradient._B, rtol=1e-10)

    # Verify the shapes are correct
    assert final_gradient._W.shape == (3, 2), (
        f"Weight shape should be (3, 2), got {final_gradient._W.shape}"
    )
    assert final_gradient._B.shape == (2,), (
        f"Bias shape should be (2,), got {final_gradient._B.shape}"
    )


def test_linear_gradient_transfer_aggregation():
    """Test that linear layer gradients are properly aggregated when multiple workers contribute different values"""

    # Create a simple model
    model = Model(
        input_dimensions=(2,),
        hidden=[
            Linear(
                input_dimensions=(2,),
                output_dimensions=(1,),
                parameters=Linear.Parameters.xavier(dim_in=(2,), dim_out=(1,)),
            )
        ],
    )

    # Create different gradients for different workers
    gradient1 = Linear.Parameters(_W=np.array([[1.0], [2.0]]), _B=np.array([3.0]))
    gradient2 = Linear.Parameters(_W=np.array([[4.0], [5.0]]), _B=np.array([6.0]))

    # Create mocked shared memory manager for 2 workers
    gradient_n_bytes = model.parameter_n_bytes
    manager = MockedSharedMemoryManager(
        worker_count=2, gradient_n_bytes=gradient_n_bytes
    )

    # Worker 0 writes gradient1
    model.grad_layers[0].cache["dP"] = gradient1
    manager.worker_put_result(0, model.grad_layers)

    # Worker 1 writes gradient2
    model.grad_layers[0].cache["dP"] = gradient2
    manager.worker_put_result(1, model.grad_layers)

    # Clear gradients before aggregation
    for layer in model.grad_layers:
        layer.cache["dP"] = None

    # Aggregate results
    manager.leader_get_aggregated_results(model)

    # Verify aggregation: should be (gradient1 + gradient2) / 2
    final_gradient = model.grad_layers[0].cache["dP"]
    expected_W = (gradient1._W + gradient2._W) / 2  # [[2.5], [3.5]]
    expected_B = (gradient1._B + gradient2._B) / 2  # [4.5]

    np.testing.assert_allclose(final_gradient._W, expected_W, rtol=1e-10)
    np.testing.assert_allclose(final_gradient._B, expected_B, rtol=1e-10)


@pytest.mark.parametrize("write_count", [1, 2])
def test_convolution_gradient_transfer(write_count: int):
    """Test convolution layer gradient serialization and deserialization with mocked shared memory"""

    # Create a simple model with one Convolution2D layer
    model = Model(
        input_dimensions=(1, 4, 4),  # 1 channel, 4x4 image
        hidden=[
            Convolution2D(
                input_dimensions=(1, 4, 4),
                n_kernels=2,
                kernel_size=3,
                stride=1,
                kernel_init_fn=Convolution2D.Parameters.xavier,
            )
        ],
    )

    # Create some test data (batch_size=2, channels=1, height=4, width=4)
    X = np.random.rand(2, 1, 4, 4)
    # Output will be (batch_size=2, kernels=2, height=2, width=2)
    Y = np.random.rand(2, 2, 2, 2)

    # Do forward and backward pass to populate gradients
    model.forward_prop(X)
    model.backward_prop(Y)
    logger.debug(
        f"Model gradients shape: weights={model.grad_layers[0].cache['dP'].weights.shape}, biases={model.grad_layers[0].cache['dP'].biases.shape}"
    )

    # Store the original gradient for comparison
    original_gradient = model.grad_layers[0].cache["dP"]
    assert original_gradient is not None, (
        "Gradient should be populated after backward pass"
    )

    # Create mocked shared memory manager
    gradient_n_bytes = model.parameter_n_bytes
    manager = MockedSharedMemoryManager(
        worker_count=write_count, gradient_n_bytes=gradient_n_bytes
    )

    # Clear the model's gradients to test deserialization
    for layer in model.grad_layers:
        layer.cache["dP"] = None

    # Simulate workers writing gradients
    for worker_id in range(write_count):
        # Reset gradient to original value for each worker
        model.grad_layers[0].cache["dP"] = original_gradient
        manager.worker_put_result(worker_id, model.grad_layers)

    # Clear gradients again before aggregation
    for layer in model.grad_layers:
        layer.cache["dP"] = None

    # Simulate leader aggregating results
    manager.leader_get_aggregated_results(model)

    # Verify the gradient was correctly deserialized and aggregated
    final_gradient = model.grad_layers[0].cache["dP"]
    assert final_gradient is not None, "Gradient should be populated after aggregation"

    # The final gradient should be the original gradient (since we average across workers)
    # For write_count=1: final = original/1 = original
    # For write_count=2: final = (original + original)/2 = original
    np.testing.assert_allclose(
        final_gradient.weights, original_gradient.weights, rtol=1e-10
    )
    np.testing.assert_allclose(
        final_gradient.biases, original_gradient.biases, rtol=1e-10
    )

    # Verify the shapes are correct
    assert final_gradient.weights.shape == (2, 1, 3, 3), (
        f"Weight shape should be (2, 1, 3, 3), got {final_gradient.weights.shape}"
    )
    assert final_gradient.biases.shape == (2,), (
        f"Bias shape should be (2,), got {final_gradient.biases.shape}"
    )


def test_convolution_gradient_transfer_aggregation():
    """Test that convolution layer gradients are properly aggregated when multiple workers contribute different values"""

    # Create a simple model
    model = Model(
        input_dimensions=(1, 3, 3),  # 1 channel, 3x3 image
        hidden=[
            Convolution2D(
                input_dimensions=(1, 3, 3),
                n_kernels=1,
                kernel_size=2,
                stride=1,
                kernel_init_fn=Convolution2D.Parameters.xavier,
            )
        ],
    )

    # Create different gradients for different workers
    gradient1 = Convolution2D.Parameters(
        weights=np.array([[[[1.0, 2.0], [3.0, 4.0]]]]),  # (1, 1, 2, 2)
        biases=np.array([5.0]),  # (1,)
    )
    gradient2 = Convolution2D.Parameters(
        weights=np.array([[[[6.0, 7.0], [8.0, 9.0]]]]),  # (1, 1, 2, 2)
        biases=np.array([10.0]),  # (1,)
    )

    # Create mocked shared memory manager for 2 workers
    gradient_n_bytes = model.parameter_n_bytes
    manager = MockedSharedMemoryManager(
        worker_count=2, gradient_n_bytes=gradient_n_bytes
    )

    # Worker 0 writes gradient1
    model.grad_layers[0].cache["dP"] = gradient1
    manager.worker_put_result(0, model.grad_layers)

    # Worker 1 writes gradient2
    model.grad_layers[0].cache["dP"] = gradient2
    manager.worker_put_result(1, model.grad_layers)

    # Clear gradients before aggregation
    for layer in model.grad_layers:
        layer.cache["dP"] = None

    # Aggregate results
    manager.leader_get_aggregated_results(model)

    # Verify aggregation: should be (gradient1 + gradient2) / 2
    final_gradient = model.grad_layers[0].cache["dP"]
    expected_weights = (
        gradient1.weights + gradient2.weights
    ) / 2  # [[[3.5, 4.5], [5.5, 6.5]]]
    expected_biases = (gradient1.biases + gradient2.biases) / 2  # [7.5]

    np.testing.assert_allclose(final_gradient.weights, expected_weights, rtol=1e-10)
    np.testing.assert_allclose(final_gradient.biases, expected_biases, rtol=1e-10)


@pytest.mark.parametrize("write_count", [1, 2])
def test_batch_norm_gradient_transfer(write_count: int):
    """Test batch norm layer gradient serialization and deserialization with mocked shared memory"""

    # Create a simple model with one BatchNorm layer
    model = Model(
        input_dimensions=(4,),  # 4 features
        hidden=[
            BatchNorm(
                input_dimensions=(4,),
                training=True,
            )
        ],
    )

    # Create some test data (batch_size=3, features=4)
    X = np.random.rand(3, 4)
    Y = np.random.rand(3, 4)  # Same shape as input since BatchNorm preserves dimensions

    # Do forward and backward pass to populate gradients
    model.forward_prop(X)
    model.backward_prop(Y)
    logger.debug(
        f"Model gradients shape: gamma={model.grad_layers[0].cache['dP']._gamma.shape}, beta={model.grad_layers[0].cache['dP']._beta.shape}"
    )

    # Store the original gradient for comparison
    original_gradient = model.grad_layers[0].cache["dP"]
    assert original_gradient is not None, (
        "Gradient should be populated after backward pass"
    )

    # Create mocked shared memory manager
    gradient_n_bytes = model.parameter_n_bytes
    manager = MockedSharedMemoryManager(
        worker_count=write_count, gradient_n_bytes=gradient_n_bytes
    )

    # Clear the model's gradients to test deserialization
    for layer in model.grad_layers:
        layer.cache["dP"] = None

    # Simulate workers writing gradients
    for worker_id in range(write_count):
        # Reset gradient to original value for each worker
        model.grad_layers[0].cache["dP"] = original_gradient
        manager.worker_put_result(worker_id, model.grad_layers)

    # Clear gradients again before aggregation
    for layer in model.grad_layers:
        layer.cache["dP"] = None

    # Simulate leader aggregating results
    manager.leader_get_aggregated_results(model)

    # Verify the gradient was correctly deserialized and aggregated
    final_gradient = model.grad_layers[0].cache["dP"]
    assert final_gradient is not None, "Gradient should be populated after aggregation"

    # The final gradient should be the original gradient (since we average across workers)
    # For write_count=1: final = original/1 = original
    # For write_count=2: final = (original + original)/2 = original
    np.testing.assert_allclose(
        final_gradient._gamma, original_gradient._gamma, rtol=1e-10
    )
    np.testing.assert_allclose(
        final_gradient._beta, original_gradient._beta, rtol=1e-10
    )

    # Verify the shapes are correct
    assert final_gradient._gamma.shape == (4,), (
        f"Gamma shape should be (4,), got {final_gradient._gamma.shape}"
    )
    assert final_gradient._beta.shape == (4,), (
        f"Beta shape should be (4,), got {final_gradient._beta.shape}"
    )


def test_batch_norm_gradient_transfer_aggregation():
    """Test that batch norm layer gradients are properly aggregated when multiple workers contribute different values"""

    # Create a simple model
    model = Model(
        input_dimensions=(3,),  # 3 features
        hidden=[
            BatchNorm(
                input_dimensions=(3,),
                training=True,
            )
        ],
    )

    # Create different gradients for different workers
    gradient1 = BatchNorm.Parameters(
        _gamma=np.array([1.0, 2.0, 3.0]),  # (3,)
        _beta=np.array([4.0, 5.0, 6.0]),  # (3,)
    )
    gradient2 = BatchNorm.Parameters(
        _gamma=np.array([7.0, 8.0, 9.0]),  # (3,)
        _beta=np.array([10.0, 11.0, 12.0]),  # (3,)
    )

    # Create mocked shared memory manager for 2 workers
    gradient_n_bytes = model.parameter_n_bytes
    manager = MockedSharedMemoryManager(
        worker_count=2, gradient_n_bytes=gradient_n_bytes
    )

    # Worker 0 writes gradient1
    model.grad_layers[0].cache["dP"] = gradient1
    manager.worker_put_result(0, model.grad_layers)

    # Worker 1 writes gradient2
    model.grad_layers[0].cache["dP"] = gradient2
    manager.worker_put_result(1, model.grad_layers)

    # Clear gradients before aggregation
    for layer in model.grad_layers:
        layer.cache["dP"] = None

    # Aggregate results
    manager.leader_get_aggregated_results(model)

    # Verify aggregation: should be (gradient1 + gradient2) / 2
    final_gradient = model.grad_layers[0].cache["dP"]
    expected_gamma = (gradient1._gamma + gradient2._gamma) / 2  # [4.0, 5.0, 6.0]
    expected_beta = (gradient1._beta + gradient2._beta) / 2  # [7.0, 8.0, 9.0]

    np.testing.assert_allclose(final_gradient._gamma, expected_gamma, rtol=1e-10)
    np.testing.assert_allclose(final_gradient._beta, expected_beta, rtol=1e-10)
