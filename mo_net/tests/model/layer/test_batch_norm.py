from dataclasses import dataclass

import jax.numpy as jnp
import jax.random as random
import pytest

from mo_net.model.layer.batch_norm.batch_norm import BatchNorm, ParametersType
from mo_net.protos import Activations, Dimensions

# Create a random key for testing
key = random.PRNGKey(42)


@dataclass(frozen=True)
class ForwardPropTestCase:
    name: str
    input_dimensions: Dimensions
    input_activations: jnp.ndarray
    parameters: ParametersType | None
    training: bool
    running_mean: jnp.ndarray | None
    running_variance: jnp.ndarray | None
    expected_output: jnp.ndarray | None
    expected_running_mean: jnp.ndarray | None = None
    expected_running_variance: jnp.ndarray | None = None


@pytest.mark.skip(reason="Skipping batch norm tests for now")
@pytest.mark.parametrize(
    "test_case",
    [
        ForwardPropTestCase(
            name="simple_training_single_feature",
            input_dimensions=(1,),
            input_activations=jnp.array([[1.0], [2.0], [3.0]]),
            parameters=BatchNorm.Parameters(
                weights=jnp.array([2.0]), biases=jnp.array([1.0])
            ),
            training=True,
            running_mean=jnp.array([0.0]),
            running_variance=jnp.array([1.0]),
            expected_output=jnp.array([[-1.0], [1.0], [3.0]]),
            expected_running_mean=jnp.array([0.2]),
            expected_running_variance=jnp.array([0.9667]),
        ),
        ForwardPropTestCase(
            name="simple_inference_single_feature",
            input_dimensions=(1,),
            input_activations=jnp.array([[1.0], [2.0], [3.0]]),
            parameters=BatchNorm.Parameters(
                weights=jnp.array([2.0]), biases=jnp.array([1.0])
            ),
            training=False,
            running_mean=jnp.array([2.0]),
            running_variance=jnp.array([1.0]),
            expected_output=jnp.array([[-1.0], [1.0], [3.0]]),
        ),
        ForwardPropTestCase(
            name="multi_feature_training",
            input_dimensions=(2,),
            input_activations=jnp.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]),
            parameters=BatchNorm.Parameters(
                weights=jnp.array([1.0, 2.0]), biases=jnp.array([0.0, 1.0])
            ),
            training=True,
            running_mean=jnp.array([0.0, 0.0]),
            running_variance=jnp.array([1.0, 1.0]),
            expected_output=jnp.array([[-1.225, -1.45], [0.0, 1.0], [1.225, 3.45]]),
            expected_running_mean=jnp.array([0.2, 0.3]),
            expected_running_variance=jnp.array([1.167, 1.167]),
        ),
        ForwardPropTestCase(
            name="identity_parameters",
            input_dimensions=(2,),
            input_activations=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
            parameters=BatchNorm.Parameters(
                weights=jnp.array([1.0, 1.0]), biases=jnp.array([0.0, 0.0])
            ),
            training=True,
            running_mean=jnp.array([0.0, 0.0]),
            running_variance=jnp.array([1.0, 1.0]),
            expected_output=jnp.array([[-1.0, -1.0], [1.0, 1.0]]),
        ),
        ForwardPropTestCase(
            name="large_batch_stability",
            input_dimensions=(1,),
            input_activations=random.normal(key, (100, 1)) * 10 + 5,
            parameters=BatchNorm.Parameters(
                weights=jnp.array([1.0]), biases=jnp.array([0.0])
            ),
            training=True,
            running_mean=jnp.array([0.0]),
            running_variance=jnp.array([1.0]),
            expected_output=None,
        ),
    ],
    ids=lambda test_case: test_case.name,
)
def test_batch_norm_forward_prop(test_case: ForwardPropTestCase):
    """Test forward propagation through BatchNorm layer."""
    layer = BatchNorm(
        input_dimensions=test_case.input_dimensions,
        parameters=test_case.parameters,
        training=test_case.training,
        running_mean=test_case.running_mean,
        running_variance=test_case.running_variance,
    )

    output = layer.forward_prop(
        input_activations=Activations(test_case.input_activations)
    )

    if test_case.expected_output is not None:
        assert jnp.allclose(output, test_case.expected_output, atol=1e-3), (
            f"Forward prop failed: got {output}, expected {test_case.expected_output}"
        )
    else:
        assert abs(jnp.mean(output)) < 0.1, (
            f"Mean should be near 0, got {jnp.mean(output)}"
        )
        assert abs(jnp.std(output) - 1.0) < 0.1, (
            f"Std should be near 1, got {jnp.std(output)}"
        )

    if test_case.training and test_case.expected_running_mean is not None:
        assert jnp.allclose(
            layer._running_mean, test_case.expected_running_mean, atol=1e-3
        )
    if test_case.training and test_case.expected_running_variance is not None:
        assert jnp.allclose(
            layer._running_variance, test_case.expected_running_variance, atol=1e-3
        )


@dataclass(frozen=True)
class BackwardPropTestCase:
    name: str
    input_dimensions: Dimensions
    input_activations: jnp.ndarray
    parameters: ParametersType
    dZ: jnp.ndarray
    expected_dX: jnp.ndarray
    expected_weights: jnp.ndarray
    expected_biases: jnp.ndarray


@pytest.mark.skip(reason="Skipping batch norm tests for now")
@pytest.mark.parametrize(
    "test_case",
    [
        BackwardPropTestCase(
            name="simple_single_feature_backward",
            input_dimensions=(1,),
            input_activations=jnp.array([[1.0], [2.0], [3.0]]),
            parameters=BatchNorm.Parameters(
                weights=jnp.array([2.0]), biases=jnp.array([1.0])
            ),
            dZ=jnp.array([[1.0], [1.0], [1.0]]),
            expected_dX=jnp.array([[0.0], [0.0], [0.0]]),
            expected_weights=jnp.array([0.0]),
            expected_biases=jnp.array([3.0]),
        ),
        BackwardPropTestCase(
            name="asymmetric_gradient_single_feature",
            input_dimensions=(1,),
            input_activations=jnp.array([[1.0], [2.0], [3.0]]),
            parameters=BatchNorm.Parameters(
                weights=jnp.array([1.0]), biases=jnp.array([0.0])
            ),
            dZ=jnp.array([[1.0], [0.0], [-1.0]]),
            expected_dX=jnp.array([[0.816], [0.0], [-0.816]]),
            expected_weights=jnp.array([0.0]),
            expected_biases=jnp.array([0.0]),
        ),
        BackwardPropTestCase(
            name="multi_feature_backward",
            input_dimensions=(2,),
            input_activations=jnp.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]),
            parameters=BatchNorm.Parameters(
                weights=jnp.array([1.0, 2.0]), biases=jnp.array([0.0, 1.0])
            ),
            dZ=jnp.array([[1.0, 1.0], [0.0, 0.0], [-1.0, -1.0]]),
            expected_dX=jnp.array([[0.816, 0.816], [0.0, 0.0], [-0.816, -0.816]]),
            expected_weights=jnp.array([0.0, 0.0]),
            expected_biases=jnp.array([0.0, 0.0]),
        ),
        BackwardPropTestCase(
            name="identity_backward_check",
            input_dimensions=(1,),
            input_activations=jnp.array([[-1.0], [0.0], [1.0]]),
            parameters=BatchNorm.Parameters(
                weights=jnp.array([1.0]), biases=jnp.array([0.0])
            ),
            dZ=jnp.array([[1.0], [1.0], [1.0]]),
            expected_dX=jnp.array([[0.0], [0.0], [0.0]]),
            expected_weights=jnp.array([0.0]),
            expected_biases=jnp.array([3.0]),
        ),
    ],
    ids=lambda test_case: test_case.name,
)
def test_batch_norm_backward_prop(test_case: BackwardPropTestCase):
    """Test backward propagation through BatchNorm layer."""
    layer = BatchNorm(
        input_dimensions=test_case.input_dimensions,
        parameters=test_case.parameters,
        training=True,
    )

    layer.forward_prop(input_activations=Activations(test_case.input_activations))

    # Split key for different random operations
    key1, key2 = random.split(key)
    dZ = random.normal(key1, (3, 2))
    dX = layer.backward_prop(dZ=dZ)

    assert jnp.allclose(dX, test_case.expected_dX, atol=1e-3), (  # type: ignore[arg-type]
        f"dX incorrect: got {dX}, expected {test_case.expected_dX}"
    )

    cached_dP = layer.cache["dP"]
    assert cached_dP is not None, "Parameter gradients not stored in cache"
    assert jnp.allclose(-cached_dP.weights, test_case.expected_weights, atol=1e-3), (  # type: ignore[attr-defined]
        f"weights incorrect: got {-cached_dP.weights}, expected {test_case.expected_weights}"  # type: ignore[attr-defined]
    )
    assert jnp.allclose(-cached_dP.biases, test_case.expected_biases, atol=1e-3), (  # type: ignore[attr-defined]
        f"biases incorrect: got {-cached_dP.biases}, expected {test_case.expected_biases}"  # type: ignore[attr-defined]
    )


@dataclass(frozen=True)
class ParameterUpdateTestCase:
    name: str
    input_dimensions: Dimensions
    initial_parameters: ParametersType
    input_activations: jnp.ndarray
    dZ: jnp.ndarray
    expected_updatedweights: jnp.ndarray
    expected_updatedbiases: jnp.ndarray


@pytest.mark.parametrize(
    "test_case",
    [
        ParameterUpdateTestCase(
            name="simple_parameter_update",
            input_dimensions=(1,),
            initial_parameters=BatchNorm.Parameters(
                weights=jnp.array([1.0]), biases=jnp.array([0.0])
            ),
            input_activations=jnp.array([[1.0], [2.0], [3.0]]),
            dZ=jnp.array([[0.0], [1.0], [0.0]]),
            expected_updatedweights=jnp.array([1.0]),
            expected_updatedbiases=jnp.array([-1.0]),
        ),
        ParameterUpdateTestCase(
            name="multi_feature_update",
            input_dimensions=(2,),
            initial_parameters=BatchNorm.Parameters(
                weights=jnp.array([1.0, 2.0]), biases=jnp.array([0.5, -0.5])
            ),
            input_activations=jnp.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]),
            dZ=jnp.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]),
            expected_updatedweights=jnp.array([1.0, 2.0]),
            expected_updatedbiases=jnp.array([-2.5, -6.5]),
        ),
    ],
    ids=lambda test_case: test_case.name,
)
def test_batch_norm_parameter_update(test_case: ParameterUpdateTestCase):
    """Test parameter updates in BatchNorm layer."""
    layer = BatchNorm(
        input_dimensions=test_case.input_dimensions,
        parameters=test_case.initial_parameters,
        training=True,
    )

    layer.forward_prop(input_activations=Activations(test_case.input_activations))
    layer.backward_prop(dZ=test_case.dZ)
    layer.update_parameters()

    assert jnp.allclose(
        layer.parameters.weights, test_case.expected_updatedweights, atol=1e-3
    )
    assert jnp.allclose(
        layer.parameters.biases, test_case.expected_updatedbiases, atol=1e-3
    )
    assert layer.cache["dP"] is None


def test_batch_norm_training_vs_inference():
    """Test that training and inference modes behave differently."""
    layer_train = BatchNorm(
        input_dimensions=(2,),
        training=True,
        running_mean=jnp.array([1.0, 2.0]),
        running_variance=jnp.array([0.5, 1.5]),
    )

    layer_inference = BatchNorm(
        input_dimensions=(2,),
        training=False,
        running_mean=jnp.array([1.0, 2.0]),
        running_variance=jnp.array([0.5, 1.5]),
        parameters=layer_train.parameters,
    )

    input_data = jnp.array([[0.0, 1.0], [2.0, 3.0]])

    output_train = layer_train.forward_prop(input_activations=Activations(input_data))
    output_inference = layer_inference.forward_prop(
        input_activations=Activations(input_data)
    )

    assert not jnp.allclose(output_train, output_inference)


def test_batch_norm_running_statistics_update():
    """Test that running statistics are updated correctly during training."""
    layer = BatchNorm(
        input_dimensions=(1,),
        momentum=0.9,
        training=True,
        running_mean=jnp.array([0.0]),
        running_variance=jnp.array([1.0]),
    )

    input1 = jnp.array([[1.0], [2.0], [3.0]])
    layer.forward_prop(input_activations=Activations(input1))

    expected_mean_1 = 0.9 * 0.0 + 0.1 * 2.0
    expected_var_1 = 0.9 * 1.0 + 0.1 * (2.0 / 3.0)

    assert jnp.allclose(layer._running_mean, jnp.array([expected_mean_1]), atol=1e-3)
    assert jnp.allclose(layer._running_variance, jnp.array([expected_var_1]), atol=1e-3)

    input2 = jnp.array([[4.0], [5.0], [6.0]])
    layer.forward_prop(input_activations=Activations(input2))

    expected_mean_2 = 0.9 * expected_mean_1 + 0.1 * 5.0
    expected_var_2 = 0.9 * expected_var_1 + 0.1 * (2.0 / 3.0)

    assert jnp.allclose(layer._running_mean, jnp.array([expected_mean_2]), atol=1e-3)
    assert jnp.allclose(layer._running_variance, jnp.array([expected_var_2]), atol=1e-3)


def test_batch_norm_cache_initialization():
    """Test that cache is properly initialized."""
    layer = BatchNorm(input_dimensions=(3,))

    assert layer.cache["input_activations"] is None
    assert layer.cache["output_activations"] is None
    assert layer.cache["mean"] is None
    assert layer.cache["var"] is None
    assert layer.cache["batch_size"] is None
    assert layer.cache["dP"] is None


def test_batch_norm_error_on_backward_without_forward():
    """Test that backward prop fails without forward prop in training mode."""
    layer = BatchNorm(input_dimensions=(2,), training=True)

    with pytest.raises(
        RuntimeError, match="Mean is not populated during backward pass"
    ):
        layer.backward_prop(dZ=jnp.array([[1.0, 1.0]]))


def test_batch_norm_error_on_update_without_gradients():
    """Test that parameter update fails without gradients."""
    layer = BatchNorm(input_dimensions=(2,))

    with pytest.raises(RuntimeError, match="Gradient is not populated during update"):
        layer.update_parameters()


@pytest.mark.skip(reason="Skipping batch norm tests for now")
def test_batch_norm_serialization_deserialization():
    """Test serialization and deserialization of BatchNorm layer."""
    original_layer = BatchNorm(
        input_dimensions=(3,),
        momentum=0.8,
        parameters=BatchNorm.Parameters(
            weights=jnp.array([1.0, 2.0, 3.0]), biases=jnp.array([0.1, 0.2, 0.3])
        ),
        running_mean=jnp.array([0.5, 1.0, 1.5]),
        running_variance=jnp.array([0.8, 1.2, 1.6]),
        training=False,
    )

    deserialized_layer = original_layer.serialize().deserialize(training=False)

    assert deserialized_layer.input_dimensions == original_layer.input_dimensions
    assert deserialized_layer._momentum == original_layer._momentum
    assert jnp.allclose(
        deserialized_layer.parameters.weights, original_layer.parameters.weights
    )
    assert jnp.allclose(
        deserialized_layer.parameters.biases, original_layer.parameters.biases
    )
    assert jnp.allclose(deserialized_layer._running_mean, original_layer._running_mean)
    assert jnp.allclose(
        deserialized_layer._running_variance, original_layer._running_variance
    )


def test_batch_norm_numerical_stability():
    """Test BatchNorm with extreme values."""
    layer = BatchNorm(input_dimensions=(1,), training=True)

    large_input = jnp.array([[1e6], [1e6 + 1], [1e6 + 2]])
    output_large = layer.forward_prop(input_activations=Activations(large_input))
    assert jnp.isfinite(output_large).all()
    assert abs(jnp.mean(output_large)) < 1e-6

    small_input = jnp.array([[1e-6], [2e-6], [3e-6]])
    output_small = layer.forward_prop(input_activations=Activations(small_input))
    assert jnp.isfinite(output_small).all()


def test_batch_norm_single_sample_batch():
    """Test BatchNorm with batch size of 1."""
    layer = BatchNorm(input_dimensions=(2,), training=True)

    single_input = jnp.array([[1.0, 2.0]])
    output = layer.forward_prop(input_activations=Activations(single_input))

    assert jnp.isfinite(output).all()
    assert output.shape == (1, 2)


def test_batch_norm_gradient_flow():
    """Test that gradients flow correctly through the layer."""
    layer = BatchNorm(input_dimensions=(2,), training=True)

    input_data = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    layer.forward_prop(input_activations=Activations(input_data))

    # Split key for different random operations
    key1, _ = random.split(key)
    dZ = random.normal(key1, (3, 2))
    dX = layer.backward_prop(dZ=dZ)

    assert dX.shape == input_data.shape
    assert jnp.isfinite(dX).all()
    assert layer.cache["dP"] is not None
    assert jnp.isfinite(layer.cache["dP"].weights).all()
    assert jnp.isfinite(layer.cache["dP"].biases).all()


def test_batch_norm_empty_gradient():
    """Test empty gradient generation."""
    layer = BatchNorm(input_dimensions=(3,))
    empty_grad = layer.empty_gradient()

    assert jnp.allclose(empty_grad.weights, jnp.zeros(3))
    assert jnp.allclose(empty_grad.biases, jnp.zeros(3))


def test_batch_norm_parameter_count():
    """Test parameter count calculation."""
    layer = BatchNorm(input_dimensions=(5,))
    assert layer.parameter_count == 10


@pytest.mark.parametrize("momentum", [0.1, 0.5, 0.9, 0.99])
def test_batch_norm_different_momentum_values(momentum):
    """Test BatchNorm with different momentum values."""
    layer = BatchNorm(
        input_dimensions=(1,),
        momentum=momentum,
        training=True,
        running_mean=jnp.array([0.0]),
        running_variance=jnp.array([1.0]),
    )

    input_data = jnp.array([[1.0], [2.0], [3.0]])
    layer.forward_prop(input_activations=Activations(input_data))

    expected_mean = momentum * 0.0 + (1 - momentum) * 2.0
    expected_var = momentum * 1.0 + (1 - momentum) * (2.0 / 3.0)

    assert jnp.allclose(layer._running_mean, jnp.array([expected_mean]), atol=1e-3)
    assert jnp.allclose(layer._running_variance, jnp.array([expected_var]), atol=1e-3)


def test_batch_norm_reinitialise():
    """Test parameter reinitialization."""
    layer = BatchNorm(
        input_dimensions=(2,),
        parameters=BatchNorm.Parameters(
            weights=jnp.array([2.0, 3.0]), biases=jnp.array([1.0, -1.0])
        ),
    )

    layer.reinitialise()

    assert jnp.allclose(layer.parameters.weights, jnp.array([1.0, 1.0]))
    assert jnp.allclose(layer.parameters.biases, jnp.array([0.0, 0.0]))
