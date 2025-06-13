from dataclasses import dataclass

import numpy as np
import pytest

from mo_net.model.layer.batch_norm import BatchNorm, BatchNorm2D, ParametersType
from mo_net.protos import Activations, Dimensions


@dataclass(frozen=True)
class ForwardPropTestCase:
    name: str
    input_dimensions: Dimensions
    input_activations: np.ndarray
    parameters: ParametersType | None
    training: bool
    running_mean: np.ndarray | None
    running_variance: np.ndarray | None
    expected_output: np.ndarray
    expected_running_mean: np.ndarray | None = None
    expected_running_variance: np.ndarray | None = None


@pytest.mark.skip(reason="Must hand calculate expected values")
@pytest.mark.parametrize(
    "test_case",
    [
        ForwardPropTestCase(
            name="simple_training_single_feature",
            input_dimensions=(1,),
            input_activations=np.array([[1.0], [2.0], [3.0]]),
            parameters=BatchNorm.Parameters(
                _gamma=np.array([2.0]), _beta=np.array([1.0])
            ),
            training=True,
            running_mean=np.array([0.0]),
            running_variance=np.array([1.0]),
            expected_output=np.array([[-1.0], [1.0], [3.0]]),
            expected_running_mean=np.array([0.2]),
            expected_running_variance=np.array([0.9667]),
        ),
        ForwardPropTestCase(
            name="simple_inference_single_feature",
            input_dimensions=(1,),
            input_activations=np.array([[1.0], [2.0], [3.0]]),
            parameters=BatchNorm.Parameters(
                _gamma=np.array([2.0]), _beta=np.array([1.0])
            ),
            training=False,
            running_mean=np.array([2.0]),
            running_variance=np.array([1.0]),
            expected_output=np.array([[-1.0], [1.0], [3.0]]),
        ),
        ForwardPropTestCase(
            name="multi_feature_training",
            input_dimensions=(2,),
            input_activations=np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]),
            parameters=BatchNorm.Parameters(
                _gamma=np.array([1.0, 2.0]), _beta=np.array([0.0, 1.0])
            ),
            training=True,
            running_mean=np.array([0.0, 0.0]),
            running_variance=np.array([1.0, 1.0]),
            expected_output=np.array([[-1.225, -1.45], [0.0, 1.0], [1.225, 3.45]]),
            expected_running_mean=np.array([0.2, 0.3]),
            expected_running_variance=np.array([1.167, 1.167]),
        ),
        ForwardPropTestCase(
            name="identity_parameters",
            input_dimensions=(2,),
            input_activations=np.array([[1.0, 2.0], [3.0, 4.0]]),
            parameters=BatchNorm.Parameters(
                _gamma=np.array([1.0, 1.0]), _beta=np.array([0.0, 0.0])
            ),
            training=True,
            running_mean=np.array([0.0, 0.0]),
            running_variance=np.array([1.0, 1.0]),
            expected_output=np.array([[-1.0, -1.0], [1.0, 1.0]]),
        ),
        ForwardPropTestCase(
            name="large_batch_stability",
            input_dimensions=(1,),
            input_activations=np.random.randn(100, 1) * 10 + 5,
            parameters=BatchNorm.Parameters(
                _gamma=np.array([1.0]), _beta=np.array([0.0])
            ),
            training=True,
            running_mean=np.array([0.0]),
            running_variance=np.array([1.0]),
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
        assert np.allclose(output, test_case.expected_output, atol=1e-3), (
            f"Forward prop failed: got {output}, expected {test_case.expected_output}"
        )
    else:
        assert abs(np.mean(output)) < 0.1, (
            f"Mean should be near 0, got {np.mean(output)}"
        )
        assert abs(np.std(output) - 1.0) < 0.1, (
            f"Std should be near 1, got {np.std(output)}"
        )

    if test_case.training and test_case.expected_running_mean is not None:
        assert np.allclose(
            layer._running_mean, test_case.expected_running_mean, atol=1e-3
        )
    if test_case.training and test_case.expected_running_variance is not None:
        assert np.allclose(
            layer._running_variance, test_case.expected_running_variance, atol=1e-3
        )


@dataclass(frozen=True)
class BackwardPropTestCase:
    name: str
    input_dimensions: Dimensions
    input_activations: np.ndarray
    parameters: ParametersType
    dZ: np.ndarray
    expected_dX: np.ndarray
    expected_d_gamma: np.ndarray
    expected_d_beta: np.ndarray


@pytest.mark.skip(reason="Must hand calculate expected values")
@pytest.mark.parametrize(
    "test_case",
    [
        BackwardPropTestCase(
            name="simple_single_feature_backward",
            input_dimensions=(1,),
            input_activations=np.array([[1.0], [2.0], [3.0]]),
            parameters=BatchNorm.Parameters(
                _gamma=np.array([2.0]), _beta=np.array([1.0])
            ),
            dZ=np.array([[1.0], [1.0], [1.0]]),
            expected_dX=np.array([[0.0], [0.0], [0.0]]),
            expected_d_gamma=np.array([0.0]),
            expected_d_beta=np.array([3.0]),
        ),
        BackwardPropTestCase(
            name="asymmetric_gradient_single_feature",
            input_dimensions=(1,),
            input_activations=np.array([[1.0], [2.0], [3.0]]),
            parameters=BatchNorm.Parameters(
                _gamma=np.array([1.0]), _beta=np.array([0.0])
            ),
            dZ=np.array([[1.0], [0.0], [-1.0]]),
            expected_dX=np.array([[0.816], [0.0], [-0.816]]),
            expected_d_gamma=np.array([0.0]),
            expected_d_beta=np.array([0.0]),
        ),
        BackwardPropTestCase(
            name="multi_feature_backward",
            input_dimensions=(2,),
            input_activations=np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]),
            parameters=BatchNorm.Parameters(
                _gamma=np.array([1.0, 2.0]), _beta=np.array([0.0, 1.0])
            ),
            dZ=np.array([[1.0, 1.0], [0.0, 0.0], [-1.0, -1.0]]),
            expected_dX=np.array([[0.816, 0.816], [0.0, 0.0], [-0.816, -0.816]]),
            expected_d_gamma=np.array([0.0, 0.0]),
            expected_d_beta=np.array([0.0, 0.0]),
        ),
        BackwardPropTestCase(
            name="identity_backward_check",
            input_dimensions=(1,),
            input_activations=np.array([[-1.0], [0.0], [1.0]]),
            parameters=BatchNorm.Parameters(
                _gamma=np.array([1.0]), _beta=np.array([0.0])
            ),
            dZ=np.array([[1.0], [1.0], [1.0]]),
            expected_dX=np.array([[0.0], [0.0], [0.0]]),
            expected_d_gamma=np.array([0.0]),
            expected_d_beta=np.array([3.0]),
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
    dX = layer.backward_prop(dZ=test_case.dZ)

    assert np.allclose(dX, test_case.expected_dX, atol=1e-3), (
        f"dX incorrect: got {dX}, expected {test_case.expected_dX}"
    )

    cached_dP = layer.cache["dP"]
    assert cached_dP is not None, "Parameter gradients not stored in cache"
    assert np.allclose(-cached_dP._gamma, test_case.expected_d_gamma, atol=1e-3), (
        f"d_gamma incorrect: got {-cached_dP._gamma}, expected {test_case.expected_d_gamma}"
    )
    assert np.allclose(-cached_dP._beta, test_case.expected_d_beta, atol=1e-3), (
        f"d_beta incorrect: got {-cached_dP._beta}, expected {test_case.expected_d_beta}"
    )


@dataclass(frozen=True)
class ParameterUpdateTestCase:
    name: str
    input_dimensions: Dimensions
    initial_parameters: ParametersType
    input_activations: np.ndarray
    dZ: np.ndarray
    expected_updated_gamma: np.ndarray
    expected_updated_beta: np.ndarray


@pytest.mark.skip(reason="Must hand calculate expected values")
@pytest.mark.parametrize(
    "test_case",
    [
        ParameterUpdateTestCase(
            name="simple_parameter_update",
            input_dimensions=(1,),
            initial_parameters=BatchNorm.Parameters(
                _gamma=np.array([1.0]), _beta=np.array([0.0])
            ),
            input_activations=np.array([[1.0], [2.0], [3.0]]),
            dZ=np.array([[0.0], [1.0], [0.0]]),
            expected_updated_gamma=np.array([1.0]),
            expected_updated_beta=np.array([-1.0]),
        ),
        ParameterUpdateTestCase(
            name="multi_feature_update",
            input_dimensions=(2,),
            initial_parameters=BatchNorm.Parameters(
                _gamma=np.array([1.0, 2.0]), _beta=np.array([0.5, -0.5])
            ),
            input_activations=np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]),
            dZ=np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]),
            expected_updated_gamma=np.array([1.0, 2.0]),
            expected_updated_beta=np.array([-2.5, -6.5]),
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

    assert np.allclose(
        layer.parameters._gamma, test_case.expected_updated_gamma, atol=1e-3
    )
    assert np.allclose(
        layer.parameters._beta, test_case.expected_updated_beta, atol=1e-3
    )
    assert layer.cache["dP"] is None


def test_batch_norm_training_vs_inference():
    """Test that training and inference modes behave differently."""
    layer_train = BatchNorm(
        input_dimensions=(2,),
        training=True,
        running_mean=np.array([1.0, 2.0]),
        running_variance=np.array([0.5, 1.5]),
    )

    layer_inference = BatchNorm(
        input_dimensions=(2,),
        training=False,
        running_mean=np.array([1.0, 2.0]),
        running_variance=np.array([0.5, 1.5]),
        parameters=layer_train.parameters,
    )

    input_data = np.array([[0.0, 1.0], [2.0, 3.0]])

    output_train = layer_train.forward_prop(input_activations=Activations(input_data))
    output_inference = layer_inference.forward_prop(
        input_activations=Activations(input_data)
    )

    assert not np.allclose(output_train, output_inference)


def test_batch_norm_running_statistics_update():
    """Test that running statistics are updated correctly during training."""
    layer = BatchNorm(
        input_dimensions=(1,),
        momentum=0.9,
        training=True,
        running_mean=np.array([0.0]),
        running_variance=np.array([1.0]),
    )

    input1 = np.array([[1.0], [2.0], [3.0]])
    layer.forward_prop(input_activations=Activations(input1))

    expected_mean_1 = 0.9 * 0.0 + 0.1 * 2.0
    expected_var_1 = 0.9 * 1.0 + 0.1 * (2.0 / 3.0)

    assert np.allclose(layer._running_mean, [expected_mean_1], atol=1e-3)
    assert np.allclose(layer._running_variance, [expected_var_1], atol=1e-3)

    input2 = np.array([[4.0], [5.0], [6.0]])
    layer.forward_prop(input_activations=Activations(input2))

    expected_mean_2 = 0.9 * expected_mean_1 + 0.1 * 5.0
    expected_var_2 = 0.9 * expected_var_1 + 0.1 * (2.0 / 3.0)

    assert np.allclose(layer._running_mean, [expected_mean_2], atol=1e-3)
    assert np.allclose(layer._running_variance, [expected_var_2], atol=1e-3)


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
        layer.backward_prop(dZ=np.array([[1.0, 1.0]]))


def test_batch_norm_error_on_update_without_gradients():
    """Test that parameter update fails without gradients."""
    layer = BatchNorm(input_dimensions=(2,))

    with pytest.raises(RuntimeError, match="Gradient is not populated during update"):
        layer.update_parameters()


def test_batch_norm_serialization_deserialization():
    """Test serialization and deserialization of BatchNorm layer."""
    original_layer = BatchNorm(
        input_dimensions=(3,),
        momentum=0.8,
        parameters=BatchNorm.Parameters(
            _gamma=np.array([1.0, 2.0, 3.0]), _beta=np.array([0.1, 0.2, 0.3])
        ),
        running_mean=np.array([0.5, 1.0, 1.5]),
        running_variance=np.array([0.8, 1.2, 1.6]),
        training=False,
    )

    deserialized_layer = original_layer.serialize().deserialize(training=False)

    assert deserialized_layer.input_dimensions == original_layer.input_dimensions
    assert deserialized_layer._momentum == original_layer._momentum
    assert np.allclose(
        deserialized_layer.parameters._gamma, original_layer.parameters._gamma
    )
    assert np.allclose(
        deserialized_layer.parameters._beta, original_layer.parameters._beta
    )
    assert np.allclose(deserialized_layer._running_mean, original_layer._running_mean)
    assert np.allclose(
        deserialized_layer._running_variance, original_layer._running_variance
    )


def test_batch_norm_numerical_stability():
    """Test BatchNorm with extreme values."""
    layer = BatchNorm(input_dimensions=(1,), training=True)

    large_input = np.array([[1e6], [1e6 + 1], [1e6 + 2]])
    output_large = layer.forward_prop(input_activations=Activations(large_input))
    assert np.isfinite(output_large).all()
    assert abs(np.mean(output_large)) < 1e-6

    small_input = np.array([[1e-6], [2e-6], [3e-6]])
    output_small = layer.forward_prop(input_activations=Activations(small_input))
    assert np.isfinite(output_small).all()


def test_batch_norm_single_sample_batch():
    """Test BatchNorm with batch size of 1."""
    layer = BatchNorm(input_dimensions=(2,), training=True)

    single_input = np.array([[1.0, 2.0]])
    output = layer.forward_prop(input_activations=Activations(single_input))

    assert np.isfinite(output).all()
    assert output.shape == (1, 2)


def test_batch_norm_gradient_flow():
    """Test that gradients flow correctly through the layer."""
    layer = BatchNorm(input_dimensions=(2,), training=True)

    input_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    layer.forward_prop(input_activations=Activations(input_data))

    dZ = np.random.randn(3, 2)
    dX = layer.backward_prop(dZ=dZ)

    assert dX.shape == input_data.shape
    assert np.isfinite(dX).all()
    assert layer.cache["dP"] is not None
    assert np.isfinite(layer.cache["dP"]._gamma).all()
    assert np.isfinite(layer.cache["dP"]._beta).all()


def test_batch_norm_empty_gradient():
    """Test empty gradient generation."""
    layer = BatchNorm(input_dimensions=(3,))
    empty_grad = layer.empty_gradient()

    assert np.allclose(empty_grad._gamma, np.zeros(3))
    assert np.allclose(empty_grad._beta, np.zeros(3))


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
        running_mean=np.array([0.0]),
        running_variance=np.array([1.0]),
    )

    input_data = np.array([[1.0], [2.0], [3.0]])
    layer.forward_prop(input_activations=Activations(input_data))

    expected_mean = momentum * 0.0 + (1 - momentum) * 2.0
    expected_var = momentum * 1.0 + (1 - momentum) * (2.0 / 3.0)

    assert np.allclose(layer._running_mean, [expected_mean], atol=1e-3)
    assert np.allclose(layer._running_variance, [expected_var], atol=1e-3)


def test_batch_norm_reinitialise():
    """Test parameter reinitialization."""
    layer = BatchNorm(
        input_dimensions=(2,),
        parameters=BatchNorm.Parameters(
            _gamma=np.array([2.0, 3.0]), _beta=np.array([1.0, -1.0])
        ),
    )

    layer.reinitialise()

    assert np.allclose(layer.parameters._gamma, [1.0, 1.0])
    assert np.allclose(layer.parameters._beta, [0.0, 0.0])


@dataclass(frozen=True)
class BatchNorm2DForwardTestCase:
    name: str
    input_dimensions: Dimensions
    input_activations: np.ndarray
    parameters: ParametersType | None
    training: bool
    running_mean: np.ndarray | None
    running_variance: np.ndarray | None
    expected_output: np.ndarray
    expected_running_mean: np.ndarray | None = None
    expected_running_variance: np.ndarray | None = None


@pytest.mark.skip(reason="Must hand calculate expected values")
@pytest.mark.parametrize(
    "test_case",
    [
        BatchNorm2DForwardTestCase(
            name="simple_2d_single_channel",
            input_dimensions=(1, 2, 2),
            input_activations=np.array([[[[1.0, 2.0], [3.0, 4.0]]]]),
            parameters=BatchNorm2D.Parameters(
                _gamma=np.array([1.0]), _beta=np.array([0.0])
            ),
            training=True,
            running_mean=np.array([0.0]),
            running_variance=np.array([1.0]),
            expected_output=np.array([[[[-1.161, -0.387], [0.387, 1.161]]]]),
            expected_running_mean=np.array([0.25]),
            expected_running_variance=np.array([0.95]),
        ),
        BatchNorm2DForwardTestCase(
            name="multi_channel_forward",
            input_dimensions=(2, 2, 2),
            input_activations=np.array(
                [
                    [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
                ]
            ),
            parameters=BatchNorm2D.Parameters(
                _gamma=np.array([1.0, 2.0]), _beta=np.array([0.0, 1.0])
            ),
            training=True,
            running_mean=np.array([0.0, 0.0]),
            running_variance=np.array([1.0, 1.0]),
            expected_output=np.array(
                [
                    [
                        [[-1.161, -0.387], [0.387, 1.161]],
                        [[-1.161, 0.226], [1.613, 3.0]],
                    ],
                ]
            ),
            expected_running_mean=np.array([0.25, 0.65]),
            expected_running_variance=np.array([0.95, 0.95]),
        ),
        BatchNorm2DForwardTestCase(
            name="batch_size_greater_than_one",
            input_dimensions=(1, 2, 2),
            input_activations=np.array(
                [
                    [[[1.0, 2.0], [3.0, 4.0]]],
                    [[[5.0, 6.0], [7.0, 8.0]]],
                ]
            ),
            parameters=BatchNorm2D.Parameters(
                _gamma=np.array([1.0]), _beta=np.array([0.0])
            ),
            training=True,
            running_mean=np.array([0.0]),
            running_variance=np.array([1.0]),
            expected_output=np.array(
                [
                    [[[-1.528, -1.091], [-0.655, -0.218]]],
                    [[[0.218, 0.655], [1.091, 1.528]]],
                ]
            ),
            expected_running_mean=np.array([0.45]),
            expected_running_variance=np.array([0.9575]),
        ),
        BatchNorm2DForwardTestCase(
            name="inference_mode",
            input_dimensions=(2, 2, 2),
            input_activations=np.array(
                [
                    [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
                ]
            ),
            parameters=BatchNorm2D.Parameters(
                _gamma=np.array([1.0, 2.0]), _beta=np.array([0.0, 1.0])
            ),
            training=False,
            running_mean=np.array([2.5, 6.5]),
            running_variance=np.array([1.25, 1.25]),
            expected_output=np.array(
                [
                    [
                        [[-1.342, -0.447], [0.447, 1.342]],
                        [[-2.683, -0.894], [0.894, 3.683]],
                    ],
                ]
            ),
        ),
    ],
    ids=lambda test_case: test_case.name,
)
def test_batch_norm_2d_forward_prop(test_case: BatchNorm2DForwardTestCase):
    layer = BatchNorm2D(
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
        assert np.allclose(output, test_case.expected_output, atol=1e-3)

    if test_case.training and test_case.expected_running_mean is not None:
        assert np.allclose(
            layer._running_mean, test_case.expected_running_mean, atol=1e-3
        )
    if test_case.training and test_case.expected_running_variance is not None:
        assert np.allclose(
            layer._running_variance, test_case.expected_running_variance, atol=1e-3
        )


@dataclass(frozen=True)
class BatchNorm2DBackwardTestCase:
    name: str
    input_dimensions: Dimensions
    input_activations: np.ndarray
    parameters: ParametersType
    dZ: np.ndarray
    expected_dX: np.ndarray
    expected_d_gamma: np.ndarray
    expected_d_beta: np.ndarray


@pytest.mark.skip(reason="Must hand calculate expected values")
@pytest.mark.parametrize(
    "test_case",
    [
        BatchNorm2DBackwardTestCase(
            name="simple_single_channel_backward",
            input_dimensions=(1, 2, 2),
            input_activations=np.array([[[[1.0, 2.0], [3.0, 4.0]]]]),
            parameters=BatchNorm2D.Parameters(
                _gamma=np.array([1.0]), _beta=np.array([0.0])
            ),
            dZ=np.array([[[[1.0, 1.0], [1.0, 1.0]]]]),
            expected_dX=np.array([[[[0.0, 0.0], [0.0, 0.0]]]]),
            expected_d_gamma=np.array([0.0]),
            expected_d_beta=np.array([4.0]),
        ),
        BatchNorm2DBackwardTestCase(
            name="multi_channel_backward",
            input_dimensions=(2, 2, 2),
            input_activations=np.array(
                [
                    [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
                ]
            ),
            parameters=BatchNorm2D.Parameters(
                _gamma=np.array([1.0, 2.0]), _beta=np.array([0.0, 0.0])
            ),
            dZ=np.array(
                [
                    [[[1.0, 0.0], [-1.0, 0.0]], [[0.0, 1.0], [0.0, -1.0]]],
                ]
            ),
            expected_dX=np.array(
                [
                    [[[0.75, -0.25], [-0.75, 0.25]], [[0.75, -0.25], [-0.75, 0.25]]],
                ]
            ),
            expected_d_gamma=np.array([0.0, 0.0]),
            expected_d_beta=np.array([0.0, 0.0]),
        ),
        BatchNorm2DBackwardTestCase(
            name="batch_size_greater_than_one_backward",
            input_dimensions=(1, 2, 2),
            input_activations=np.array(
                [
                    [[[1.0, 2.0], [3.0, 4.0]]],
                    [[[5.0, 6.0], [7.0, 8.0]]],
                ]
            ),
            parameters=BatchNorm2D.Parameters(
                _gamma=np.array([1.0]), _beta=np.array([0.0])
            ),
            dZ=np.array(
                [
                    [[[1.0, 1.0], [1.0, 1.0]]],
                    [[[1.0, 1.0], [1.0, 1.0]]],
                ]
            ),
            expected_dX=np.array(
                [
                    [[[0.0, 0.0], [0.0, 0.0]]],
                    [[[0.0, 0.0], [0.0, 0.0]]],
                ]
            ),
            expected_d_gamma=np.array([0.0]),
            expected_d_beta=np.array([8.0]),
        ),
    ],
    ids=lambda test_case: test_case.name,
)
def test_batch_norm_2d_backward_prop(test_case: BatchNorm2DBackwardTestCase):
    layer = BatchNorm2D(
        input_dimensions=test_case.input_dimensions,
        parameters=test_case.parameters,
        training=True,
    )

    layer.forward_prop(input_activations=Activations(test_case.input_activations))
    dX = layer.backward_prop(dZ=test_case.dZ)

    assert np.allclose(dX, test_case.expected_dX, atol=1e-3)

    cached_dP = layer.cache["dP"]
    assert cached_dP is not None
    assert np.allclose(-cached_dP._gamma, test_case.expected_d_gamma, atol=1e-3)
    assert np.allclose(-cached_dP._beta, test_case.expected_d_beta, atol=1e-3)


def test_batch_norm_2d_parameter_count():
    layer = BatchNorm2D(input_dimensions=(3, 4, 4))
    assert layer.parameter_count == 6


def test_batch_norm_2d_empty_parameters():
    layer = BatchNorm2D(input_dimensions=(3, 4, 4))
    empty_params = layer.empty_parameters()

    assert np.allclose(empty_params._gamma, np.ones(3))
    assert np.allclose(empty_params._beta, np.zeros(3))



def test_batch_norm_2d_initialization_validation():
    with pytest.raises(ValueError, match="BatchNorm2D expects 3D input dimensions"):
        BatchNorm2D(input_dimensions=(2,))

    with pytest.raises(ValueError, match="BatchNorm2D expects 3D input dimensions"):
        BatchNorm2D(input_dimensions=(2, 3))


def test_batch_norm_2d_with_convolution_output():
    batch_size = 2
    channels = 3
    height = 4
    width = 4

    # Create input that looks like a convolution output
    input_data = np.random.randn(batch_size, channels, height, width)

    layer = BatchNorm2D(input_dimensions=(channels, height, width), training=True)

    output = layer.forward_prop(input_activations=Activations(input_data))

    # Output should maintain shape
    assert output.shape == (batch_size, channels, height, width)

    # Check statistics per channel
    for c in range(channels):
        channel_data = output[:, c, :, :]
        assert np.abs(np.mean(channel_data)) < 0.1
        assert np.abs(np.std(channel_data) - 1.0) < 0.1
