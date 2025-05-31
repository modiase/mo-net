from dataclasses import dataclass

import numpy as np
import pytest

from mnist_numpy.model.layer.linear import Linear
from mnist_numpy.protos import Activations, Dimensions


@dataclass(frozen=True)
class ForwardPropTestCase:
    name: str
    input_dimensions: Dimensions
    output_dimensions: Dimensions
    parameters: Linear.Parameters
    input_activations: np.ndarray
    expected_output: np.ndarray


@pytest.mark.parametrize(
    "test_case",
    [
        ForwardPropTestCase(
            name="identity_matrix_single_dimension",
            input_dimensions=(2,),
            output_dimensions=(2,),
            parameters=Linear.Parameters(_W=np.eye(2), _B=np.zeros(2)),
            input_activations=np.array([[1.0, 2.0]]),
            expected_output=np.array([[1.0, 2.0]]),
        ),
        ForwardPropTestCase(
            name="simple_weights_and_bias",
            input_dimensions=(2,),
            output_dimensions=(1,),
            parameters=Linear.Parameters(
                _W=np.array([[1.0], [2.0]]), _B=np.array([3.0])
            ),
            input_activations=np.array([[1.0, 2.0]]),
            expected_output=np.array([[8.0]]),
        ),
        ForwardPropTestCase(
            name="batch_processing",
            input_dimensions=(2,),
            output_dimensions=(2,),
            parameters=Linear.Parameters(
                _W=np.array([[1.0, 0.0], [0.0, 2.0]]), _B=np.array([1.0, -1.0])
            ),
            input_activations=np.array([[1.0, 2.0], [3.0, 4.0]]),
            expected_output=np.array([[2.0, 3.0], [4.0, 7.0]]),
        ),
        ForwardPropTestCase(
            name="dimension_expansion",
            input_dimensions=(1,),
            output_dimensions=(3,),
            parameters=Linear.Parameters(
                _W=np.array([[1.0, 2.0, 3.0]]), _B=np.array([0.5, -0.5, 1.0])
            ),
            input_activations=np.array([[2.0]]),
            expected_output=np.array([[2.5, 3.5, 7.0]]),
        ),
        ForwardPropTestCase(
            name="dimension_reduction",
            input_dimensions=(3,),
            output_dimensions=(1,),
            parameters=Linear.Parameters(
                _W=np.array([[1.0], [2.0], [3.0]]), _B=np.array([0.0])
            ),
            input_activations=np.array([[1.0, 2.0, 3.0]]),
            expected_output=np.array([[14.0]]),
        ),
    ],
    ids=lambda test_case: test_case.name,
)
def test_linear_forward_prop(test_case: ForwardPropTestCase):
    layer = Linear(
        input_dimensions=test_case.input_dimensions,
        output_dimensions=test_case.output_dimensions,
        parameters=test_case.parameters,
    )
    assert np.allclose(
        layer.forward_prop(input_activations=Activations(test_case.input_activations)),
        test_case.expected_output,
    )


@dataclass(frozen=True)
class BackwardPropTestCase:
    name: str
    input_dimensions: Dimensions
    output_dimensions: Dimensions
    parameters: Linear.Parameters
    input_activations: np.ndarray
    dZ: np.ndarray
    expected_dX: np.ndarray
    expected_dW: np.ndarray
    expected_dB: np.ndarray


@pytest.mark.parametrize(
    "test_case",
    [
        BackwardPropTestCase(
            name="simple_single_input_output",
            input_dimensions=(1,),
            output_dimensions=(1,),
            parameters=Linear.Parameters(_W=np.array([[2.0]]), _B=np.array([1.0])),
            input_activations=np.array([[3.0]]),
            dZ=np.array([[1.0]]),
            expected_dX=np.array([[2.0]]),
            expected_dW=np.array([[3.0]]),
            expected_dB=np.array([1.0]),
        ),
        BackwardPropTestCase(
            name="multiple_inputs_single_output",
            input_dimensions=(2,),
            output_dimensions=(1,),
            parameters=Linear.Parameters(
                _W=np.array([[1.0], [2.0]]), _B=np.array([0.0])
            ),
            input_activations=np.array([[1.0, 2.0]]),
            dZ=np.array([[1.0]]),
            expected_dX=np.array([[1.0, 2.0]]),
            expected_dW=np.array([[1.0], [2.0]]),
            expected_dB=np.array([1.0]),
        ),
        BackwardPropTestCase(
            name="single_input_multiple_outputs",
            input_dimensions=(1,),
            output_dimensions=(2,),
            parameters=Linear.Parameters(
                _W=np.array([[1.0, 2.0]]), _B=np.array([0.0, 0.0])
            ),
            input_activations=np.array([[3.0]]),
            dZ=np.array([[1.0, 1.0]]),
            expected_dX=np.array([[3.0]]),
            expected_dW=np.array([[3.0, 3.0]]),
            expected_dB=np.array([2.0]),
        ),
        BackwardPropTestCase(
            name="batch_processing_gradients",
            input_dimensions=(2,),
            output_dimensions=(2,),
            parameters=Linear.Parameters(
                _W=np.array([[1.0, 0.0], [0.0, 1.0]]), _B=np.array([0.0, 0.0])
            ),
            input_activations=np.array([[1.0, 2.0], [3.0, 4.0]]),
            dZ=np.array([[1.0, 1.0], [1.0, 1.0]]),
            expected_dX=np.array([[1.0, 1.0], [1.0, 1.0]]),
            expected_dW=np.array([[4.0, 4.0], [6.0, 6.0]]),
            expected_dB=np.array([4.0]),
        ),
    ],
    ids=lambda test_case: test_case.name,
)
def test_linear_backward_prop(test_case: BackwardPropTestCase):
    layer = Linear(
        input_dimensions=test_case.input_dimensions,
        output_dimensions=test_case.output_dimensions,
        parameters=test_case.parameters,
        clip_gradients=False,
    )
    layer.forward_prop(input_activations=Activations(test_case.input_activations))
    assert np.allclose(layer.backward_prop(dZ=test_case.dZ), test_case.expected_dX)
    assert layer.cache["dP"] is not None
    assert np.allclose(layer.cache["dP"]._W, test_case.expected_dW)
    assert np.allclose(layer.cache["dP"]._B, test_case.expected_dB)


@dataclass(frozen=True)
class ParameterUpdateTestCase:
    name: str
    input_dimensions: Dimensions
    output_dimensions: Dimensions
    initial_parameters: Linear.Parameters
    input_activations: np.ndarray
    dZ: np.ndarray
    expected_updated_W: np.ndarray
    expected_updated_B: np.ndarray


@pytest.mark.parametrize(
    "test_case",
    [
        ParameterUpdateTestCase(
            name="simple_parameter_update",
            input_dimensions=(1,),
            output_dimensions=(1,),
            initial_parameters=Linear.Parameters(
                _W=np.array([[1.0]]), _B=np.array([0.0])
            ),
            input_activations=np.array([[2.0]]),
            dZ=np.array([[1.0]]),
            expected_updated_W=np.array([[3.0]]),
            expected_updated_B=np.array([1.0]),
        ),
        ParameterUpdateTestCase(
            name="multi_dimensional_update",
            input_dimensions=(2,),
            output_dimensions=(2,),
            initial_parameters=Linear.Parameters(
                _W=np.array([[1.0, 2.0], [3.0, 4.0]]), _B=np.array([0.5, -0.5])
            ),
            input_activations=np.array([[1.0, 1.0]]),
            dZ=np.array([[1.0, 1.0]]),
            expected_updated_W=np.array([[2.0, 3.0], [4.0, 5.0]]),
            expected_updated_B=np.array([2.5, 1.5]),
        ),
    ],
    ids=lambda test_case: test_case.name,
)
def test_linear_parameter_update(test_case: ParameterUpdateTestCase):
    layer = Linear(
        input_dimensions=test_case.input_dimensions,
        output_dimensions=test_case.output_dimensions,
        parameters=test_case.initial_parameters,
        clip_gradients=False,
    )
    layer.forward_prop(input_activations=Activations(test_case.input_activations))
    layer.backward_prop(dZ=test_case.dZ)
    layer.update_parameters()

    assert np.allclose(layer.parameters._W, test_case.expected_updated_W)
    assert np.allclose(layer.parameters._B, test_case.expected_updated_B)
    assert layer.cache["dP"] is None


@pytest.fixture
def identity_layer() -> Linear:
    return Linear(
        input_dimensions=(3,),
        output_dimensions=(3,),
        parameters=Linear.Parameters.eye((3,)),
    )


@pytest.fixture
def simple_layer() -> Linear:
    return Linear(
        input_dimensions=(2,),
        output_dimensions=(2,),
        parameters=Linear.Parameters(
            _W=np.array([[2.0, 0.0], [0.0, 3.0]]), _B=np.array([1.0, -1.0])
        ),
    )


@pytest.fixture
def test_input() -> np.ndarray:
    return np.array([[1.0, 2.0, 3.0]])


def test_linear_cache_initialization(identity_layer: Linear):
    assert identity_layer.cache["input_activations"] is None
    assert identity_layer.cache["output_activations"] is None
    assert identity_layer.cache["dP"] is None


def test_linear_forward_prop_caches_input(
    identity_layer: Linear, test_input: np.ndarray
):
    identity_layer.forward_prop(input_activations=Activations(test_input))
    assert identity_layer.cache["input_activations"] is not None
    assert np.allclose(identity_layer.cache["input_activations"], test_input)


def test_linear_gradient_clipping():
    layer = Linear(
        input_dimensions=(2,),
        output_dimensions=(2,),
        parameters=Linear.Parameters(
            _W=np.array([[1.0, 0.0], [0.0, 1.0]]), _B=np.array([0.0, 0.0])
        ),
        clip_gradients=True,
        weight_max_norm=1.0,
        bias_max_norm=1.0,
    )

    layer.forward_prop(input_activations=Activations(np.array([[1.0, 1.0]])))
    layer.backward_prop(dZ=np.array([[1000.0, 1000.0]]))

    assert layer.cache["dP"] is not None
    assert (
        np.linalg.norm(layer.cache["dP"]._W) / np.sqrt(layer.cache["dP"]._W.size)
        <= 1.0 + 1e-6
    )
    assert (
        np.linalg.norm(layer.cache["dP"]._B) / np.sqrt(layer.cache["dP"]._B.size)
        <= 1.0 + 1e-6
    )


def test_linear_frozen_parameters():
    layer = Linear(
        input_dimensions=(1,),
        output_dimensions=(1,),
        parameters=Linear.Parameters(_W=np.array([[1.0]]), _B=np.array([0.0])),
        freeze_parameters=True,
    )

    original_W = layer.parameters._W.copy()
    original_B = layer.parameters._B.copy()

    layer.forward_prop(input_activations=Activations(np.array([[2.0]])))
    layer.backward_prop(dZ=np.array([[1.0]]))
    layer.update_parameters()

    assert np.allclose(layer.parameters._W, original_W)
    assert np.allclose(layer.parameters._B, original_B)


def test_linear_empty_gradient(simple_layer: Linear):
    empty_grad = simple_layer.empty_gradient()
    assert np.allclose(empty_grad._W, np.zeros_like(simple_layer.parameters._W))
    assert np.allclose(empty_grad._B, np.zeros_like(simple_layer.parameters._B))


def test_linear_parameter_count(simple_layer: Linear):
    assert (
        simple_layer.parameter_count
        == simple_layer.parameters._W.size + simple_layer.parameters._B.size
    )


def test_linear_serialization_deserialization():
    original_layer = Linear(
        input_dimensions=(2,),
        output_dimensions=(3,),
        parameters=Linear.Parameters(
            _W=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            _B=np.array([1.0, 2.0, 3.0]),
        ),
    )

    deserialized_layer = original_layer.serialize().deserialize()

    assert deserialized_layer.input_dimensions == original_layer.input_dimensions
    assert deserialized_layer.output_dimensions == original_layer.output_dimensions
    assert np.allclose(deserialized_layer.parameters._W, original_layer.parameters._W)
    assert np.allclose(deserialized_layer.parameters._B, original_layer.parameters._B)


def test_linear_error_on_backward_prop_without_forward():
    layer = Linear(input_dimensions=(2,), output_dimensions=(2,))
    with pytest.raises(
        ValueError, match="Input activations not set during forward pass"
    ):
        layer.backward_prop(dZ=np.array([[1.0, 1.0]]))


def test_linear_error_on_update_without_gradients():
    layer = Linear(input_dimensions=(2,), output_dimensions=(2,))
    with pytest.raises(ValueError, match="Gradient not set during backward pass"):
        layer.update_parameters()


@pytest.mark.parametrize(
    "init_method,input_dim,output_dim",
    [
        (Linear.Parameters.xavier, (3,), (5,)),
        (Linear.Parameters.he, (4,), (2,)),
        (Linear.Parameters.random, (2,), (3,)),
    ],
)
def test_linear_initialization_methods(init_method, input_dim, output_dim):
    layer = Linear(
        input_dimensions=input_dim,
        output_dimensions=output_dim,
        parameters_init_fn=init_method,
    )

    assert layer.parameters._W.shape == (input_dim[0], output_dim[0])
    assert layer.parameters._B.shape == (output_dim[0],)
    assert not np.allclose(layer.parameters._W, 0)


def test_linear_mathematical_properties():
    W = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    layer = Linear(
        input_dimensions=(3,),
        output_dimensions=(2,),
        parameters=Linear.Parameters(_W=W, _B=np.array([0.0, 0.0])),
    )

    x1, x2 = np.array([[1.0, 2.0, 3.0]]), np.array([[4.0, 5.0, 6.0]])
    a, b = 2.0, 3.0

    result_combined = layer.forward_prop(input_activations=Activations(a * x1 + b * x2))
    result_x1 = layer.forward_prop(input_activations=Activations(x1))
    result_x2 = layer.forward_prop(input_activations=Activations(x2))

    assert np.allclose(result_combined, a * result_x1 + b * result_x2)

    layer_with_bias = Linear(
        input_dimensions=(3,),
        output_dimensions=(2,),
        parameters=Linear.Parameters(_W=W, _B=np.array([1.0, -1.0])),
    )

    x = np.array([[1.0, 2.0, 3.0]])
    assert np.allclose(
        layer_with_bias.forward_prop(input_activations=Activations(x)),
        x @ W + np.array([1.0, -1.0]),
    )


def test_linear_zero_input():
    layer = Linear(
        input_dimensions=(3,),
        output_dimensions=(2,),
        parameters=Linear.Parameters(
            _W=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), _B=np.array([1.0, -1.0])
        ),
    )

    assert np.allclose(
        layer.forward_prop(input_activations=Activations(np.zeros((1, 3)))),
        layer.parameters._B,
    )


def test_linear_large_batch():
    layer = Linear(
        input_dimensions=(2,),
        output_dimensions=(3,),
        parameters=Linear.Parameters.xavier((2,), (3,)),
    )

    input_data = np.random.randn(100, 2)
    output = layer.forward_prop(input_activations=Activations(input_data))

    assert output.shape == (100, 3)

    dX = layer.backward_prop(dZ=np.random.randn(100, 3))

    assert dX.shape == (100, 2)
    assert layer.cache["dP"] is not None


def test_linear_gradient_accumulation():
    layer = Linear(
        input_dimensions=(2,),
        output_dimensions=(1,),
        parameters=Linear.Parameters(_W=np.array([[1.0], [1.0]]), _B=np.array([0.0])),
        clip_gradients=False,
    )

    layer.forward_prop(input_activations=Activations(np.array([[1.0, 1.0]])))
    layer.backward_prop(dZ=np.array([[1.0]]))
    grad1 = layer.cache["dP"]

    layer.forward_prop(input_activations=Activations(np.array([[2.0, 2.0]])))
    layer.backward_prop(dZ=np.array([[1.0]]))
    grad2 = layer.cache["dP"]

    assert not np.allclose(grad1._W, grad2._W)


def test_linear_weight_and_bias_shapes():
    layer = Linear(
        input_dimensions=(2,),
        output_dimensions=(3,),
        parameters=Linear.Parameters(_W=np.random.randn(2, 3), _B=np.random.randn(3)),
    )
    assert layer.parameters._W.shape == (2, 3)
    assert layer.parameters._B.shape == (3,)

    with pytest.raises(ValueError, match="Weight matrix shape"):
        Linear(
            input_dimensions=(2,),
            output_dimensions=(3,),
            parameters=Linear.Parameters(
                _W=np.random.randn(3, 2), _B=np.random.randn(3)
            ),
        )

    with pytest.raises(ValueError, match="Bias vector shape"):
        Linear(
            input_dimensions=(2,),
            output_dimensions=(3,),
            parameters=Linear.Parameters(
                _W=np.random.randn(2, 3), _B=np.random.randn(2)
            ),
        )


def test_linear_numerical_stability():
    layer = Linear(
        input_dimensions=(2,),
        output_dimensions=(2,),
        parameters=Linear.Parameters(
            _W=np.array([[1e-8, 1e8], [1e8, 1e-8]]), _B=np.array([1e-8, 1e8])
        ),
        clip_gradients=False,
    )

    assert np.isfinite(
        layer.forward_prop(input_activations=Activations(np.array([[1e-8, 1e-8]])))
    ).all()
    assert np.isfinite(
        layer.forward_prop(input_activations=Activations(np.array([[1e8, 1e8]])))
    ).all()


@pytest.mark.parametrize("store_output", [True, False])
def test_linear_output_storage_option(store_output):
    layer = Linear(
        input_dimensions=(2,),
        output_dimensions=(2,),
        store_output_activations=store_output,
    )

    output = layer.forward_prop(input_activations=Activations(np.array([[1.0, 2.0]])))

    if store_output:
        assert layer.cache["output_activations"] is not None
        assert np.allclose(layer.cache["output_activations"], output)
    else:
        assert layer.cache["output_activations"] is None


def test_linear_constructor_edge_cases():
    layer = Linear(input_dimensions=(3,))
    assert layer.input_dimensions == (3,)
    assert layer.output_dimensions == (3,)

    bias_layer = Linear.of_bias(dim=(2,), bias=5.0)
    assert np.allclose(bias_layer.parameters._W, 0)
    assert np.allclose(bias_layer.parameters._B, 5.0)

    identity_layer = Linear.of_eye(dim=(3,))
    assert np.allclose(identity_layer.parameters._W, np.eye(3))
    assert np.allclose(identity_layer.parameters._B, 0)


def test_linear_gradient_operation_interface():
    layer = Linear(input_dimensions=(2,), output_dimensions=(2,))

    called = False

    def grad_callback(grad_layer):
        nonlocal called
        called = True
        assert grad_layer is layer

    layer.gradient_operation(grad_callback)
    assert called
