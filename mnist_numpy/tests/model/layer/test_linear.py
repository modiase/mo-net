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
            expected_output=np.array([[8.0]]),  # 1*1 + 2*2 + 3 = 8
        ),
        ForwardPropTestCase(
            name="batch_processing",
            input_dimensions=(2,),
            output_dimensions=(2,),
            parameters=Linear.Parameters(
                _W=np.array([[1.0, 0.0], [0.0, 2.0]]), _B=np.array([1.0, -1.0])
            ),
            input_activations=np.array([[1.0, 2.0], [3.0, 4.0]]),
            expected_output=np.array(
                [[2.0, 3.0], [4.0, 7.0]]
            ),  # [1*1+1, 2*2-1], [3*1+1, 4*2-1]
        ),
        ForwardPropTestCase(
            name="dimension_expansion",
            input_dimensions=(1,),
            output_dimensions=(3,),
            parameters=Linear.Parameters(
                _W=np.array([[1.0, 2.0, 3.0]]), _B=np.array([0.5, -0.5, 1.0])
            ),
            input_activations=np.array([[2.0]]),
            expected_output=np.array([[2.5, 3.5, 7.0]]),  # [2*1+0.5, 2*2-0.5, 2*3+1]
        ),
        ForwardPropTestCase(
            name="dimension_reduction",
            input_dimensions=(3,),
            output_dimensions=(1,),
            parameters=Linear.Parameters(
                _W=np.array([[1.0], [2.0], [3.0]]), _B=np.array([0.0])
            ),
            input_activations=np.array([[1.0, 2.0, 3.0]]),
            expected_output=np.array([[14.0]]),  # 1*1 + 2*2 + 3*3 = 14
        ),
    ],
    ids=lambda test_case: test_case.name,
)
def test_linear_forward_prop(test_case: ForwardPropTestCase):
    """Test forward propagation through Linear layer."""
    layer = Linear(
        input_dimensions=test_case.input_dimensions,
        output_dimensions=test_case.output_dimensions,
        parameters=test_case.parameters,
    )
    output = layer.forward_prop(
        input_activations=Activations(test_case.input_activations)
    )
    assert np.allclose(output, test_case.expected_output)


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
            expected_dX=np.array([[2.0]]),  # dZ @ W.T = 1 * 2
            expected_dW=np.array([[3.0]]),  # X.T @ dZ = 3 * 1
            expected_dB=np.array([1.0]),  # sum(dZ) = 1
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
            expected_dX=np.array([[1.0, 2.0]]),  # [1] @ [[1, 2]] = [1, 2]
            expected_dW=np.array([[1.0], [2.0]]),  # [[1], [2]] @ [1] = [[1], [2]]
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
            expected_dX=np.array([[3.0]]),  # [1, 1] @ [[1], [2]] = [3]
            expected_dW=np.array([[3.0, 3.0]]),  # [3] @ [1, 1] = [[3, 3]]
            expected_dB=np.array([1.0, 1.0]),
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
            expected_dX=np.array([[1.0, 1.0], [1.0, 1.0]]),  # Identity matrix
            expected_dW=np.array(
                [[4.0, 4.0], [6.0, 6.0]]
            ),  # Sum over batch: [1+3, 1+3], [2+4, 2+4]
            expected_dB=np.array([2.0, 2.0]),  # Sum over batch dimension
        ),
    ],
    ids=lambda test_case: test_case.name,
)
def test_linear_backward_prop(test_case: BackwardPropTestCase):
    """Test backward propagation through Linear layer."""
    layer = Linear(
        input_dimensions=test_case.input_dimensions,
        output_dimensions=test_case.output_dimensions,
        parameters=test_case.parameters,
        clip_gradients=False,  # Disable clipping for exact testing
    )
    layer.forward_prop(input_activations=Activations(test_case.input_activations))
    dX = layer.backward_prop(dZ=test_case.dZ)

    assert np.allclose(dX, test_case.expected_dX)
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
            expected_updated_W=np.array([[3.0]]),  # 1.0 + 2.0
            expected_updated_B=np.array([1.0]),  # 0.0 + 1.0
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
            expected_updated_W=np.array(
                [[2.0, 3.0], [4.0, 5.0]]
            ),  # Add gradient to each element
            expected_updated_B=np.array([1.5, 0.5]),  # 0.5+1.0, -0.5+1.0
        ),
    ],
    ids=lambda test_case: test_case.name,
)
def test_linear_parameter_update(test_case: ParameterUpdateTestCase):
    """Test parameter updates in Linear layer."""
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
    assert layer.cache["dP"] is None  # Should be cleared after update


@pytest.fixture
def identity_layer() -> Linear:
    """Fixture for identity linear layer."""
    return Linear(
        input_dimensions=(3,),
        output_dimensions=(3,),
        parameters=Linear.Parameters.eye((3,)),
    )


@pytest.fixture
def simple_layer() -> Linear:
    """Fixture for simple 2x2 linear layer."""
    return Linear(
        input_dimensions=(2,),
        output_dimensions=(2,),
        parameters=Linear.Parameters(
            _W=np.array([[2.0, 0.0], [0.0, 3.0]]), _B=np.array([1.0, -1.0])
        ),
    )


def test_linear_cache_initialization(identity_layer: Linear):
    """Test that cache is properly initialized."""
    assert identity_layer.cache["input_activations"] is None
    assert identity_layer.cache["output_activations"] is None
    assert identity_layer.cache["dP"] is None


def test_linear_forward_prop_caches_input(identity_layer: Linear):
    """Test that forward prop stores input activations in cache."""
    input_data = np.array([[1.0, 2.0, 3.0]])
    identity_layer.forward_prop(input_activations=Activations(input_data))

    assert identity_layer.cache["input_activations"] is not None
    assert np.allclose(identity_layer.cache["input_activations"], input_data)


def test_linear_gradient_clipping(simple_layer: Linear):
    """Test gradient clipping functionality."""
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

    # Create large gradients that should be clipped
    input_data = np.array([[10.0, 10.0]])
    dZ = np.array([[10.0, 10.0]])

    layer.forward_prop(input_activations=Activations(input_data))
    layer.backward_prop(dZ=dZ)

    # Check that gradients were clipped
    dP = layer.cache["dP"]
    assert dP is not None
    assert (
        np.linalg.norm(dP._W) <= 1.0 + 1e-6
    )  # Small tolerance for numerical precision
    assert np.linalg.norm(dP._B) <= 1.0 + 1e-6


def test_linear_frozen_parameters():
    """Test that frozen parameters don't update."""
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
    """Test empty gradient creation."""
    empty_grad = simple_layer.empty_gradient()
    assert np.allclose(empty_grad._W, np.zeros_like(simple_layer.parameters._W))
    assert np.allclose(empty_grad._B, np.zeros_like(simple_layer.parameters._B))


def test_linear_parameter_count(simple_layer: Linear):
    """Test parameter count calculation."""
    expected_count = simple_layer.parameters._W.size + simple_layer.parameters._B.size
    assert simple_layer.parameter_count == expected_count


def test_linear_serialization_deserialization():
    """Test layer serialization and deserialization."""
    original_layer = Linear(
        input_dimensions=(2,),
        output_dimensions=(3,),
        parameters=Linear.Parameters(
            _W=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            _B=np.array([1.0, 2.0, 3.0]),
        ),
    )

    serialized = original_layer.serialize()
    deserialized_layer = serialized.deserialize()

    assert deserialized_layer.input_dimensions == original_layer.input_dimensions
    assert deserialized_layer.output_dimensions == original_layer.output_dimensions
    assert np.allclose(deserialized_layer.parameters._W, original_layer.parameters._W)
    assert np.allclose(deserialized_layer.parameters._B, original_layer.parameters._B)


def test_linear_error_on_backward_prop_without_forward():
    """Test that backward prop raises error if forward prop wasn't called."""
    layer = Linear(input_dimensions=(2,), output_dimensions=(2,))

    with pytest.raises(
        ValueError, match="Input activations not set during forward pass"
    ):
        layer.backward_prop(dZ=np.array([[1.0, 1.0]]))


def test_linear_error_on_update_without_gradients():
    """Test that parameter update raises error if gradients weren't computed."""
    layer = Linear(input_dimensions=(2,), output_dimensions=(2,))

    with pytest.raises(ValueError, match="Gradient not set during backward pass"):
        layer.update_parameters()
