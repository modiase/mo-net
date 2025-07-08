import io
from dataclasses import dataclass
from functools import partial

import jax.numpy as jnp
import jax.random as random
import pytest

from mo_net.model.layer.base import BadLayerId
from mo_net.model.layer.linear import Linear, ParametersType
from mo_net.protos import Activations, Dimensions

# Create a random key for testing
key = random.PRNGKey(42)


@dataclass(frozen=True)
class ForwardPropTestCase:
    name: str
    input_dimensions: Dimensions
    output_dimensions: Dimensions
    parameters: ParametersType
    input_activations: jnp.ndarray
    expected_output: jnp.ndarray


@pytest.mark.parametrize(
    "test_case",
    [
        ForwardPropTestCase(
            name="identity_matrix_single_dimension",
            input_dimensions=(2,),
            output_dimensions=(2,),
            parameters=Linear.Parameters(weights=jnp.eye(2), biases=jnp.zeros(2)),
            input_activations=jnp.array([[1.0, 2.0]]),
            expected_output=jnp.array([[1.0, 2.0]]),
        ),
        ForwardPropTestCase(
            name="simple_weights_and_bias",
            input_dimensions=(2,),
            output_dimensions=(1,),
            parameters=Linear.Parameters(
                weights=jnp.array([[1.0], [2.0]]), biases=jnp.array([3.0])
            ),
            input_activations=jnp.array([[1.0, 2.0]]),
            expected_output=jnp.array([[8.0]]),
        ),
        ForwardPropTestCase(
            name="batch_processing",
            input_dimensions=(2,),
            output_dimensions=(2,),
            parameters=Linear.Parameters(
                weights=jnp.array([[1.0, 0.0], [0.0, 2.0]]),
                biases=jnp.array([1.0, -1.0]),
            ),
            input_activations=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
            expected_output=jnp.array([[2.0, 3.0], [4.0, 7.0]]),
        ),
        ForwardPropTestCase(
            name="dimension_expansion",
            input_dimensions=(1,),
            output_dimensions=(3,),
            parameters=Linear.Parameters(
                weights=jnp.array([[1.0, 2.0, 3.0]]), biases=jnp.array([0.5, -0.5, 1.0])
            ),
            input_activations=jnp.array([[2.0]]),
            expected_output=jnp.array([[2.5, 3.5, 7.0]]),
        ),
        ForwardPropTestCase(
            name="dimension_reduction",
            input_dimensions=(3,),
            output_dimensions=(1,),
            parameters=Linear.Parameters(
                weights=jnp.array([[1.0], [2.0], [3.0]]), biases=jnp.array([0.0])
            ),
            input_activations=jnp.array([[1.0, 2.0, 3.0]]),
            expected_output=jnp.array([[14.0]]),
        ),
    ],
    ids=lambda test_case: test_case.name,
)
def test_linear_forward_prop(test_case: ForwardPropTestCase):
    layer = Linear(
        input_dimensions=test_case.input_dimensions,
        output_dimensions=test_case.output_dimensions,
        parameters_init_fn=lambda *_: test_case.parameters,
    )
    assert jnp.allclose(
        layer.forward_prop(input_activations=Activations(test_case.input_activations)),
        test_case.expected_output,
    )


@dataclass(frozen=True)
class BackwardPropTestCase:
    name: str
    input_dimensions: Dimensions
    output_dimensions: Dimensions
    parameters: ParametersType
    input_activations: jnp.ndarray
    dZ: jnp.ndarray
    expected_dX: jnp.ndarray
    expected_dW: jnp.ndarray
    expected_dB: jnp.ndarray


@pytest.mark.parametrize(
    "test_case",
    [
        BackwardPropTestCase(
            name="simple_single_input_output",
            input_dimensions=(1,),
            output_dimensions=(1,),
            parameters=Linear.Parameters(
                weights=jnp.array([[2.0]]), biases=jnp.array([1.0])
            ),
            input_activations=jnp.array([[3.0]]),
            dZ=jnp.array([[1.0]]),
            expected_dX=jnp.array([[2.0]]),
            expected_dW=jnp.array([[3.0]]),
            expected_dB=jnp.array([1.0]),
        ),
        BackwardPropTestCase(
            name="multiple_inputs_single_output",
            input_dimensions=(2,),
            output_dimensions=(1,),
            parameters=Linear.Parameters(
                weights=jnp.array([[1.0], [2.0]]), biases=jnp.array([0.0])
            ),
            input_activations=jnp.array([[1.0, 2.0]]),
            dZ=jnp.array([[1.0]]),
            expected_dX=jnp.array([[1.0, 2.0]]),
            expected_dW=jnp.array([[1.0], [2.0]]),
            expected_dB=jnp.array([1.0]),
        ),
        BackwardPropTestCase(
            name="single_input_multiple_outputs",
            input_dimensions=(1,),
            output_dimensions=(2,),
            parameters=Linear.Parameters(
                weights=jnp.array([[1.0, 2.0]]), biases=jnp.array([0.0, 0.0])
            ),
            input_activations=jnp.array([[3.0]]),
            dZ=jnp.array([[1.0, 1.0]]),
            expected_dX=jnp.array([[3.0]]),
            expected_dW=jnp.array([[3.0, 3.0]]),
            expected_dB=jnp.array([1.0, 1.0]),
        ),
        BackwardPropTestCase(
            name="batch_processing_gradients",
            input_dimensions=(2,),
            output_dimensions=(2,),
            parameters=Linear.Parameters(
                weights=jnp.array([[1.0, 0.0], [0.0, 1.0]]),
                biases=jnp.array([0.0, 0.0]),
            ),
            input_activations=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
            dZ=jnp.array([[1.0, 1.0], [1.0, 1.0]]),
            expected_dX=jnp.array([[1.0, 1.0], [1.0, 1.0]]),
            expected_dW=jnp.array([[4.0, 4.0], [6.0, 6.0]]),
            expected_dB=jnp.array([2.0, 2.0]),
        ),
    ],
    ids=lambda test_case: test_case.name,
)
def test_linear_backward_prop(test_case: BackwardPropTestCase):
    layer = Linear(
        input_dimensions=test_case.input_dimensions,
        output_dimensions=test_case.output_dimensions,
        parameters_init_fn=lambda *_: test_case.parameters,
        clip_gradients=False,
    )
    layer.forward_prop(input_activations=Activations(test_case.input_activations))
    assert jnp.allclose(layer.backward_prop(dZ=test_case.dZ), test_case.expected_dX)  # type: ignore[arg-type]
    assert layer.cache["dP"] is not None  # type: ignore[index]
    assert jnp.allclose(layer.cache["dP"].weights, test_case.expected_dW)  # type: ignore[attr-defined]
    assert jnp.allclose(layer.cache["dP"].biases, test_case.expected_dB)  # type: ignore[attr-defined]


@dataclass(frozen=True)
class ParameterUpdateTestCase:
    name: str
    input_dimensions: Dimensions
    output_dimensions: Dimensions
    initial_parameters: ParametersType
    input_activations: jnp.ndarray
    dZ: jnp.ndarray
    expected_updated_weights: jnp.ndarray
    expected_updated_biases: jnp.ndarray


@pytest.mark.parametrize(
    "test_case",
    [
        ParameterUpdateTestCase(
            name="simple_parameter_update",
            input_dimensions=(1,),
            output_dimensions=(1,),
            initial_parameters=Linear.Parameters(
                weights=jnp.array([[1.0]]), biases=jnp.array([0.0])
            ),
            input_activations=jnp.array([[2.0]]),
            dZ=jnp.array([[1.0]]),
            expected_updated_weights=jnp.array([[3.0]]),
            expected_updated_biases=jnp.array([1.0]),
        ),
        ParameterUpdateTestCase(
            name="multi_dimensional_update",
            input_dimensions=(2,),
            output_dimensions=(2,),
            initial_parameters=Linear.Parameters(
                weights=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
                biases=jnp.array([0.5, -0.5]),
            ),
            input_activations=jnp.array([[1.0, 1.0]]),
            dZ=jnp.array([[1.0, 1.0]]),
            expected_updated_weights=jnp.array([[2.0, 3.0], [4.0, 5.0]]),
            expected_updated_biases=jnp.array([1.5, 0.5]),
        ),
    ],
    ids=lambda test_case: test_case.name,
)
def test_linear_parameter_update(test_case: ParameterUpdateTestCase):
    layer = Linear(
        input_dimensions=test_case.input_dimensions,
        output_dimensions=test_case.output_dimensions,
        parameters_init_fn=lambda *_: test_case.initial_parameters,
        clip_gradients=False,
    )
    layer.forward_prop(input_activations=Activations(test_case.input_activations))
    layer.backward_prop(dZ=test_case.dZ)
    layer.update_parameters()

    assert jnp.allclose(layer.parameters.weights, test_case.expected_updated_weights)
    assert jnp.allclose(layer.parameters.biases, test_case.expected_updated_biases)
    assert layer.cache["dP"] is None


@pytest.fixture
def identity_layer() -> Linear:
    return Linear(
        input_dimensions=(3,),
        output_dimensions=(3,),
        parameters_init_fn=lambda *_: Linear.Parameters.eye((3,)),
    )


@pytest.fixture
def simple_layer() -> Linear:
    return Linear(
        input_dimensions=(2,),
        output_dimensions=(2,),
        parameters_init_fn=lambda *_: Linear.Parameters(
            weights=jnp.array([[2.0, 0.0], [0.0, 3.0]]), biases=jnp.array([1.0, -1.0])
        ),
    )


@pytest.fixture
def test_input() -> jnp.ndarray:
    return jnp.array([[1.0, 2.0, 3.0]])


def test_linear_cache_initialization(identity_layer: Linear):
    assert identity_layer.cache["input_activations"] is None
    assert identity_layer.cache["output_activations"] is None
    assert identity_layer.cache["dP"] is None


def test_linear_forward_prop_caches_input(
    identity_layer: Linear, test_input: jnp.ndarray
):
    identity_layer.forward_prop(input_activations=Activations(test_input))
    assert identity_layer.cache["input_activations"] is not None
    assert jnp.allclose(identity_layer.cache["input_activations"], test_input)


def test_linear_gradient_clipping():
    layer = Linear(
        input_dimensions=(2,),
        output_dimensions=(2,),
        parameters_init_fn=lambda *_: Linear.Parameters(
            weights=jnp.array([[1.0, 0.0], [0.0, 1.0]]), biases=jnp.array([0.0, 0.0])
        ),
        clip_gradients=True,
        weight_max_norm=1.0,
        bias_max_norm=1.0,
    )

    layer.forward_prop(input_activations=Activations(jnp.array([[1.0, 1.0]])))
    layer.backward_prop(dZ=jnp.array([[1000.0, 1000.0]]))

    assert layer.cache["dP"] is not None
    assert (
        jnp.linalg.norm(layer.cache["dP"].weights)
        / jnp.sqrt(layer.cache["dP"].weights.size)
        <= 1.0 + 1e-6
    )
    assert (
        jnp.linalg.norm(layer.cache["dP"].biases)
        / jnp.sqrt(layer.cache["dP"].biases.size)
        <= 1.0 + 1e-6
    )


def test_linear_frozen_parameters():
    layer = Linear(
        input_dimensions=(1,),
        output_dimensions=(1,),
        parameters_init_fn=lambda *_: Linear.Parameters(
            weights=jnp.array([[1.0]]), biases=jnp.array([0.0])
        ),
        freeze_parameters=True,
    )

    original_weights = layer.parameters.weights.copy()
    original_biases = layer.parameters.biases.copy()

    layer.forward_prop(input_activations=Activations(jnp.array([[2.0]])))
    layer.backward_prop(dZ=jnp.array([[1.0]]))
    layer.update_parameters()

    assert jnp.allclose(layer.parameters.weights, original_weights)
    assert jnp.allclose(layer.parameters.biases, original_biases)


def test_linear_empty_gradient(simple_layer: Linear):
    empty_grad = simple_layer.empty_gradient()
    assert jnp.allclose(
        empty_grad.weights,  # type: ignore[attr-defined]
        jnp.zeros_like(simple_layer.parameters.weights),  # type: ignore[attr-defined]
    )
    assert jnp.allclose(
        empty_grad.biases,  # type: ignore[attr-defined]
        jnp.zeros_like(simple_layer.parameters.biases),  # type: ignore[attr-defined]
    )


def test_linear_parameter_count(simple_layer: Linear):
    assert (
        simple_layer.parameter_count
        == simple_layer.parameters.weights.size + simple_layer.parameters.biases.size
    )


def test_linear_serialization_deserialization():
    layer = Linear(
        input_dimensions=(2,),
        output_dimensions=(2,),
        parameters_init_fn=lambda *_: Linear.Parameters.xavier(
            (2,), (2,), key=random.PRNGKey(42)
        ),
    )

    # First do a forward and backward pass to populate gradients
    input_data = jnp.array([[1.0, 2.0]])
    layer.forward_prop(input_activations=Activations(input_data))
    layer.backward_prop(dZ=jnp.array([[1.0, 1.0]]))

    original_dP = layer.cache["dP"]
    assert original_dP is not None

    buffer = io.BytesIO()
    layer._layer_id = "test_layer_for_serialization"
    layer.write_serialized_parameters(buffer)

    # Clear gradients and test deserialization
    layer.cache["dP"] = None

    buffer.seek(0)
    layer.read_serialized_parameters(buffer)

    assert layer.cache["dP"] is not None
    assert jnp.allclose(layer.cache["dP"].weights, original_dP.weights)
    assert jnp.allclose(layer.cache["dP"].biases, original_dP.biases)


def test_linear_serialize_deserialize_parameters_with_wrong_layer_id():
    layer_1 = Linear(
        input_dimensions=(2,),
        output_dimensions=(2,),
        parameters_init_fn=lambda *_: Linear.Parameters.xavier(
            (2,), (2,), key=random.PRNGKey(42)
        ),
    )
    layer_2 = Linear(
        input_dimensions=(2,),
        output_dimensions=(2,),
        parameters_init_fn=lambda *_: Linear.Parameters.xavier(
            (2,), (2,), key=random.PRNGKey(43)
        ),
    )

    # First do a forward and backward pass to populate gradients
    input_data = jnp.array([[1.0, 2.0]])
    layer_1.forward_prop(input_activations=Activations(input_data))
    layer_1.backward_prop(dZ=jnp.array([[1.0, 1.0]]))

    buffer = io.BytesIO()
    layer_1.write_serialized_parameters(buffer)
    buffer.seek(0)
    with pytest.raises(BadLayerId):
        layer_2.read_serialized_parameters(buffer)


def test_linear_error_on_backward_prop_without_forward():
    layer = Linear(
        input_dimensions=(2,),
        output_dimensions=(2,),
        parameters_init_fn=lambda *_: Linear.Parameters.xavier(
            (2,), (2,), key=random.PRNGKey(42)
        ),
    )
    with pytest.raises(
        ValueError, match="Input activations not set during forward pass"
    ):
        layer.backward_prop(dZ=jnp.array([[1.0, 1.0]]))


def test_linear_error_on_update_without_gradients():
    layer = Linear(
        input_dimensions=(2,),
        output_dimensions=(2,),
        parameters_init_fn=lambda *_: Linear.Parameters.xavier(
            (2,), (2,), key=random.PRNGKey(42)
        ),
    )
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
        parameters_init_fn=partial(init_method, key=random.PRNGKey(42)),
    )

    assert layer.parameters.weights.shape == (input_dim[0], output_dim[0])
    assert layer.parameters.biases.shape == (output_dim[0],)
    assert not jnp.allclose(layer.parameters.weights, 0)


def test_linear_mathematical_properties():
    W = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    layer = Linear(
        input_dimensions=(3,),
        output_dimensions=(2,),
        parameters_init_fn=lambda *_: Linear.Parameters(
            weights=W, biases=jnp.array([0.0, 0.0])
        ),
    )

    x1, x2 = jnp.array([[1.0, 2.0, 3.0]]), jnp.array([[4.0, 5.0, 6.0]])
    a, b = 2.0, 3.0

    result_combined = layer.forward_prop(input_activations=Activations(a * x1 + b * x2))
    result_x1 = layer.forward_prop(input_activations=Activations(x1))
    result_x2 = layer.forward_prop(input_activations=Activations(x2))

    assert jnp.allclose(result_combined, a * result_x1 + b * result_x2)

    layer_with_bias = Linear(
        input_dimensions=(3,),
        output_dimensions=(2,),
        parameters_init_fn=lambda *_: Linear.Parameters(
            weights=W, biases=jnp.array([1.0, -1.0])
        ),
    )

    x = jnp.array([[1.0, 2.0, 3.0]])
    assert jnp.allclose(
        layer_with_bias.forward_prop(input_activations=Activations(x)),
        x @ W + jnp.array([1.0, -1.0]),
    )


def test_linear_zero_input():
    layer = Linear(
        input_dimensions=(3,),
        output_dimensions=(2,),
        parameters_init_fn=lambda *_: Linear.Parameters(
            weights=jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            biases=jnp.array([1.0, -1.0]),
        ),
    )

    assert jnp.allclose(
        layer.forward_prop(input_activations=Activations(jnp.zeros((1, 3)))),
        layer.parameters.biases,
    )


def test_linear_large_batch():
    layer = Linear(
        input_dimensions=(2,),
        output_dimensions=(3,),
        parameters_init_fn=lambda *_: Linear.Parameters.xavier(
            (2,), (3,), key=random.PRNGKey(42)
        ),
    )

    # Split key for different random operations
    key1, key2 = random.split(random.PRNGKey(42))
    input_data = random.normal(key1, (100, 2))
    output = layer.forward_prop(input_activations=Activations(input_data))

    assert output.shape == (100, 3)

    dX = layer.backward_prop(dZ=random.normal(key2, (100, 3)))

    assert dX.shape == (100, 2)
    assert layer.cache["dP"] is not None


def test_linear_gradient_accumulation():
    layer = Linear(
        input_dimensions=(2,),
        output_dimensions=(1,),
        parameters_init_fn=lambda *_: Linear.Parameters(
            weights=jnp.array([[1.0], [1.0]]), biases=jnp.array([0.0])
        ),
        clip_gradients=False,
    )

    layer.forward_prop(input_activations=Activations(jnp.array([[1.0, 1.0]])))
    layer.backward_prop(dZ=jnp.array([[1.0]]))
    grad1 = layer.cache["dP"]

    layer.forward_prop(input_activations=Activations(jnp.array([[2.0, 2.0]])))
    layer.backward_prop(dZ=jnp.array([[1.0]]))
    grad2 = layer.cache["dP"]

    assert not jnp.allclose(grad1.weights, grad2.weights)


def test_linear_weight_and_bias_shapes():
    layer = Linear(
        input_dimensions=(2,),
        output_dimensions=(3,),
        parameters_init_fn=lambda *_: Linear.Parameters(
            weights=random.normal(key, (2, 3)), biases=random.normal(key, (3,))
        ),
    )
    assert layer.parameters.weights.shape == (2, 3)
    assert layer.parameters.biases.shape == (3,)

    with pytest.raises(ValueError, match="Weight matrix shape"):
        Linear(
            input_dimensions=(2,),
            output_dimensions=(3,),
            parameters_init_fn=lambda *_: Linear.Parameters(
                weights=random.normal(key, (3, 2)), biases=random.normal(key, (3,))
            ),
        )

    with pytest.raises(ValueError, match="Bias vector shape"):
        Linear(
            input_dimensions=(2,),
            output_dimensions=(3,),
            parameters_init_fn=lambda *_: Linear.Parameters(
                weights=random.normal(key, (2, 3)), biases=random.normal(key, (2,))
            ),
        )


def test_linear_numerical_stability():
    layer = Linear(
        input_dimensions=(2,),
        output_dimensions=(2,),
        parameters_init_fn=lambda *_: Linear.Parameters(
            weights=jnp.array([[1e-8, 1e8], [1e8, 1e-8]]), biases=jnp.array([1e-8, 1e8])
        ),
        clip_gradients=False,
    )

    assert jnp.isfinite(
        layer.forward_prop(input_activations=Activations(jnp.array([[1e-8, 1e-8]])))
    ).all()
    assert jnp.isfinite(
        layer.forward_prop(input_activations=Activations(jnp.array([[1e8, 1e8]])))
    ).all()


@pytest.mark.parametrize("store_output", [True, False])
def test_linear_output_storage_option(store_output):
    layer = Linear(
        input_dimensions=(2,),
        output_dimensions=(2,),
        parameters_init_fn=lambda *_: Linear.Parameters.xavier(
            (2,), (2,), key=random.PRNGKey(42)
        ),
        store_output_activations=store_output,
    )

    output = layer.forward_prop(input_activations=Activations(jnp.array([[1.0, 2.0]])))

    if store_output:
        assert layer.cache["output_activations"] is not None
        assert jnp.allclose(layer.cache["output_activations"], output)
    else:
        assert layer.cache["output_activations"] is None


def test_linear_constructor_edge_cases():
    layer = Linear(
        input_dimensions=(3,),
        parameters_init_fn=lambda *_: Linear.Parameters.xavier(
            (3,), (3,), key=random.PRNGKey(42)
        ),
    )
    assert layer.input_dimensions == (3,)
    assert layer.output_dimensions == (3,)

    bias_layer = Linear.of_bias(dim=(2,), bias=5.0)
    assert jnp.allclose(bias_layer.parameters.weights, 0)
    assert jnp.allclose(bias_layer.parameters.biases, 5.0)

    identity_layer = Linear.of_eye(dim=(3,))
    assert jnp.allclose(identity_layer.parameters.weights, jnp.eye(3))
    assert jnp.allclose(identity_layer.parameters.biases, 0)


def test_linear_gradient_operation_interface():
    layer = Linear(
        input_dimensions=(2,),
        output_dimensions=(2,),
        parameters_init_fn=lambda *_: Linear.Parameters.xavier(
            (2,), (2,), key=random.PRNGKey(42)
        ),
    )

    called = False

    def grad_callback(grad_layer):
        nonlocal called
        called = True
        assert grad_layer is layer

    layer.gradient_operation(grad_callback)
    assert called
