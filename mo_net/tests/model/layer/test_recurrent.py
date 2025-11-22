import io
from dataclasses import dataclass
from functools import partial
from typing import Final

import jax
import jax.numpy as jnp
import jax.random as random
import pytest

from mo_net.functions import Identity, Tanh
from mo_net.model.layer.base import BadLayerId
from mo_net.model.layer.recurrent import Recurrent, ParametersType
from mo_net.protos import Activations, Dimensions

key: Final = jax.random.PRNGKey(42)


@dataclass(frozen=True)
class ForwardPropTestCase:
    name: str
    input_dimensions: Dimensions
    hidden_dimensions: Dimensions
    parameters: ParametersType
    input_activations: jnp.ndarray  # (batch, seq_len, input_dim)
    expected_output: jnp.ndarray
    return_sequences: bool = True


@pytest.mark.parametrize(
    "test_case",
    [
        ForwardPropTestCase(
            name="simple_single_step",
            input_dimensions=(2,),
            hidden_dimensions=(3,),
            parameters=Recurrent.Parameters(
                weights_ih=jnp.ones((2, 3)),
                weights_hh=jnp.zeros((3, 3)),
                biases=jnp.zeros(3),
            ),
            input_activations=jnp.array([[[1.0, 1.0]]]),  # Single timestep
            expected_output=jnp.array([[[jnp.tanh(2.0)] * 3]]),
            return_sequences=True,
        ),
        ForwardPropTestCase(
            name="identity_weights",
            input_dimensions=(2,),
            hidden_dimensions=(2,),
            parameters=Recurrent.Parameters(
                weights_ih=jnp.eye(2),
                weights_hh=jnp.zeros((2, 2)),
                biases=jnp.zeros(2),
            ),
            input_activations=jnp.array([[[1.0, 2.0], [3.0, 4.0]]]),
            expected_output=jnp.array(
                [[[jnp.tanh(1.0), jnp.tanh(2.0)], [jnp.tanh(3.0), jnp.tanh(4.0)]]]
            ),
            return_sequences=True,
        ),
        ForwardPropTestCase(
            name="zero_recurrent_weights",
            input_dimensions=(1,),
            hidden_dimensions=(2,),
            parameters=Recurrent.Parameters(
                weights_ih=jnp.array([[1.0, 2.0]]),
                weights_hh=jnp.zeros((2, 2)),
                biases=jnp.zeros(2),
            ),
            input_activations=jnp.array([[[1.0], [2.0]]]),  # Two timesteps
            expected_output=jnp.array(
                [
                    [
                        [jnp.tanh(1.0), jnp.tanh(2.0)],
                        [jnp.tanh(2.0), jnp.tanh(4.0)],
                    ]
                ]
            ),
            return_sequences=True,
        ),
        ForwardPropTestCase(
            name="return_sequences_false",
            input_dimensions=(1,),
            hidden_dimensions=(2,),
            parameters=Recurrent.Parameters(
                weights_ih=jnp.array([[1.0, 2.0]]),
                weights_hh=jnp.zeros((2, 2)),
                biases=jnp.zeros(2),
            ),
            input_activations=jnp.array([[[1.0], [2.0]]]),  # Two timesteps
            expected_output=jnp.array([[jnp.tanh(2.0), jnp.tanh(4.0)]]),  # Final state
            return_sequences=False,
        ),
    ],
    ids=lambda test_case: test_case.name,
)
def test_recurrent_forward_prop(test_case: ForwardPropTestCase):
    layer = Recurrent(
        input_dimensions=test_case.input_dimensions,
        hidden_dimensions=test_case.hidden_dimensions,
        parameters_init_fn=lambda *_: test_case.parameters,
        activation_fn=Tanh(),
        return_sequences=test_case.return_sequences,
    )
    output = layer.forward_prop(input_activations=Activations(test_case.input_activations))
    assert jnp.allclose(output, test_case.expected_output, atol=1e-6)


@dataclass(frozen=True)
class BackwardPropTestCase:
    name: str
    input_dimensions: Dimensions
    hidden_dimensions: Dimensions
    parameters: ParametersType
    input_activations: jnp.ndarray
    dZ: jnp.ndarray
    return_sequences: bool = True


@pytest.mark.parametrize(
    "test_case",
    [
        BackwardPropTestCase(
            name="simple_single_timestep",
            input_dimensions=(2,),
            hidden_dimensions=(2,),
            parameters=Recurrent.Parameters(
                weights_ih=jnp.eye(2),
                weights_hh=jnp.zeros((2, 2)),
                biases=jnp.zeros(2),
            ),
            input_activations=jnp.array([[[1.0, 2.0]]]),
            dZ=jnp.array([[[1.0, 1.0]]]),
            return_sequences=True,
        ),
        BackwardPropTestCase(
            name="multi_timestep",
            input_dimensions=(1,),
            hidden_dimensions=(2,),
            parameters=Recurrent.Parameters(
                weights_ih=jnp.ones((1, 2)),
                weights_hh=jnp.zeros((2, 2)),
                biases=jnp.zeros(2),
            ),
            input_activations=jnp.array([[[1.0], [2.0], [3.0]]]),
            dZ=jnp.array([[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]]),
            return_sequences=True,
        ),
    ],
    ids=lambda test_case: test_case.name,
)
def test_recurrent_backward_prop(test_case: BackwardPropTestCase):
    layer = Recurrent(
        input_dimensions=test_case.input_dimensions,
        hidden_dimensions=test_case.hidden_dimensions,
        parameters_init_fn=lambda *_: test_case.parameters,
        activation_fn=Tanh(),
        return_sequences=test_case.return_sequences,
        clip_gradients=False,
    )

    # Forward pass
    layer.forward_prop(input_activations=Activations(test_case.input_activations))

    # Backward pass
    dX = layer.backward_prop(dZ=test_case.dZ)

    # Check that gradients are computed
    assert layer.cache["dP"] is not None
    assert dX.shape == test_case.input_activations.shape

    # Check that gradients are finite
    assert jnp.isfinite(layer.cache["dP"].weights_ih).all()
    assert jnp.isfinite(layer.cache["dP"].weights_hh).all()
    assert jnp.isfinite(layer.cache["dP"].biases).all()
    assert jnp.isfinite(dX).all()


def test_recurrent_parameter_update():
    """Test that parameters are updated correctly."""
    layer = Recurrent(
        input_dimensions=(1,),
        hidden_dimensions=(2,),
        parameters_init_fn=lambda *_: Recurrent.Parameters(
            weights_ih=jnp.array([[1.0, 1.0]]),
            weights_hh=jnp.zeros((2, 2)),
            biases=jnp.zeros(2),
        ),
        activation_fn=Tanh(),
        clip_gradients=False,
    )

    original_weights_ih = layer.parameters.weights_ih.copy()
    original_weights_hh = layer.parameters.weights_hh.copy()
    original_biases = layer.parameters.biases.copy()

    # Forward and backward pass
    input_data = jnp.array([[[1.0], [2.0]]])
    layer.forward_prop(input_activations=Activations(input_data))
    layer.backward_prop(dZ=jnp.array([[[1.0, 1.0], [1.0, 1.0]]]))

    # Store gradients
    stored_dP = layer.cache["dP"]
    assert stored_dP is not None

    # Update parameters
    layer.update_parameters()

    # Check parameters changed
    assert jnp.allclose(
        layer.parameters.weights_ih, original_weights_ih + stored_dP.weights_ih
    )
    assert jnp.allclose(
        layer.parameters.weights_hh, original_weights_hh + stored_dP.weights_hh
    )
    assert jnp.allclose(layer.parameters.biases, original_biases + stored_dP.biases)

    # Check gradients were cleared
    assert layer.cache["dP"] is None


def test_recurrent_batch_processing():
    """Test that RNN handles batched inputs correctly."""
    batch_size = 3
    seq_len = 4
    input_dim = 2
    hidden_dim = 3

    layer = Recurrent(
        input_dimensions=(input_dim,),
        hidden_dimensions=(hidden_dim,),
        parameters_init_fn=partial(
            Recurrent.Parameters.orthogonal, key=key
        ),
        activation_fn=Tanh(),
    )

    # Create batched input
    key1, key2 = random.split(key)
    input_data = random.normal(key1, (batch_size, seq_len, input_dim))

    # Forward pass
    output = layer.forward_prop(input_activations=Activations(input_data))
    assert output.shape == (batch_size, seq_len, hidden_dim)

    # Backward pass
    dZ = random.normal(key2, (batch_size, seq_len, hidden_dim))
    dX = layer.backward_prop(dZ=dZ)
    assert dX.shape == (batch_size, seq_len, input_dim)


def test_recurrent_return_sequences_flag():
    """Test return_sequences flag behavior."""
    layer_sequences = Recurrent(
        input_dimensions=(2,),
        hidden_dimensions=(3,),
        parameters_init_fn=partial(Recurrent.Parameters.orthogonal, key=key),
        activation_fn=Tanh(),
        return_sequences=True,
    )

    layer_final = Recurrent(
        input_dimensions=(2,),
        hidden_dimensions=(3,),
        parameters_init_fn=partial(Recurrent.Parameters.orthogonal, key=key),
        activation_fn=Tanh(),
        return_sequences=False,
    )

    input_data = jnp.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])

    output_sequences = layer_sequences.forward_prop(input_activations=Activations(input_data))
    output_final = layer_final.forward_prop(input_activations=Activations(input_data))

    # With sequences, output should be (batch, seq_len, hidden_dim)
    assert output_sequences.shape == (1, 3, 3)

    # Without sequences, output should be (batch, hidden_dim)
    assert output_final.shape == (1, 3)


def test_recurrent_stateful_mode():
    """Test stateful RNN that maintains hidden state across batches."""
    layer = Recurrent(
        input_dimensions=(1,),
        hidden_dimensions=(2,),
        parameters_init_fn=lambda *_: Recurrent.Parameters(
            weights_ih=jnp.array([[1.0, 1.0]]),
            weights_hh=jnp.array([[0.5, 0.0], [0.0, 0.5]]),
            biases=jnp.zeros(2),
        ),
        activation_fn=Identity(),  # Use identity for predictable behavior
        stateful=True,
        return_sequences=False,
    )

    # First batch
    output1 = layer.forward_prop(input_activations=Activations(jnp.array([[[1.0]]])))

    # Second batch should use state from first batch
    output2 = layer.forward_prop(input_activations=Activations(jnp.array([[[1.0]]])))

    # Outputs should be different due to carried state
    assert not jnp.allclose(output1, output2)

    # Reset state
    layer.reset_state()

    # After reset, should get same output as first time
    output3 = layer.forward_prop(input_activations=Activations(jnp.array([[[1.0]]])))
    assert jnp.allclose(output1, output3)


def test_recurrent_gradient_clipping():
    """Test gradient clipping prevents exploding gradients."""
    layer = Recurrent(
        input_dimensions=(2,),
        hidden_dimensions=(2,),
        parameters_init_fn=lambda *_: Recurrent.Parameters(
            weights_ih=jnp.ones((2, 2)),
            weights_hh=jnp.ones((2, 2)),
            biases=jnp.zeros(2),
        ),
        clip_gradients=True,
        weight_max_norm=1.0,
    )

    # Create situation that would cause large gradients
    input_data = jnp.array([[[1000.0, 1000.0], [1000.0, 1000.0]]])
    layer.forward_prop(input_activations=Activations(input_data))
    layer.backward_prop(dZ=jnp.array([[[1000.0, 1000.0], [1000.0, 1000.0]]]))

    assert layer.cache["dP"] is not None

    # Check gradients are clipped
    def check_norm(grad, size):
        norm = jnp.linalg.norm(grad) / jnp.sqrt(size)
        return norm <= 1.0 + 1e-5

    assert check_norm(layer.cache["dP"].weights_ih, layer.cache["dP"].weights_ih.size)
    assert check_norm(layer.cache["dP"].weights_hh, layer.cache["dP"].weights_hh.size)
    assert check_norm(layer.cache["dP"].biases, layer.cache["dP"].biases.size)


def test_recurrent_frozen_parameters():
    """Test that frozen parameters don't update."""
    layer = Recurrent(
        input_dimensions=(1,),
        hidden_dimensions=(2,),
        parameters_init_fn=lambda *_: Recurrent.Parameters(
            weights_ih=jnp.ones((1, 2)),
            weights_hh=jnp.zeros((2, 2)),
            biases=jnp.zeros(2),
        ),
        freeze_parameters=True,
    )

    original_weights_ih = layer.parameters.weights_ih.copy()
    original_weights_hh = layer.parameters.weights_hh.copy()
    original_biases = layer.parameters.biases.copy()

    # Forward, backward, and update
    layer.forward_prop(input_activations=Activations(jnp.array([[[1.0]]])))
    layer.backward_prop(dZ=jnp.array([[[1.0, 1.0]]]))
    layer.update_parameters()

    # Parameters should not change
    assert jnp.allclose(layer.parameters.weights_ih, original_weights_ih)
    assert jnp.allclose(layer.parameters.weights_hh, original_weights_hh)
    assert jnp.allclose(layer.parameters.biases, original_biases)


def test_recurrent_empty_gradient():
    """Test empty gradient generation."""
    layer = Recurrent(
        input_dimensions=(2,),
        hidden_dimensions=(3,),
        parameters_init_fn=partial(Recurrent.Parameters.orthogonal, key=key),
    )

    empty_grad = layer.empty_gradient()
    assert jnp.allclose(empty_grad.weights_ih, jnp.zeros_like(layer.parameters.weights_ih))
    assert jnp.allclose(empty_grad.weights_hh, jnp.zeros_like(layer.parameters.weights_hh))
    assert jnp.allclose(empty_grad.biases, jnp.zeros_like(layer.parameters.biases))


def test_recurrent_parameter_count():
    """Test parameter count calculation."""
    input_dim, hidden_dim = 3, 5
    layer = Recurrent(
        input_dimensions=(input_dim,),
        hidden_dimensions=(hidden_dim,),
        parameters_init_fn=partial(Recurrent.Parameters.orthogonal, key=key),
    )

    expected_count = (
        input_dim * hidden_dim  # weights_ih
        + hidden_dim * hidden_dim  # weights_hh
        + hidden_dim  # biases
    )
    assert layer.parameter_count == expected_count


def test_recurrent_serialization_deserialization():
    """Test model serialization and deserialization."""
    layer = Recurrent(
        input_dimensions=(2,),
        hidden_dimensions=(3,),
        parameters_init_fn=partial(Recurrent.Parameters.orthogonal, key=key),
    )

    # Forward and backward pass to populate gradients
    input_data = jnp.array([[[1.0, 2.0], [3.0, 4.0]]])
    layer.forward_prop(input_activations=Activations(input_data))
    layer.backward_prop(dZ=jnp.array([[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]))

    original_dP = layer.cache["dP"]
    assert original_dP is not None

    # Serialize
    buffer = io.BytesIO()
    layer._layer_id = "test_recurrent_layer"
    layer.write_serialized_parameters(buffer)

    # Clear and deserialize
    layer.cache["dP"] = None
    buffer.seek(0)
    layer.read_serialized_parameters(buffer)

    # Check gradients match
    assert layer.cache["dP"] is not None
    assert jnp.allclose(layer.cache["dP"].weights_ih, original_dP.weights_ih)
    assert jnp.allclose(layer.cache["dP"].weights_hh, original_dP.weights_hh)
    assert jnp.allclose(layer.cache["dP"].biases, original_dP.biases)


def test_recurrent_wrong_layer_id():
    """Test that deserialization fails with wrong layer ID."""
    layer_1 = Recurrent(
        input_dimensions=(2,),
        hidden_dimensions=(2,),
        parameters_init_fn=partial(Recurrent.Parameters.orthogonal, key=key),
    )
    layer_2 = Recurrent(
        input_dimensions=(2,),
        hidden_dimensions=(2,),
        parameters_init_fn=partial(Recurrent.Parameters.orthogonal, key=key),
    )

    # Populate gradients
    layer_1.forward_prop(input_activations=Activations(jnp.array([[[1.0, 2.0]]])))
    layer_1.backward_prop(dZ=jnp.array([[[1.0, 1.0]]]))

    # Serialize layer 1
    buffer = io.BytesIO()
    layer_1.write_serialized_parameters(buffer)

    # Try to deserialize into layer 2
    buffer.seek(0)
    with pytest.raises(BadLayerId):
        layer_2.read_serialized_parameters(buffer)


def test_recurrent_error_on_backward_without_forward():
    """Test that backward prop fails without forward prop."""
    layer = Recurrent(
        input_dimensions=(2,),
        hidden_dimensions=(2,),
        parameters_init_fn=partial(Recurrent.Parameters.orthogonal, key=key),
    )

    with pytest.raises(ValueError, match="Input activations not set"):
        layer.backward_prop(dZ=jnp.array([[[1.0, 1.0]]]))


def test_recurrent_error_on_update_without_gradients():
    """Test that parameter update fails without gradients."""
    layer = Recurrent(
        input_dimensions=(2,),
        hidden_dimensions=(2,),
        parameters_init_fn=partial(Recurrent.Parameters.orthogonal, key=key),
    )

    with pytest.raises(ValueError, match="Gradient not set"):
        layer.update_parameters()


@pytest.mark.parametrize(
    "init_method,input_dim,hidden_dim",
    [
        (Recurrent.Parameters.xavier, (3,), (5,)),
        (Recurrent.Parameters.orthogonal, (4,), (6,)),
        (Recurrent.Parameters.random, (2,), (3,)),
    ],
)
def test_recurrent_initialization_methods(init_method, input_dim, hidden_dim):
    """Test different initialization methods."""
    layer = Recurrent(
        input_dimensions=input_dim,
        hidden_dimensions=hidden_dim,
        parameters_init_fn=partial(init_method, key=key),
    )

    assert layer.parameters.weights_ih.shape == (input_dim[0], hidden_dim[0])
    assert layer.parameters.weights_hh.shape == (hidden_dim[0], hidden_dim[0])
    assert layer.parameters.biases.shape == (hidden_dim[0],)
    assert not jnp.allclose(layer.parameters.weights_ih, 0)


def test_recurrent_orthogonal_initialization():
    """Test that orthogonal initialization produces orthogonal recurrent weights."""
    hidden_dim = (10,)
    layer = Recurrent(
        input_dimensions=(5,),
        hidden_dimensions=hidden_dim,
        parameters_init_fn=partial(Recurrent.Parameters.orthogonal, key=key),
    )

    W_hh = layer.parameters.weights_hh
    # Check that W_hh @ W_hh.T is approximately identity
    product = W_hh @ W_hh.T
    assert jnp.allclose(product, jnp.eye(hidden_dim[0]), atol=1e-5)


def test_recurrent_cache_initialization():
    """Test that cache is initialized properly."""
    layer = Recurrent(
        input_dimensions=(2,),
        hidden_dimensions=(3,),
        parameters_init_fn=partial(Recurrent.Parameters.orthogonal, key=key),
    )

    assert layer.cache["input_activations"] is None
    assert layer.cache["hidden_states"] is None
    assert layer.cache["output_activations"] is None
    assert layer.cache["dP"] is None


def test_recurrent_caches_activations():
    """Test that forward prop caches activations."""
    layer = Recurrent(
        input_dimensions=(2,),
        hidden_dimensions=(3,),
        parameters_init_fn=partial(Recurrent.Parameters.orthogonal, key=key),
        store_output_activations=True,
    )

    input_data = jnp.array([[[1.0, 2.0], [3.0, 4.0]]])
    output = layer.forward_prop(input_activations=Activations(input_data))

    assert layer.cache["input_activations"] is not None
    assert layer.cache["hidden_states"] is not None
    assert layer.cache["output_activations"] is not None
    assert jnp.allclose(layer.cache["output_activations"], output)


def test_recurrent_single_timestep_vs_2d_input():
    """Test that single timestep works with both 2D and 3D inputs."""
    layer = Recurrent(
        input_dimensions=(2,),
        hidden_dimensions=(3,),
        parameters_init_fn=lambda *_: Recurrent.Parameters(
            weights_ih=jnp.ones((2, 3)),
            weights_hh=jnp.zeros((3, 3)),
            biases=jnp.zeros(3),
        ),
        activation_fn=Tanh(),
    )

    # 2D input (batch, features)
    input_2d = jnp.array([[1.0, 2.0]])
    output_2d = layer.forward_prop(input_activations=Activations(input_2d))

    # 3D input (batch, 1, features)
    input_3d = jnp.array([[[1.0, 2.0]]])
    layer.cache["input_activations"] = None  # Reset cache
    output_3d = layer.forward_prop(input_activations=Activations(input_3d))

    # Both should produce same result
    assert jnp.allclose(output_2d, output_3d)


def test_recurrent_gradient_operation_interface():
    """Test gradient operation callback interface."""
    layer = Recurrent(
        input_dimensions=(2,),
        hidden_dimensions=(2,),
        parameters_init_fn=partial(Recurrent.Parameters.orthogonal, key=key),
    )

    called = False

    def grad_callback(grad_layer):
        nonlocal called
        called = True
        assert grad_layer is layer

    layer.gradient_operation(grad_callback)
    assert called


def test_recurrent_reinitialise():
    """Test that reinitialise resets hidden state in stateful mode."""
    layer = Recurrent(
        input_dimensions=(2,),
        hidden_dimensions=(3,),
        parameters_init_fn=partial(Recurrent.Parameters.orthogonal, key=key),
        stateful=True,
    )

    # Do some forward passes to build up hidden state
    layer.forward_prop(input_activations=Activations(jnp.array([[[1.0, 2.0]]])))
    assert layer._hidden_state is not None

    # Reinitialise should reset hidden state
    layer.reinitialise()
    assert layer._hidden_state is None
>>>>>>> Stashed changes
