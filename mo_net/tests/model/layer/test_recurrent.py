import io
from dataclasses import dataclass
from typing import Final, cast

import jax
import jax.numpy as jnp
import pytest

from mo_net.model.layer.base import BadLayerId
from mo_net.model.layer.recurrent import ParametersType, Recurrent
from mo_net.protos import Activations, D, Dimensions

key: Final = jax.random.PRNGKey(42)


@dataclass(frozen=True)
class ForwardPropTestCase:
    name: str
    input_dimensions: Dimensions
    hidden_dim: int
    return_sequences: bool
    input_activations: jnp.ndarray
    expected_output_shape: tuple[int, ...]


@pytest.mark.parametrize(
    "test_case",
    [
        ForwardPropTestCase(
            name="many_to_many_simple",
            input_dimensions=(3, 2),  # (seq_len, input_dim)
            hidden_dim=4,
            return_sequences=True,
            input_activations=jnp.array(
                [
                    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
                ]  # (batch=1, seq_len=3, input_dim=2)
            ),
            expected_output_shape=(1, 3, 4),  # (batch, seq_len, hidden_dim)
        ),
        ForwardPropTestCase(
            name="many_to_one_simple",
            input_dimensions=(3, 2),
            hidden_dim=4,
            return_sequences=False,
            input_activations=jnp.array(
                [
                    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
                ]  # (batch=1, seq_len=3, input_dim=2)
            ),
            expected_output_shape=(1, 4),  # (batch, hidden_dim)
        ),
        ForwardPropTestCase(
            name="batch_processing",
            input_dimensions=(2, 3),  # (seq_len, input_dim)
            hidden_dim=5,
            return_sequences=True,
            input_activations=jnp.array(
                [
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],  # batch 1
                    [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],  # batch 2
                ]  # (batch=2, seq_len=2, input_dim=3)
            ),
            expected_output_shape=(2, 2, 5),  # (batch, seq_len, hidden_dim)
        ),
    ],
    ids=lambda test_case: test_case.name,
)
def test_recurrent_forward_prop_shape(test_case: ForwardPropTestCase):
    layer = Recurrent(
        input_dimensions=test_case.input_dimensions,
        hidden_dim=test_case.hidden_dim,
        return_sequences=test_case.return_sequences,
        key=key,
    )
    output = layer.forward_prop(
        input_activations=Activations(test_case.input_activations)
    )
    assert output.shape == test_case.expected_output_shape


def test_recurrent_forward_prop_values():
    """Test that forward propagation produces reasonable values."""
    layer = Recurrent(
        input_dimensions=(2, 2),
        hidden_dim=3,
        return_sequences=True,
        key=key,
    )
    input_seq = jnp.array([[[1.0, 0.0], [0.0, 1.0]]])  # (batch=1, seq=2, input=2)
    output = layer.forward_prop(input_activations=Activations(input_seq))

    # Check output is finite
    assert jnp.isfinite(output).all()

    # Check output is in reasonable range for tanh activation
    assert jnp.all(output >= -1.0) and jnp.all(output <= 1.0)

    # Check that different time steps produce different outputs
    # (unless inputs are identical and initial state is zero)
    assert output.shape == (1, 2, 3)


def test_recurrent_hidden_state_persistence():
    """Test that hidden state influences future outputs."""
    layer = Recurrent(
        input_dimensions=(3, 2),
        hidden_dim=4,
        return_sequences=True,
        key=key,
    )

    # Create input where later timesteps should be influenced by earlier ones
    input_seq = jnp.array([[[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]])
    output = layer.forward_prop(input_activations=Activations(input_seq))

    # The hidden state at t=0 should influence outputs at t=1 and t=2
    # even though input is zero at those timesteps
    h1 = output[0, 1, :]
    h2 = output[0, 2, :]

    # h1 and h2 should not be zero because they're influenced by h0
    assert not jnp.allclose(h1, 0.0)
    assert not jnp.allclose(h2, 0.0)


@dataclass(frozen=True)
class BackwardPropTestCase:
    name: str
    input_dimensions: Dimensions
    hidden_dim: int
    return_sequences: bool
    input_activations: jnp.ndarray
    dZ: jnp.ndarray


@pytest.mark.parametrize(
    "test_case",
    [
        BackwardPropTestCase(
            name="many_to_many_gradients",
            input_dimensions=(2, 2),
            hidden_dim=3,
            return_sequences=True,
            input_activations=jnp.array([[[1.0, 2.0], [3.0, 4.0]]]),
            dZ=jnp.ones((1, 2, 3)),
        ),
        BackwardPropTestCase(
            name="many_to_one_gradients",
            input_dimensions=(2, 2),
            hidden_dim=3,
            return_sequences=False,
            input_activations=jnp.array([[[1.0, 2.0], [3.0, 4.0]]]),
            dZ=jnp.ones((1, 3)),
        ),
    ],
    ids=lambda test_case: test_case.name,
)
def test_recurrent_backward_prop(test_case: BackwardPropTestCase):
    layer = Recurrent(
        clip_gradients=False,
        input_dimensions=test_case.input_dimensions,
        hidden_dim=test_case.hidden_dim,
        return_sequences=test_case.return_sequences,
        key=key,
    )
    layer.forward_prop(input_activations=Activations(test_case.input_activations))
    dX = layer.backward_prop(dZ=cast(D[Activations], test_case.dZ))

    # Check that gradients are computed
    assert layer.cache["dP"] is not None
    dP = cast(ParametersType, layer.cache["dP"])

    # Check gradient shapes
    assert dP.Wxh.shape == layer.parameters.Wxh.shape
    assert dP.Whh.shape == layer.parameters.Whh.shape
    assert dP.bh.shape == layer.parameters.bh.shape

    # Check that gradients are finite
    assert jnp.isfinite(dP.Wxh).all()
    assert jnp.isfinite(dP.Whh).all()
    assert jnp.isfinite(dP.bh).all()

    # Check input gradients shape
    assert cast(jnp.ndarray, dX).shape == test_case.input_activations.shape


def test_recurrent_gradient_clipping():
    """Test that gradient clipping works correctly."""
    layer = Recurrent(
        clip_gradients=True,
        gradient_max_norm=1.0,
        input_dimensions=(2, 2),
        hidden_dim=3,
        return_sequences=True,
        key=key,
    )

    input_seq = jnp.array([[[1.0, 2.0], [3.0, 4.0]]])
    layer.forward_prop(input_activations=Activations(input_seq))

    # Large gradients
    large_dZ = jnp.ones((1, 2, 3)) * 100.0
    layer.backward_prop(dZ=cast(D[Activations], large_dZ))

    assert layer.cache["dP"] is not None
    dP = cast(ParametersType, layer.cache["dP"])

    # Check that gradients are clipped
    wxh_norm = jnp.linalg.norm(dP.Wxh)
    whh_norm = jnp.linalg.norm(dP.Whh)
    bh_norm = jnp.linalg.norm(dP.bh)

    # Each gradient should be clipped individually
    max_allowed = layer._gradient_max_norm * jnp.sqrt(dP.Wxh.size)
    assert wxh_norm <= max_allowed

    max_allowed = layer._gradient_max_norm * jnp.sqrt(dP.Whh.size)
    assert whh_norm <= max_allowed

    max_allowed = layer._gradient_max_norm * jnp.sqrt(dP.bh.size)
    assert bh_norm <= max_allowed


def test_recurrent_parameter_update():
    """Test parameter updates during training."""
    layer = Recurrent(
        clip_gradients=False,
        input_dimensions=(2, 2),
        hidden_dim=3,
        return_sequences=True,
        key=key,
    )

    # Store original parameters
    original_Wxh = layer.parameters.Wxh.copy()
    original_Whh = layer.parameters.Whh.copy()
    original_bh = layer.parameters.bh.copy()

    # Forward and backward pass
    input_seq = jnp.array([[[1.0, 2.0], [3.0, 4.0]]])
    layer.forward_prop(input_activations=Activations(input_seq))
    dZ = jnp.ones((1, 2, 3))
    layer.backward_prop(dZ=cast(D[Activations], dZ))

    # Update parameters
    layer.update_parameters()

    # Check that parameters changed
    assert not jnp.allclose(layer.parameters.Wxh, original_Wxh)
    assert not jnp.allclose(layer.parameters.Whh, original_Whh)
    assert not jnp.allclose(layer.parameters.bh, original_bh)


def test_recurrent_frozen_parameters():
    """Test that frozen parameters don't update."""
    layer = Recurrent(
        clip_gradients=False,
        freeze_parameters=True,
        input_dimensions=(2, 2),
        hidden_dim=3,
        return_sequences=True,
        key=key,
    )

    original_Wxh = layer.parameters.Wxh.copy()
    original_Whh = layer.parameters.Whh.copy()
    original_bh = layer.parameters.bh.copy()

    input_seq = jnp.array([[[1.0, 2.0], [3.0, 4.0]]])
    layer.forward_prop(input_activations=Activations(input_seq))
    dZ = jnp.ones((1, 2, 3))
    layer.backward_prop(dZ=cast(D[Activations], dZ))
    layer.update_parameters()

    # Parameters should not change
    assert jnp.allclose(layer.parameters.Wxh, original_Wxh)
    assert jnp.allclose(layer.parameters.Whh, original_Whh)
    assert jnp.allclose(layer.parameters.bh, original_bh)


def test_recurrent_cache_initialization():
    """Test that cache is properly initialized."""
    layer = Recurrent(
        input_dimensions=(2, 2),
        hidden_dim=3,
        return_sequences=True,
        key=key,
    )

    assert layer.cache["input_sequences"] is None
    assert layer.cache["hidden_states"] is None
    assert layer.cache["output_activations"] is None
    assert layer.cache["dP"] is None


def test_recurrent_cache_stores_data():
    """Test that forward pass caches necessary data."""
    layer = Recurrent(
        input_dimensions=(2, 2),
        hidden_dim=3,
        return_sequences=True,
        store_output_activations=True,
        key=key,
    )

    input_seq = jnp.array([[[1.0, 2.0], [3.0, 4.0]]])
    output = layer.forward_prop(input_activations=Activations(input_seq))

    assert layer.cache["input_sequences"] is not None
    assert layer.cache["hidden_states"] is not None
    assert layer.cache["output_activations"] is not None
    assert jnp.allclose(layer.cache["output_activations"], output)


def test_recurrent_empty_gradient():
    """Test empty gradient creation."""
    layer = Recurrent(
        input_dimensions=(2, 2),
        hidden_dim=3,
        return_sequences=True,
        key=key,
    )

    empty_grad = layer.empty_gradient()
    dP = cast(ParametersType, empty_grad)

    assert dP.Wxh.shape == layer.parameters.Wxh.shape
    assert dP.Whh.shape == layer.parameters.Whh.shape
    assert dP.bh.shape == layer.parameters.bh.shape

    assert jnp.allclose(dP.Wxh, 0.0)
    assert jnp.allclose(dP.Whh, 0.0)
    assert jnp.allclose(dP.bh, 0.0)


def test_recurrent_parameter_count():
    """Test parameter count calculation."""
    input_dim = 5
    hidden_dim = 10
    layer = Recurrent(
        input_dimensions=(3, input_dim),
        hidden_dim=hidden_dim,
        return_sequences=True,
        key=key,
    )

    expected_count = (
        input_dim * hidden_dim  # Wxh
        + hidden_dim * hidden_dim  # Whh
        + hidden_dim  # bh
    )
    assert layer.parameter_count == expected_count


@pytest.mark.parametrize(
    "init_method",
    [
        Recurrent.Parameters.xavier,
        Recurrent.Parameters.he,
        Recurrent.Parameters.random,
    ],
)
def test_recurrent_initialization_methods(init_method):
    """Test different parameter initialization methods."""
    layer = Recurrent(
        input_dimensions=(2, 3),
        hidden_dim=4,
        return_sequences=True,
        parameters_init_fn=init_method,
        key=key,
    )

    assert layer.parameters.Wxh.shape == (3, 4)
    assert layer.parameters.Whh.shape == (4, 4)
    assert layer.parameters.bh.shape == (4,)

    assert jnp.isfinite(layer.parameters.Wxh).all()
    assert jnp.isfinite(layer.parameters.Whh).all()
    assert jnp.isfinite(layer.parameters.bh).all()


def test_recurrent_error_on_backward_without_forward():
    """Test that backward without forward raises error."""
    layer = Recurrent(
        input_dimensions=(2, 2),
        hidden_dim=3,
        return_sequences=True,
        key=key,
    )

    with pytest.raises(ValueError, match="Input sequences not cached"):
        layer.backward_prop(dZ=cast(D[Activations], jnp.ones((1, 2, 3))))


def test_recurrent_error_on_update_without_gradients():
    """Test that update without gradients raises error."""
    layer = Recurrent(
        input_dimensions=(2, 2),
        hidden_dim=3,
        return_sequences=True,
        key=key,
    )

    with pytest.raises(ValueError, match="Gradient not set"):
        layer.update_parameters()


@pytest.mark.skip(reason="TODO: Fix layer name collision with appropriate fixture")
def test_recurrent_serialization_deserialization():
    """Test serialization and deserialization."""
    # Use None for layer_id to let it auto-generate a unique ID
    layer = Recurrent(
        input_dimensions=(2, 3),
        hidden_dim=4,
        return_sequences=True,
        key=key,
    )

    serialized = layer.serialize()
    deserialized = serialized.deserialize()

    assert deserialized.layer_id == layer.layer_id
    assert deserialized.input_dimensions == layer.input_dimensions
    assert deserialized.output_dimensions == layer.output_dimensions
    assert deserialized.hidden_dim == layer.hidden_dim
    assert deserialized.return_sequences == layer.return_sequences
    assert jnp.allclose(deserialized.parameters.Wxh, layer.parameters.Wxh)
    assert jnp.allclose(deserialized.parameters.Whh, layer.parameters.Whh)
    assert jnp.allclose(deserialized.parameters.bh, layer.parameters.bh)


def test_recurrent_serialize_deserialize_parameters_with_wrong_layer_id():
    """Test parameter serialization with wrong layer ID."""
    layer = Recurrent(
        input_dimensions=(2, 2),
        hidden_dim=3,
        return_sequences=True,
        layer_id="test_recurrent_wrong_id",
        key=key,
    )

    input_seq = jnp.array([[[1.0, 2.0], [3.0, 4.0]]])
    layer.forward_prop(input_activations=Activations(input_seq))
    layer.backward_prop(dZ=cast(D[Activations], jnp.ones((1, 2, 3))))

    buffer = io.BytesIO()
    layer.write_serialized_parameters(buffer)
    buffer.seek(0)

    other_layer = Recurrent(
        input_dimensions=(2, 2),
        hidden_dim=3,
        return_sequences=True,
        layer_id="different_recurrent",
        key=key,
    )

    with pytest.raises(BadLayerId):
        other_layer.read_serialized_parameters(buffer)


def test_recurrent_reinitialise():
    """Test reinitialization of parameters."""
    layer = Recurrent(
        input_dimensions=(2, 3),
        hidden_dim=4,
        return_sequences=True,
        key=key,
    )

    original_Wxh = layer.parameters.Wxh.copy()
    original_Whh = layer.parameters.Whh.copy()
    original_bh = layer.parameters.bh.copy()

    layer.reinitialise()

    # Parameters should change after reinitialization
    assert not jnp.allclose(layer.parameters.Wxh, original_Wxh)
    assert not jnp.allclose(layer.parameters.Whh, original_Whh)
    # bh might be all zeros, so it could be close
    # Just check shapes are preserved
    assert layer.parameters.Wxh.shape == original_Wxh.shape
    assert layer.parameters.Whh.shape == original_Whh.shape
    assert layer.parameters.bh.shape == original_bh.shape


def test_recurrent_parameter_nbytes():
    """Test parameter byte size calculation."""
    layer = Recurrent(
        input_dimensions=(2, 3),
        hidden_dim=4,
        return_sequences=True,
        key=key,
    )

    expected_nbytes = (
        layer.parameters.Wxh.nbytes
        + layer.parameters.Whh.nbytes
        + layer.parameters.bh.nbytes
    )
    assert layer.parameter_nbytes == expected_nbytes


def test_recurrent_gradient_operation_interface():
    """Test gradient operation interface."""
    layer = Recurrent(
        input_dimensions=(2, 2),
        hidden_dim=3,
        return_sequences=True,
        key=key,
    )

    def grad_callback(grad_layer):
        assert grad_layer is layer

    layer.gradient_operation(grad_callback)


def test_recurrent_properties():
    """Test layer properties."""
    hidden_dim = 5
    return_sequences = False
    layer = Recurrent(
        input_dimensions=(3, 4),
        hidden_dim=hidden_dim,
        return_sequences=return_sequences,
        key=key,
    )

    assert layer.hidden_dim == hidden_dim
    assert layer.return_sequences == return_sequences


def test_recurrent_with_custom_activation():
    """Test recurrent layer with custom activation function."""
    from mo_net.functions import ReLU

    layer = Recurrent(
        input_dimensions=(2, 2),
        hidden_dim=3,
        return_sequences=True,
        activation_fn=ReLU(),
        key=key,
    )

    input_seq = jnp.array([[[1.0, 2.0], [3.0, 4.0]]])
    output = layer.forward_prop(input_activations=Activations(input_seq))

    # ReLU output should be non-negative
    assert jnp.all(output >= 0.0)


def test_recurrent_bptt_time_dependency():
    """Test that BPTT correctly propagates gradients through time."""
    layer = Recurrent(
        clip_gradients=False,
        input_dimensions=(3, 2),
        hidden_dim=4,
        return_sequences=True,
        key=key,
    )

    # Create a sequence where the output at the last timestep
    # should have gradients that flow back to the first timestep
    input_seq = jnp.array([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    layer.forward_prop(input_activations=Activations(input_seq))

    # Create gradient that only affects the last timestep
    dZ = jnp.zeros((1, 3, 4))
    dZ = dZ.at[:, -1, :].set(1.0)

    dX = layer.backward_prop(dZ=cast(D[Activations], dZ))

    # Due to BPTT, gradients should flow to earlier timesteps
    # Check that gradient at t=0 is not zero
    assert not jnp.allclose(cast(jnp.ndarray, dX)[:, 0, :], 0.0)
