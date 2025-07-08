import jax
import jax.numpy as jnp
import pytest

from mo_net.model.layer.activation import Activation
from mo_net.protos import Activations


@pytest.mark.parametrize(
    ("X", "expected"),
    [
        (jnp.array([1, 2, 3]), jnp.array([1, 2, 3])),
        (jnp.array([-1, -2, -3]), jnp.array([0, 0, 0])),
    ],
)
def test_relu_forward_prop(X: Activations, expected: jnp.ndarray):
    activation = Activation(
        input_dimensions=(3,),
        activation_fn=jax.nn.relu,
    )
    assert jnp.allclose(activation.forward_prop(X), expected)


@pytest.mark.parametrize(
    ("X", "expected"),
    [
        (jnp.array([1, 2, 3]), jnp.ones(3)),
        (jnp.array([-1, -2, -3]), jnp.zeros(3)),
    ],
)
def test_relu_backward_prop(X: Activations, expected: jnp.ndarray):
    activation = Activation(
        input_dimensions=(3,),
        activation_fn=jax.nn.relu,
    )
    activation.forward_prop(X)

    assert jnp.allclose(activation.backward_prop(jnp.ones(3)), expected)  # type: ignore[arg-type]
