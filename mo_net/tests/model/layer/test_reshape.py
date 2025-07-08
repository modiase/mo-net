import jax.numpy as jnp
import pytest

from mo_net.model.layer.reshape import Flatten, Reshape
from mo_net.protos import Activations, Dimensions


def test_reshape_layer_forward_prop():
    X = jnp.array([[1, 2, 3, 4], [1, 2, 3, 4]])
    reshape = Reshape(input_dimensions=(4,), output_dimensions=(2, 2))
    assert jnp.allclose(
        reshape.forward_prop(input_activations=X), X.reshape(X.shape[0], 2, 2)
    )


def test_reshape_layer_backward_prop():
    reshape = Reshape(input_dimensions=(4,), output_dimensions=(2, 2))
    dZ = jnp.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]]])
    assert reshape._backward_prop(dZ=dZ).shape == (2, 4)


@pytest.mark.parametrize(
    (
        "input_dimensions",
        "batch_size",
        "expected_output_shape",
    ),
    [
        (
            (1, 2, 2),
            2,
            (2, 4),
        ),
        (
            (3, 2, 2),
            1,
            (1, 12),
        ),
        (
            (2, 3, 4, 5),
            10,
            (10, 120),
        ),
    ],
)
def test_flatten_layer_forward_prop(
    input_dimensions: Dimensions,
    batch_size: int,
    expected_output_shape: tuple[int, int],
):
    flatten = Flatten(input_dimensions=input_dimensions)
    X = Activations(jnp.ones((batch_size, *input_dimensions)))
    output = flatten.forward_prop(X)
    assert jnp.allclose(output, jnp.ones(expected_output_shape))


def test_flatten_layer_backward_prop():
    flatten = Flatten(input_dimensions=((3, 3)))
    dZ = jnp.arange(9).reshape((1, 9))
    assert jnp.allclose(flatten.backward_prop(dZ=dZ), jnp.arange(9).reshape((1, 3, 3)))
