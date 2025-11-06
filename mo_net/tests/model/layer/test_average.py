from typing import cast

import jax.numpy as jnp
import numpy.testing as np_testing
import pytest

from mo_net.model.layer.average import Average
from mo_net.protos import Activations, D


def test_average_forward_axis0():
    layer = Average(input_dimensions=(2, 3), axis=0)
    x = jnp.array([[[1, 2, 3], [4, 5, 6]]])
    out = layer.forward_prop(input_activations=Activations(x))
    np_testing.assert_array_equal(out, jnp.mean(x, axis=1))
    assert out.shape == (1, 3)


def test_average_forward_axis1():
    layer = Average(input_dimensions=(2, 3), axis=1)
    x = jnp.array([[[1, 2, 3], [4, 5, 6]]])
    out = layer.forward_prop(input_activations=Activations(x))
    np_testing.assert_array_equal(out, jnp.mean(x, axis=2))
    assert out.shape == (1, 2)


def test_average_forward_multi_axis():
    layer = Average(input_dimensions=(2, 3, 4), axis=(1, 2))
    x = jnp.arange(24).reshape(1, 2, 3, 4)
    out = layer.forward_prop(input_activations=Activations(x))
    np_testing.assert_array_equal(out, jnp.mean(x, axis=(2, 3)))
    assert out.shape == (1, 2)


def test_average_backward_axis0():
    layer = Average(input_dimensions=(2, 3), axis=0)
    x = jnp.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    out = layer.forward_prop(input_activations=Activations(x))
    grad_out = jnp.ones_like(out)
    grad_in = layer.backward_prop(dZ=cast(D[Activations], grad_out))
    # Each input element gets 1/2 of the gradient for its column
    np_testing.assert_allclose(cast(jnp.ndarray, grad_in), jnp.ones_like(x) * 0.5)
    assert cast(jnp.ndarray, grad_in).shape == x.shape


def test_average_backward_axis1():
    layer = Average(input_dimensions=(2, 3), axis=1)
    x = jnp.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    out = layer.forward_prop(input_activations=Activations(x))
    grad_out = jnp.ones_like(out)
    grad_in = layer.backward_prop(dZ=cast(D[Activations], grad_out))
    # Each input element gets 1/3 of the gradient for its row
    np_testing.assert_allclose(cast(jnp.ndarray, grad_in), jnp.ones_like(x) / 3)
    assert cast(jnp.ndarray, grad_in).shape == x.shape


def test_average_backward_multi_axis():
    layer = Average(input_dimensions=(2, 3, 4), axis=(1, 2))
    x = jnp.arange(24.0).reshape(1, 2, 3, 4)
    out = layer.forward_prop(input_activations=Activations(x))
    grad_out = jnp.ones_like(out)
    grad_in = layer.backward_prop(dZ=cast(D[Activations], grad_out))
    # Each input element gets 1/(3*4) of the gradient for its batch
    np_testing.assert_allclose(cast(jnp.ndarray, grad_in), jnp.ones_like(x) / 12)
    assert cast(jnp.ndarray, grad_in).shape == x.shape


def test_average_serialize_and_deserialize():
    layer = Average(input_dimensions=(2, 3), axis=1)
    ser = layer.serialize()
    assert ser.input_dimensions == (2, 3)
    assert ser.axis == (1,)
    # Deserialization should reconstruct the layer
    new_layer = ser.deserialize()
    assert new_layer.input_dimensions == (2, 3)
    assert new_layer.axis == (1,)


def test_average_invalid_axis():
    # Axis out of range
    with pytest.raises(IndexError):
        Average(input_dimensions=(2, 3), axis=2)

    # Axis as empty tuple
    with pytest.raises(ValueError):
        Average(input_dimensions=(2, 3), axis=())
