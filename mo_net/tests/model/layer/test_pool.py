from dataclasses import dataclass
from typing import Final

import jax.numpy as jnp
import pytest

from mo_net.model.layer.pool import MaxPooling2D
from mo_net.protos import Activations, Dimensions

TEST_INPUT: Final[jnp.ndarray] = jnp.ones(shape=(3, 3))


@pytest.mark.parametrize(
    ("pool_size", "stride", "X", "expected"),
    [
        (
            2,
            1,
            jnp.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]),
            jnp.array([[[[5, 6], [8, 9]]]]),
        ),
        (3, 1, jnp.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]), jnp.array([[[[9]]]])),
        (
            1,
            2,
            jnp.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]),
            jnp.array([[[[1, 3], [7, 9]]]]),
        ),
        (2, 2, jnp.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]), jnp.array([[[[5]]]])),
        (2, 2, jnp.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]), jnp.array([[[[5]]]])),
        # TODO: Add non-square input test cases
        (
            2,
            2,
            jnp.array(
                [
                    [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]],
                    [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]],
                ]
            ),
            jnp.array([[[[5]]], [[[5]]]]),
        ),
    ],
)
def test_max_pool_2d_forward_prop(
    pool_size: int, stride: int, X: Activations, expected: Activations
):
    pool_layer = MaxPooling2D(
        input_dimensions=(1, 3, 3), pool_size=pool_size, stride=stride
    )
    output = pool_layer.forward_prop(X)
    assert jnp.allclose(output, expected)


@dataclass(frozen=True)
class BackpropTestCase:
    name: str
    input_dimensions: Dimensions
    pool_size: int
    stride: int
    X: jnp.ndarray
    dZ: jnp.ndarray
    expected: jnp.ndarray


@pytest.mark.parametrize(
    "test_case",
    [
        BackpropTestCase(
            name="2x2_input_pool2_stride1",
            input_dimensions=(1, 2, 2),
            pool_size=2,
            stride=1,
            X=jnp.array([[[[1, 2], [3, 4]]]]),
            dZ=jnp.array([[[[1]]]]),
            expected=jnp.array([[[[0, 0], [0, 1]]]]),
        ),
        BackpropTestCase(
            name="3x3_input_pool2_stride1_uniform_values",
            input_dimensions=(1, 3, 3),
            pool_size=2,
            stride=1,
            X=jnp.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]),
            dZ=jnp.array([[[[1, 1], [1, 1]]]]),
            expected=jnp.array([[[[0, 0, 0], [0, 1, 1], [0, 1, 1]]]]),
        ),
        BackpropTestCase(
            name="3x3_input_pool2_stride1_single_max",
            input_dimensions=(1, 3, 3),
            pool_size=2,
            stride=1,
            X=jnp.array([[[[1, 2, 3], [4, 10, 6], [7, 8, 9]]]]),
            dZ=jnp.array([[[[1, 1], [1, 1]]]]),
            expected=jnp.array([[[[0, 0, 0], [0, 4, 0], [0, 0, 0]]]]),
        ),
        BackpropTestCase(
            name="4x3_input_pool2_stride1_single_max",
            input_dimensions=(1, 4, 3),
            pool_size=2,
            stride=1,
            X=jnp.array([[[[1, 2, 3], [4, 10, 6], [7, 8, 9], [10, 11, 12]]]]),
            dZ=jnp.array([[[[1, 1], [1, 1], [1, 1]]]]),
            expected=jnp.array([[[[0, 0, 0], [0, 4, 0], [0, 0, 0], [0, 1, 1]]]]),
        ),
    ],
    ids=lambda test_case: test_case.name,
)
def test_max_pool_2d_backward_prop(test_case: BackpropTestCase):
    pool_layer = MaxPooling2D(
        input_dimensions=test_case.input_dimensions,
        pool_size=test_case.pool_size,
        stride=test_case.stride,
    )
    pool_layer.forward_prop(Activations(test_case.X))
    dX = pool_layer.backward_prop(test_case.dZ)
    assert jnp.allclose(dX, test_case.expected)  # type: ignore[arg-type]
