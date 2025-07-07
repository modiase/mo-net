import sys
from typing import Final

import jax.numpy as jnp
import pytest

from mo_net.model.layer.dropout import Dropout
from mo_net.protos import Activations

TEST_INPUT: Final[jnp.ndarray] = Activations(jnp.array([1, 2, 3]))


@pytest.mark.parametrize(
    ("keep_prob", "expected"),
    [
        (1.0, jnp.array([1, 2, 3])),
        (2**sys.float_info.min_exp, jnp.array([0, 0, 0])),
    ],
)
def test_dropout_forward_prop(keep_prob: float, expected: jnp.ndarray):
    dropout = Dropout(
        input_dimensions=TEST_INPUT.shape,
        keep_prob=keep_prob,
        training=True,
    )
    assert jnp.allclose(dropout.forward_prop(TEST_INPUT), expected)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("keep_prob", "expected"),
    [
        (1.0, jnp.ones(TEST_INPUT.shape)),
        (2**sys.float_info.min_exp, jnp.zeros(TEST_INPUT.shape)),
    ],
)
def test_dropout_backward_prop(keep_prob: float, expected: jnp.ndarray):
    dropout = Dropout(
        input_dimensions=TEST_INPUT.shape,
        keep_prob=keep_prob,
        training=True,
    )
    dropout.forward_prop(TEST_INPUT)  # type: ignore[arg-type]
    assert jnp.allclose(dropout.backward_prop(dZ=jnp.ones(TEST_INPUT.shape)), expected)  # type: ignore[arg-type]
