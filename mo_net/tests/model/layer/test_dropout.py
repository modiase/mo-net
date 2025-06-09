import sys
from typing import Final

import numpy as np
import pytest

from mo_net.model.layer.dropout import Dropout
from mo_net.protos import Activations

TEST_INPUT: Final[np.ndarray] = Activations(np.array([1, 2, 3]))


@pytest.mark.parametrize(
    ("keep_prob", "expected"),
    [
        (1.0, np.array([1, 2, 3])),
        (2**sys.float_info.min_exp, np.array([0, 0, 0])),
    ],
)
def test_dropout_forward_prop(keep_prob: float, expected: np.ndarray):
    dropout = Dropout(
        input_dimensions=TEST_INPUT.shape,
        keep_prob=keep_prob,
        training=True,
    )
    assert np.allclose(dropout.forward_prop(TEST_INPUT), expected)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("keep_prob", "expected"),
    [
        (1.0, np.ones(TEST_INPUT.shape)),
        (2**sys.float_info.min_exp, np.zeros(TEST_INPUT.shape)),
    ],
)
def test_dropout_backward_prop(keep_prob: float, expected: np.ndarray):
    dropout = Dropout(
        input_dimensions=TEST_INPUT.shape,
        keep_prob=keep_prob,
        training=True,
    )
    dropout.forward_prop(TEST_INPUT)  # type: ignore[arg-type]
    assert np.allclose(dropout.backward_prop(dZ=np.ones(TEST_INPUT.shape)), expected)  # type: ignore[arg-type]
