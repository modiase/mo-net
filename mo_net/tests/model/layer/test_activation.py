import numpy as np
import pytest

from mo_net.functions import ReLU
from mo_net.model.layer.activation import Activation
from mo_net.protos import Activations


@pytest.mark.parametrize(
    ("X", "expected"),
    [
        (np.array([1, 2, 3]), np.array([1, 2, 3])),
        (np.array([-1, -2, -3]), np.array([0, 0, 0])),
    ],
)
def test_relu_forward_prop(X: Activations, expected: np.ndarray):
    activation = Activation(
        input_dimensions=(3,),
        activation_fn=ReLU,
    )
    assert np.allclose(activation.forward_prop(X), expected)


@pytest.mark.parametrize(
    ("X", "expected"),
    [
        (np.array([1, 2, 3]), np.ones(3)),
        (np.array([-1, -2, -3]), np.zeros(3)),
    ],
)
def test_relu_backward_prop(X: Activations, expected: np.ndarray):
    activation = Activation(
        input_dimensions=(3,),
        activation_fn=ReLU,
    )
    activation.forward_prop(X)

    assert np.allclose(activation.backward_prop(np.ones(3)), expected)  # type: ignore[arg-type]
