import numpy as np
import pytest

from mnist_numpy.model.block.base import Hidden
from mnist_numpy.model.layer.linear import Linear
from mnist_numpy.model.model import Model


def test_model_initialisation_of_mlp_has_correct_dimensions():
    model = Model.mlp_of(module_dimensions=((2,), (2,), (2,)))

    assert model.input_dimensions == (2,)
    assert model.output_dimensions == (2,)

    assert model.hidden_blocks[0].layers[0]._parameters._W.shape == (2, 2)
    assert model.hidden_blocks[0].layers[0]._parameters._B.shape == (2,)
    assert model.output_block.layers[0]._parameters._W.shape == (2, 2)
    assert model.output_block.layers[0]._parameters._B.shape == (2,)


@pytest.mark.parametrize("n_hidden_layers", [1, 2, 3])
@pytest.mark.parametrize("n_neurons", [2, 3, 4])
def test_forward_prop_identity(n_hidden_layers: int, n_neurons: int):
    """
    Test that the forward propagation of an identity model is the identity.
    """
    model = Model(
        input_dimensions=(n_neurons,),
        hidden=[
            Hidden(layers=[Linear.of_eye(dim=(n_neurons,))])
            for _ in range(n_hidden_layers)
        ],
    )

    X = np.atleast_2d(np.array(range(n_neurons)))
    output = model.forward_prop(X)
    assert np.allclose(output, np.atleast_2d(np.array(range(n_neurons))))


@pytest.mark.parametrize("factor", [2, 3, 4])
@pytest.mark.parametrize("dX", [np.zeros(5), np.ones(5)])
def test_forward_prop_linear_model(factor: int, dX: np.ndarray):
    """
    Test that the forward propagation of a linear model is a linear function.
    => L(X + dX) = L(X) + dL(X)
    => L(kX) = kL(X)
    """
    X = np.array([1, -1, 2, 1, 0])
    weights = np.array([[1, 1, 1, -2, 0], [1, 4, 1, 1, 0]]).T
    bias_1 = np.array([1, 1])
    model = Model(
        input_dimensions=(5,),
        hidden=[
            Linear(
                input_dimensions=(5,),
                output_dimensions=(2,),
                parameters=Linear.Parameters(
                    _W=weights,
                    _B=bias_1,
                ),
            ),
        ],
    )

    output = model.forward_prop(factor * (X + dX))
    assert np.allclose(
        output,
        np.array(
            [
                factor
                * (
                    (X[0] + dX[0]) * weights[0, 0]
                    + (X[1] + dX[1]) * weights[1, 0]
                    + (X[2] + dX[2]) * weights[2, 0]
                    + (X[3] + dX[3]) * weights[3, 0]
                    + (X[4] + dX[4]) * weights[4, 0]
                )
                + bias_1[0],
                factor
                * (
                    (X[0] + dX[0]) * weights[0, 1]
                    + (X[1] + dX[1]) * weights[1, 1]
                    + (X[2] + dX[2]) * weights[2, 1]
                    + (X[3] + dX[3]) * weights[3, 1]
                    + (X[4] + dX[4]) * weights[4, 1]
                )
                + bias_1[1],
            ]
        ),
    )
