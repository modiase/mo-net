import numpy as np
import pytest

from mnist_numpy.model import (
    MultiLayerPerceptronV2,
)
from mnist_numpy.model.layer import DenseParameters


def eye_params(n: int) -> DenseParameters:
    return DenseParameters(_W=np.eye(n), _B=np.zeros(n))


def test_init():
    model = MultiLayerPerceptronV2.of((2, 2, 2))

    assert model.input_layer.neurons == 2
    assert tuple(layer.neurons for layer in model.hidden_layers) == (2,)
    assert model.output_layer.neurons == 2

    assert model.input_layer.parameters is None
    assert model.hidden_layers[0].parameters._W.shape == (2, 2)
    assert model.hidden_layers[0].parameters._B.shape == (2,)
    assert model.output_layer.parameters._W.shape == (2, 2)
    assert model.output_layer.parameters._B.shape == (2,)


@pytest.mark.parametrize("n_hidden_layers", [1, 2, 3])
@pytest.mark.parametrize("n_neurons", [2, 3, 4])
def test_forward_prop_eye(n_hidden_layers: int, n_neurons: int):
    model = MultiLayerPerceptronV2.of([n_neurons] * (n_hidden_layers + 2))

    model.output_layer._parameters = eye_params(n_neurons)
    for layer in model.hidden_layers:
        layer._parameters = eye_params(n_neurons)

    input = np.array(list(range(n_neurons)))
    output = model.forward_prop(input)
    assert np.allclose(output, input)


@pytest.mark.parametrize("factor", [2, 3, 4])
def test_foward_prop_basic_math(factor: int):
    bias_1 = np.array([1, 2])
    bias_2 = np.array([1, 1])
    model = MultiLayerPerceptronV2.of((5, 2, 2, 1))
    model._layers[1]._parameters = DenseParameters(
        _W=factor * np.array([[1, 1, 1, -2, 0], [1, 4, 1, 1, 0]]).T,
        _B=bias_1 * factor,
    )
    model._layers[2]._parameters = DenseParameters(_W=np.eye((2)), _B=bias_2)
    model.output_layer._parameters = DenseParameters(_W=np.eye(2), _B=np.zeros(2))

    input = np.array([1, -1, 2, 1, 0])
    output = model.forward_prop(input)
    assert np.allclose(output, factor * bias_1 + bias_2)
