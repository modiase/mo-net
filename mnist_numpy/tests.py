import numpy as np
import pytest
from more_itertools import one

from mnist_numpy.functions import ReLU
from mnist_numpy.model import (
    DenseLayer,
    MultiLayerPerceptronV2,
)


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

    model.output_layer._parameters = DenseLayer.Parameters.eye(n_neurons)
    for layer in model.hidden_layers:
        layer._parameters = DenseLayer.Parameters.eye(n_neurons)

    input = np.array(list(range(n_neurons)))
    output = model.forward_prop(input)
    assert np.allclose(output, input)


@pytest.mark.parametrize("factor", [2, 3, 4])
def test_foward_prop_basic_math(factor: int):
    bias_1 = np.array([1, 2])
    bias_2 = np.array([1, 1])
    model = MultiLayerPerceptronV2.of((5, 2, 2, 1))
    model._layers[1]._parameters = DenseLayer.Parameters(
        _W=factor * np.array([[1, 1, 1, -2, 0], [1, 4, 1, 1, 0]]).T,
        _B=bias_1 * factor,
    )
    model._layers[2]._parameters = DenseLayer.Parameters(_W=np.eye((2)), _B=bias_2)
    model.output_layer._parameters = DenseLayer.Parameters(_W=np.eye(2), _B=np.zeros(2))

    input = np.array([1, -1, 2, 1, 0])
    output = model.forward_prop(input)
    assert np.allclose(output, factor * bias_1 + bias_2)


@pytest.mark.parametrize(
    ("input", "expected"), [(np.array([1]), 1), (np.array([-1]), 0)]
)
def test_ReLU(input: np.ndarray, expected: float):
    model = MultiLayerPerceptronV2.of((1, 1, 1), activation_fn=ReLU)
    one(model.hidden_layers)._parameters = DenseLayer.Parameters.eye(1)
    model.output_layer._parameters = DenseLayer.Parameters.eye(1)
    output = model.forward_prop(input)
    assert np.allclose(output, expected)


@pytest.fixture
def m1():
    model = MultiLayerPerceptronV2.of((1, 1, 1, 1))
    hidden_layer_1, hidden_layer_2 = model.hidden_layers
    hidden_layer_1._parameters = DenseLayer.Parameters(
        _W=np.array([[1]]), _B=np.array([0])
    )
    hidden_layer_2._parameters = DenseLayer.Parameters(
        _W=np.array([[1]]), _B=np.array([0])
    )
    model.output_layer._parameters = DenseLayer.Parameters(
        _W=np.array([[2]]), _B=np.array([0])
    )
    return model


def test_backward_prop_basic_math(m1):
    input = np.array([[1]])
    m1.forward_prop(input)
    backprop = m1.backward_prop(np.array([1]))

    assert backprop == (
        DenseLayer.Parameters(_W=np.array([[-2]]), _B=np.array([-2])),
        DenseLayer.Parameters(_W=np.array([[-2]]), _B=np.array([-2])),
        DenseLayer.Parameters(_W=np.array([[-2]]), _B=np.array([-1])),
    )
