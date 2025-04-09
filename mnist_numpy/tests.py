import numpy as np
import pytest
from more_itertools import one

from mnist_numpy.functions import ReLU
from mnist_numpy.model import (
    DenseLayer,
    MultiLayerPerceptronV2,
)


def test_init():
    model = MultiLayerPerceptronV2.of(layer_neuron_counts=(2, 2, 2))

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
    model = MultiLayerPerceptronV2.of(
        layer_neuron_counts=[n_neurons] * (n_hidden_layers + 2), output_layer_type="raw"
    )

    model.output_layer._parameters = DenseLayer.Parameters.eye(n_neurons)
    for layer in model.hidden_layers:
        layer._parameters = DenseLayer.Parameters.eye(n_neurons)

    input_ = np.array(list(range(n_neurons)))
    output = model.forward_prop(input_)
    assert np.allclose(output, input_)


@pytest.mark.parametrize("factor", [2, 3, 4])
def test_foward_prop_basic_math(factor: int):
    bias_1 = np.array([1, 2])
    bias_2 = np.array([1, 1])
    model = MultiLayerPerceptronV2.of(
        layer_neuron_counts=(5, 2, 2, 1), output_layer_type="raw"
    )
    model.hidden_layers[0]._parameters = DenseLayer.Parameters(
        _W=factor * np.array([[1, 1, 1, -2, 0], [1, 4, 1, 1, 0]]).T,
        _B=bias_1 * factor,
    )
    model.hidden_layers[1]._parameters = DenseLayer.Parameters(
        _W=np.eye((2)), _B=bias_2
    )
    model.output_layer._parameters = DenseLayer.Parameters(_W=np.eye(2), _B=np.zeros(2))

    input_ = np.array([1, -1, 2, 1, 0])
    output = model.forward_prop(input_)
    assert np.allclose(output, factor * bias_1 + bias_2)


@pytest.mark.parametrize(
    ("input_", "expected"), [(np.array([1]), 1), (np.array([-1]), 0)]
)
def test_ReLU(input_: np.ndarray, expected: float):
    model = MultiLayerPerceptronV2.of(
        layer_neuron_counts=(1, 1, 1), activation_fn=ReLU, output_layer_type="raw"
    )
    one(model.hidden_layers)._parameters = DenseLayer.Parameters.eye(1)
    model.output_layer._parameters = DenseLayer.Parameters.eye(1)
    output = model.forward_prop(input_)
    assert np.allclose(output, expected)


@pytest.fixture
def m1() -> MultiLayerPerceptronV2:
    model = MultiLayerPerceptronV2.of(
        layer_neuron_counts=(1, 1, 1, 1), output_layer_type="raw"
    )
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


def test_backward_prop_basic_math(m1: MultiLayerPerceptronV2):
    input = np.array([[1]])
    m1.forward_prop(input)
    backprop = m1.backward_prop(np.array([1]))

    assert backprop.dParams == (
        DenseLayer.Parameters(_W=np.array([[2]]), _B=np.array([2])),
        DenseLayer.Parameters(_W=np.array([[2]]), _B=np.array([2])),
        DenseLayer.Parameters(_W=np.array([[1]]), _B=np.array([1])),
    )


@pytest.fixture
def m2() -> MultiLayerPerceptronV2:
    model = MultiLayerPerceptronV2.of(
        layer_neuron_counts=(2, 2, 2),
    )
    hidden_layer = one(model.hidden_layers)
    hidden_layer._parameters = DenseLayer.Parameters(
        _W=np.array([[1, 0], [0, 1]]), _B=np.array([0, 0])
    )
    model.output_layer._parameters = DenseLayer.Parameters(
        _W=np.array([[1, 0], [0, 1]]), _B=np.array([0, 0])
    )
    return model


def test_backward_prop_update(m2: MultiLayerPerceptronV2):
    Y_true = np.array([[0, 1], [1, 0]])
    input_ = np.array([[1, 0], [0, 1]])
    learning_rate = 0.1

    for i in range(1000):
        m2.forward_prop(input_)
        backprop = m2.backward_prop(Y_true)
        m2.update_params(-learning_rate * backprop)

    assert np.allclose(m2.forward_prop(input_), Y_true, atol=1e-2)


@pytest.fixture
def m3() -> MultiLayerPerceptronV2:
    model = MultiLayerPerceptronV2.of(
        layer_neuron_counts=(2, 2, 2, 2),
    )
    hidden_layer_1, hidden_layer_2 = model.hidden_layers
    hidden_layer_1._parameters = DenseLayer.Parameters(
        _W=np.array([[1, 1], [1, 1]]), _B=np.array([0, 0])
    )
    hidden_layer_2._parameters = DenseLayer.Parameters(
        _W=np.array([[1, 1], [1, 1]]), _B=np.array([0, 0])
    )
    model.output_layer._parameters = DenseLayer.Parameters(
        _W=np.array([[1, 1], [1, 1]]), _B=np.array([0, 0])
    )
    return model


def test_backward_prop_update_deeper(m3: MultiLayerPerceptronV2):
    Y_true = np.array([[0.2, 0.8], [0.8, 0.2]])
    input_ = np.array([[1, 0], [0, 1]])
    learning_rate = 0.1

    for i in range(1000):
        m3.forward_prop(input_)
        backprop = m3.backward_prop(Y_true)
        boost = 0.1 / np.max(np.concat([dP._W.flatten() for dP in backprop.dParams]))
        m3.update_params(-learning_rate * backprop * boost)

    assert np.allclose(m3.forward_prop(input_), Y_true, atol=0.1)
