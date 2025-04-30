import io
import pickle

import numpy as np
import pytest

from mnist_numpy.functions import ReLU
from mnist_numpy.model import (
    MultiLayerPerceptron,
)
from mnist_numpy.model.block.base import Hidden, Output
from mnist_numpy.model.layer import (
    Activation,
)
from mnist_numpy.model.layer.dense import Dense
from mnist_numpy.model.layer.output import (
    RawOutputLayer,
    SoftmaxOutputLayer,
)
from mnist_numpy.optimizer.adam import AdaM
from mnist_numpy.optimizer.base import Null
from mnist_numpy.optimizer.scheduler import ConstantScheduler


def test_init():
    model = MultiLayerPerceptron.of(dimensions=(2, 2, 2))

    assert model.input_dimensions == 2
    assert model.output_dimensions == 2

    assert model.hidden_blocks[0].layers[0]._parameters._W.shape == (2, 2)
    assert model.hidden_blocks[0].layers[0]._parameters._B.shape == (2,)
    assert model.output_block.layers[0]._parameters._W.shape == (2, 2)
    assert model.output_block.layers[0]._parameters._B.shape == (2,)


@pytest.mark.parametrize("n_hidden_layers", [1, 2, 3])
@pytest.mark.parametrize("n_neurons", [2, 3, 4])
def test_forward_prop_eye(n_hidden_layers: int, n_neurons: int):
    model = MultiLayerPerceptron(
        input_dimensions=n_neurons,
        hidden_blocks=tuple(
            Hidden(
                layers=[
                    Dense(
                        input_dimensions=n_neurons,
                        output_dimensions=n_neurons,
                        parameters=Dense.Parameters.eye(n_neurons),
                    ),
                ],
            )
            for _ in range(n_hidden_layers)
        ),
        output_block=Output(
            layers=[
                Dense(
                    input_dimensions=n_neurons,
                    output_dimensions=n_neurons,
                    parameters=Dense.Parameters.eye(n_neurons),
                ),
            ],
            output_layer=RawOutputLayer(input_dimensions=n_neurons),
        ),
    )

    X = np.atleast_2d(np.array(range(n_neurons)))
    output = model.forward_prop(X)
    assert np.allclose(output, np.atleast_2d(np.array(range(n_neurons))))


@pytest.mark.parametrize("factor", [2, 3, 4])
def test_forward_prop_basic_math(factor: int):
    bias_1 = np.array([1, 2])
    bias_2 = np.array([1, 1])
    model = MultiLayerPerceptron(
        input_dimensions=5,
        hidden_blocks=tuple(
            [
                Hidden(
                    layers=[
                        Dense(
                            input_dimensions=5,
                            output_dimensions=2,
                            parameters=Dense.Parameters(
                                _W=factor
                                * np.array([[1, 1, 1, -2, 0], [1, 4, 1, 1, 0]]).T,
                                _B=bias_1 * factor,
                            ),
                        ),
                        Dense(
                            input_dimensions=2,
                            output_dimensions=2,
                            parameters=Dense.Parameters(
                                _W=np.eye((2)),
                                _B=bias_2,
                            ),
                        ),
                    ]
                )
            ]
        ),
        output_block=Output(
            layers=[
                Dense(
                    input_dimensions=2,
                    output_dimensions=2,
                    parameters=Dense.Parameters(
                        _W=np.eye(2),
                        _B=np.zeros(2),
                    ),
                )
            ],
            output_layer=RawOutputLayer(input_dimensions=2),
        ),
    )

    X = np.array([1, -1, 2, 1, 0])
    output = model.forward_prop(X)
    assert np.allclose(output, factor * bias_1 + bias_2)


@pytest.mark.parametrize(("X", "expected"), [(np.array([1]), 1), (np.array([-1]), 0)])
def test_forward_prop_ReLU(X: np.ndarray, expected: float):
    model = MultiLayerPerceptron(
        input_dimensions=1,
        hidden_blocks=tuple(
            [
                Hidden(
                    layers=[
                        Dense(
                            input_dimensions=1,
                            output_dimensions=1,
                            parameters=Dense.Parameters.eye(1),
                        ),
                        Activation(
                            input_dimensions=1,
                            activation_fn=ReLU,
                        ),
                    ],
                )
            ]
        ),
        output_block=Output(
            layers=[
                Dense(
                    input_dimensions=1,
                    output_dimensions=1,
                    parameters=Dense.Parameters.eye(1),
                )
            ],
            output_layer=RawOutputLayer(input_dimensions=1),
        ),
    )
    output = model.forward_prop(X)
    assert np.allclose(output, expected)


@pytest.fixture
def m1() -> MultiLayerPerceptron:
    return MultiLayerPerceptron(
        input_dimensions=1,
        hidden_blocks=tuple(
            [
                Hidden(
                    layers=[
                        Dense(
                            input_dimensions=1,
                            output_dimensions=1,
                            parameters=Dense.Parameters.eye(1),
                        )
                        for _ in range(2)
                    ]
                )
            ]
        ),
        output_block=Output(
            layers=[
                Dense(
                    input_dimensions=1,
                    output_dimensions=1,
                    parameters=Dense.Parameters(_W=np.array([[2]]), _B=np.array([0])),
                )
            ],
            output_layer=RawOutputLayer(input_dimensions=1),
        ),
    )


def test_backward_prop_basic_math(m1: MultiLayerPerceptron):
    X = Y_true = np.array([[1]])
    m1.forward_prop(X)
    m1.backward_prop(Y_true)

    assert tuple(
        [
            layer._cache["dP"]
            for block in m1.blocks
            for layer in block.layers
            if isinstance(layer, Dense)
        ]
    ) == (
        Dense.Parameters(_W=np.array([[2]]), _B=np.array([2])),
        Dense.Parameters(_W=np.array([[2]]), _B=np.array([2])),
        Dense.Parameters(_W=np.array([[1]]), _B=np.array([1])),
    )


@pytest.fixture
def m2() -> MultiLayerPerceptron:
    return MultiLayerPerceptron(
        input_dimensions=2,
        hidden_blocks=tuple(
            [
                Hidden(
                    layers=[
                        Dense(
                            input_dimensions=2,
                            output_dimensions=2,
                            parameters=Dense.Parameters(
                                _W=np.array([[1, 0], [0, 1]]), _B=np.array([0, 0])
                            ),
                        )
                    ]
                )
            ]
        ),
        output_block=Output(
            layers=[
                Dense(
                    input_dimensions=2,
                    output_dimensions=2,
                    parameters=Dense.Parameters(
                        _W=np.array([[1, 0], [0, 1]]), _B=np.array([0, 0])
                    ),
                )
            ],
            output_layer=SoftmaxOutputLayer(input_dimensions=2),
        ),
    )


def test_backward_prop_update(m2: MultiLayerPerceptron):
    X = np.array([[1, 0], [0, 1]])
    Y_true = np.array([[0.5, 0.5], [0.5, 0.5]])
    optimizer = Null(model=m2, config=Null.Config(learning_rate=0.1))

    for i in range(1000):
        optimizer.training_step(X_train_batch=X, Y_train_batch=Y_true)

    assert np.allclose(m2.forward_prop(X), Y_true, atol=1e-2)


@pytest.fixture
def m3() -> MultiLayerPerceptron:
    return MultiLayerPerceptron(
        input_dimensions=2,
        hidden_blocks=tuple(
            [
                Hidden(
                    layers=[
                        Dense(
                            input_dimensions=2,
                            output_dimensions=2,
                            parameters=Dense.Parameters(
                                _W=np.array([[1, 1], [1, 1]]), _B=np.array([0, 0])
                            ),
                        ),
                    ]
                ),
                Hidden(
                    layers=[
                        Dense(
                            input_dimensions=2,
                            output_dimensions=2,
                            parameters=Dense.Parameters(
                                _W=np.array([[1, 1], [1, 1]]), _B=np.array([0, 0])
                            ),
                        ),
                    ]
                ),
            ]
        ),
        output_block=Output(
            layers=[
                Dense(
                    input_dimensions=2,
                    output_dimensions=2,
                    parameters=Dense.Parameters(
                        _W=np.array([[1, 1], [1, 1]]), _B=np.array([0, 0])
                    ),
                )
            ],
            output_layer=SoftmaxOutputLayer(input_dimensions=2),
        ),
    )


@pytest.mark.skip(
    "This test is not currently testing anything. We need to think about a better test."
)
def test_backward_prop_update_deeper(m3: MultiLayerPerceptron, delta: float):
    X = np.array([[1, 1], [1, 1]])
    Y_true = np.array([[-1, -1], [-1, -1]])
    optimizer = AdaM(
        model=m3,
        config=AdaM.Config(
            scheduler=ConstantScheduler(learning_rate=0.1),
        ),
    )

    for i in range(1000):
        optimizer.training_step(X_train_batch=X, Y_train_batch=Y_true)

    assert np.allclose(m3.forward_prop(X), Y_true, atol=0.01)


@pytest.mark.parametrize("modelname", ["m1", "m2", "m3"])
def test_serialize_deserialize(modelname: str, request: pytest.FixtureRequest):
    model: MultiLayerPerceptron = request.getfixturevalue(modelname)
    X = np.ones((1, model.input_dimensions))

    X_prop_before = model.forward_prop(X)
    buffer = io.BytesIO()
    buffer.write(pickle.dumps(model.serialize()))
    buffer.seek(0)
    deserialized = MultiLayerPerceptron.load(buffer)
    X_prop_after = deserialized.forward_prop(X)

    assert model.dimensions == deserialized.dimensions
    assert np.allclose(X_prop_before, X_prop_after)


def test_adam_optimizer(m3: MultiLayerPerceptron):
    X = np.array([[1, 0], [0, 1]])
    Y_true = np.array([[0.8, 0.2], [0.2, 0.8]])
    optimizer = AdaM(
        model=m3,
        config=AdaM.Config(
            scheduler=ConstantScheduler(learning_rate=0.01),
        ),
    )
    loss_before = m3.compute_loss(X=X, Y_true=Y_true)
    for i in range(100):
        optimizer.training_step(X_train_batch=X, Y_train_batch=Y_true)
    loss_after = m3.compute_loss(X=X, Y_true=Y_true)
    assert loss_before > loss_after, loss_before - loss_after
