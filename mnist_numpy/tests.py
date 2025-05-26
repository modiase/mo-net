import io
import pickle

import numpy as np
import pytest
from more_itertools import one

from mnist_numpy.functions import ReLU
from mnist_numpy.model import (
    Model,
)
from mnist_numpy.model.block.base import Hidden, Output
from mnist_numpy.model.layer import (
    Activation,
)
from mnist_numpy.model.layer.convolution import Convolution2D
from mnist_numpy.model.layer.linear import Linear
from mnist_numpy.model.layer.output import (
    RawOutputLayer,
    SoftmaxOutputLayer,
)
from mnist_numpy.model.layer.pool import MaxPooling2D
from mnist_numpy.model.layer.reshape import Flatten, Reshape
from mnist_numpy.optimizer.adam import AdaM
from mnist_numpy.optimizer.base import Null
from mnist_numpy.optimizer.scheduler import ConstantScheduler
from mnist_numpy.protos import Activations, D, Dimensions, GradLayer


@pytest.mark.parametrize(("X", "expected"), [(np.array([1]), 1), (np.array([-1]), 0)])
def test_forward_prop_ReLU(X: np.ndarray, expected: float):
    model = Model(
        input_dimensions=(1,),
        hidden=tuple(
            [
                Hidden(
                    layers=[
                        Linear(
                            input_dimensions=(1,),
                            output_dimensions=(1,),
                            parameters=Linear.Parameters.eye((1,)),
                        ),
                        Activation(
                            input_dimensions=(1,),
                            activation_fn=ReLU,
                        ),
                    ],
                )
            ]
        ),
        output=Output(
            layers=[
                Linear(
                    input_dimensions=(1,),
                    output_dimensions=(1,),
                    parameters=Linear.Parameters.eye((1,)),
                )
            ],
            output_layer=RawOutputLayer(input_dimensions=(1,)),
        ),
    )
    output = model.forward_prop(X)
    assert np.allclose(output, expected)


@pytest.fixture
def m1() -> Model:
    return Model(
        input_dimensions=(1,),
        hidden=tuple(
            [
                Hidden(
                    layers=[
                        Linear(
                            input_dimensions=(1,),
                            output_dimensions=(1,),
                            parameters=Linear.Parameters.eye((1,)),
                        )
                        for _ in range(2)
                    ]
                )
            ]
        ),
        output=Output(
            layers=[
                Linear(
                    input_dimensions=(1,),
                    output_dimensions=(1,),
                    parameters=Linear.Parameters(_W=np.array([[2]]), _B=np.array([0])),
                )
            ],
            output_layer=RawOutputLayer(input_dimensions=(1,)),
        ),
    )


def test_backward_prop_basic_math(m1: Model):
    X = Y_true = np.array([[1]])
    m1.forward_prop(X)
    m1.backward_prop(Y_true)

    assert tuple(
        [
            layer._cache["dP"]
            for block in m1.blocks
            for layer in block.layers
            if isinstance(layer, Linear)
        ]
    ) == (
        Linear.Parameters(_W=np.array([[2]]), _B=np.array([2])),
        Linear.Parameters(_W=np.array([[2]]), _B=np.array([2])),
        Linear.Parameters(_W=np.array([[1]]), _B=np.array([1])),
    )


@pytest.fixture
def m2() -> Model:
    return Model(
        input_dimensions=(2,),
        hidden=tuple(
            [
                Hidden(
                    layers=[
                        Linear(
                            input_dimensions=(2,),
                            output_dimensions=(2,),
                            parameters=Linear.Parameters(
                                _W=np.array([[1, 0], [0, 1]]), _B=np.array([0, 0])
                            ),
                        )
                    ]
                )
            ]
        ),
        output=Output(
            layers=[
                Linear(
                    input_dimensions=(2,),
                    output_dimensions=(2,),
                    parameters=Linear.Parameters(
                        _W=np.array([[1, 0], [0, 1]]), _B=np.array([0, 0])
                    ),
                )
            ],
            output_layer=SoftmaxOutputLayer(input_dimensions=(2,)),
        ),
    )


def test_backward_prop_update(m2: Model):
    X = np.array([[1, 0], [0, 1]])
    Y_true = np.array([[0.5, 0.5], [0.5, 0.5]])
    optimizer = Null(model=m2, config=Null.Config(learning_rate=0.1))

    for i in range(1000):
        optimizer.training_step(X_train_batch=X, Y_train_batch=Y_true)

    assert np.allclose(m2.forward_prop(X), Y_true, atol=1e-2)


@pytest.fixture
def m3() -> Model:
    return Model(
        input_dimensions=(2,),
        hidden=tuple(
            [
                Hidden(
                    layers=[
                        Linear(
                            input_dimensions=(2,),
                            output_dimensions=(2,),
                            parameters=Linear.Parameters(
                                _W=np.array([[1, 1], [1, 1]]), _B=np.array([0, 0])
                            ),
                        ),
                    ]
                ),
                Hidden(
                    layers=[
                        Linear(
                            input_dimensions=(2,),
                            output_dimensions=(2,),
                            parameters=Linear.Parameters(
                                _W=np.array([[1, 1], [1, 1]]), _B=np.array([0, 0])
                            ),
                        ),
                    ]
                ),
            ]
        ),
        output=Output(
            layers=[
                Linear(
                    input_dimensions=(2,),
                    output_dimensions=(2,),
                    parameters=Linear.Parameters(
                        _W=np.array([[1, 1], [1, 1]]), _B=np.array([0, 0])
                    ),
                )
            ],
            output_layer=SoftmaxOutputLayer(input_dimensions=(2,)),
        ),
    )


@pytest.mark.skip(
    "This test is not currently testing anything. We need to think about a better test."
)
def test_backward_prop_update_deeper(m3: Model, delta: float):
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
    model: Model = request.getfixturevalue(modelname)
    X = np.ones((1, one(model.input_dimensions)))

    X_prop_before = model.forward_prop(X)
    buffer = io.BytesIO()
    buffer.write(pickle.dumps(model.serialize()))
    buffer.seek(0)
    deserialized = Model.load(buffer)
    X_prop_after = deserialized.forward_prop(X)

    assert model.block_dimensions == deserialized.block_dimensions
    assert np.allclose(X_prop_before, X_prop_after)


def test_adam_optimizer(m3: Model):
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


def test_reshape_layer():
    X = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])
    reshape = Reshape(input_dimensions=(4,), output_dimensions=(2, 2))
    assert np.allclose(
        reshape.forward_prop(input_activations=X), X.reshape(X.shape[0], 2, 2)
    )


def test_reshape_layer_backward_prop():
    reshape = Reshape(input_dimensions=(4,), output_dimensions=(2, 2))
    dZ = np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]]])
    assert reshape._backward_prop(dZ=dZ).shape == (2, 4)


def test_convolution_2d_layer_output_shape():
    layer1 = Convolution2D(input_dimensions=(1, 3, 3), kernel_size=1, n_kernels=1)
    input_activations = np.ones((1, 1, 3, 3))
    output = layer1.forward_prop(input_activations=input_activations)
    assert output.shape == tuple(
        [input_activations.shape[0], *layer1.output_dimensions]
    )

    layer2 = Convolution2D(input_dimensions=(1, 3, 3), kernel_size=2, n_kernels=1)
    input_activations = np.ones((1, 1, 3, 3))
    output = layer2.forward_prop(input_activations=input_activations)
    assert output.shape == tuple(
        [input_activations.shape[0], *layer2.output_dimensions]
    )

    layer3 = Convolution2D(input_dimensions=(1, 3, 3), kernel_size=2, n_kernels=2)
    input_activations = np.ones((1, 1, 3, 3))
    output = layer3.forward_prop(input_activations=input_activations)
    assert output.shape == tuple(
        [input_activations.shape[0], *layer3.output_dimensions]
    )


def test_convolution_2d_layer_forward_prop():
    def _kernel_init_fn(*args: object) -> Convolution2D.Parameters:
        del args  # unused
        return Convolution2D.Parameters(
            weights=np.ones((1, 1, 2, 2)), biases=np.zeros((1, 2, 2)) - 2
        )

    conv_layer = Convolution2D(
        input_dimensions=(1, 3, 3),
        kernel_size=2,
        n_kernels=1,
        kernel_init_fn=_kernel_init_fn,
    )
    activation_layer = Activation(
        input_dimensions=conv_layer.output_dimensions, activation_fn=ReLU
    )
    # [[[[1, 1],
    #    [1, 1]]]]
    input_activations = np.ones((1, 1, 3, 3))
    # [[[[1, 1, 1],
    #    [1, 1, 1],
    #    [1, 1, 1]]]]
    output = activation_layer.forward_prop(
        input_activations=conv_layer.forward_prop(input_activations=input_activations)
    )
    expected_output = np.array(
        [
            [
                [
                    [2.0, 2.0],
                    [2.0, 2.0],
                ]
            ]
        ]
    )
    assert np.allclose(output, expected_output)


def test_convolution_2d_layer_backward_prop():
    layer = Convolution2D(
        input_dimensions=(1, 3, 3),
        kernel_size=2,
        n_kernels=1,
        kernel_init_fn=Convolution2D.Parameters.ones,
    )
    input_activations = np.ones((1, 1, 3, 3))
    output = layer.forward_prop(input_activations=input_activations)
    dZ = np.ones_like(output)
    dX = layer.backward_prop(dZ=dZ)
    assert np.allclose(
        dX, np.array([[[[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]]])
    )


def test_convolution_2d_layer_gradient_operation():
    layer = Convolution2D(input_dimensions=(1, 3, 3), kernel_size=2, n_kernels=1)
    assert isinstance(layer, GradLayer)


@pytest.mark.skip("Will need to figure out how to test this.")
def test_convolution_2d_layer_adam_step():
    layer = Convolution2D(input_dimensions=(1, 3, 3), kernel_size=2, n_kernels=1)
    flatten = Flatten(input_dimensions=layer.output_dimensions)
    model = Model(
        input_dimensions=(1, 3, 3),
        hidden=tuple(
            [
                Hidden(layers=[layer, flatten]),
            ]
        ),
        output=Output(
            layers=[
                Linear(
                    input_dimensions=flatten.output_dimensions,
                    output_dimensions=flatten.output_dimensions,
                    parameters=Linear.Parameters.eye(flatten.output_dimensions),
                ),
            ],
            output_layer=RawOutputLayer(input_dimensions=flatten.output_dimensions),
        ),
    )
    optimizer = AdaM(
        model=model,
        config=AdaM.Config(
            scheduler=ConstantScheduler(learning_rate=0.01),
        ),
    )
    optimizer.training_step(
        X_train_batch=np.ones((1, 1, 3, 3)), Y_train_batch=np.ones((1, 4))
    )
    assert np.allclose(layer.parameters.weights, np.ones((1, 1, 2, 2)) * 0.99)


@pytest.mark.parametrize(
    ("pool_size", "stride", "input_activations", "expected"),
    [
        (2, 1, np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]), [[[[5, 6], [8, 9]]]]),
        (3, 1, np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]), [[[[9]]]]),
        (1, 2, np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]), [[[[1, 3], [7, 9]]]]),
        (2, 2, np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]), [[[[5]]]]),
        (2, 2, np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]), [[[[5]]]]),
        # TODO: Add non-square input test cases
        (
            2,
            2,
            np.array(
                [
                    [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]],
                    [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]],
                ]
            ),
            [[[[5]]], [[[5]]]],
        ),
    ],
)
def test_max_pool_2d_forward_prop(
    pool_size: int, stride: int, input_activations: Activations, expected: Activations
):
    pool_layer = MaxPooling2D(
        input_dimensions=(1, 3, 3), pool_size=pool_size, stride=stride
    )
    output = pool_layer.forward_prop(input_activations=input_activations)
    assert np.allclose(output, expected)


@pytest.mark.parametrize(
    (
        "input_dimensions",
        "pool_size",
        "stride",
        "input_activations",
        "dZ",
        "expected",
    ),
    [
        (
            (1, 2, 2),
            2,
            1,
            np.array([[[[1, 2], [3, 4]]]]),
            np.array([[[[1, 1], [1, 1]]]]),
            np.array([[[[0, 0], [0, 1]]]]),
        ),
        (
            (1, 3, 3),
            2,
            1,
            np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]),
            np.array([[[[1, 1], [1, 1]]]]),
            np.array([[[[0, 0, 0], [0, 1, 1], [0, 1, 1]]]]),
        ),
        (
            (1, 3, 3),
            2,
            1,
            np.array([[[[1, 2, 3], [4, 10, 6], [7, 8, 9]]]]),
            np.array([[[[1, 1], [1, 1]]]]),
            np.array([[[[0, 0, 0], [0, 4, 0], [0, 0, 0]]]]),
            # TODO: Add non-square input test cases
        ),
    ],
)
def test_max_pool_2d_backward_prop(
    input_dimensions: Dimensions,
    pool_size: int,
    stride: int,
    input_activations: Activations,
    dZ: D[Activations],
    expected: D[Activations],
):
    pool_layer = MaxPooling2D(
        input_dimensions=input_dimensions, pool_size=pool_size, stride=stride
    )
    pool_layer.forward_prop(input_activations=input_activations)
    dX = pool_layer.backward_prop(dZ=dZ)
    assert np.allclose(dX, expected)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    (
        "input_dimensions",
        "batch_size",
        "expected_output_shape",
    ),
    [
        (
            (1, 2, 2),
            2,
            (2, 4),
        ),
        (
            (3, 2, 2),
            1,
            (1, 12),
        ),
        (
            (2, 3, 4, 5),
            10,
            (10, 120),
        ),
    ],
)
def test_flatten_layer(
    input_dimensions: Dimensions,
    batch_size: int,
    expected_output_shape: tuple[int, int],
):
    flatten = Flatten(input_dimensions=input_dimensions)
    input_activations = Activations(np.ones((batch_size, *input_dimensions)))
    output = flatten.forward_prop(input_activations=input_activations)
    assert np.allclose(output, np.ones(expected_output_shape))
