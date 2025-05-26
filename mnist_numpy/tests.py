import numpy as np
import pytest

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
from mnist_numpy.model.layer.reshape import Flatten
from mnist_numpy.optimizer.adam import AdaM
from mnist_numpy.optimizer.base import Null
from mnist_numpy.optimizer.scheduler import ConstantScheduler
from mnist_numpy.protos import GradLayer


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
