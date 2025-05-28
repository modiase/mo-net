from dataclasses import dataclass

import numpy as np
import pytest

from mnist_numpy.functions import ReLU
from mnist_numpy.model.layer import Activation
from mnist_numpy.model.layer.convolution import (
    Convolution2D,
    KernelInitFn,
    ParametersType,
)
from mnist_numpy.protos import Activations, Dimensions, GradLayer


@dataclass(frozen=True)
class OutputShapeTestCase:
    name: str
    input_dimensions: Dimensions
    kernel_size: int | tuple[int, int]
    n_kernels: int
    input_activations: np.ndarray
    expected_output_shape: tuple[int, ...]


@pytest.mark.parametrize(
    "test_case",
    [
        OutputShapeTestCase(
            name="1x1_kernel_single_channel",
            input_dimensions=(1, 3, 3),
            kernel_size=1,
            n_kernels=1,
            input_activations=np.ones((1, 1, 3, 3)),
            expected_output_shape=(1, 1, 3, 3),
        ),
        OutputShapeTestCase(
            name="2x2_kernel_single_channel",
            input_dimensions=(1, 3, 3),
            kernel_size=2,
            n_kernels=1,
            input_activations=np.ones((1, 1, 3, 3)),
            expected_output_shape=(1, 1, 2, 2),
        ),
        OutputShapeTestCase(
            name="2x2_kernel_dual_channel",
            input_dimensions=(1, 3, 3),
            kernel_size=2,
            n_kernels=2,
            input_activations=np.ones((1, 1, 3, 3)),
            expected_output_shape=(1, 2, 2, 2),
        ),
        OutputShapeTestCase(
            name="3x2_kernel_rectangular_input",
            input_dimensions=(1, 5, 4),
            kernel_size=(3, 2),  # (height=3, width=2)
            n_kernels=1,
            input_activations=np.ones((1, 1, 5, 4)),
            expected_output_shape=(1, 1, 3, 3),  # (5-3)//1+1=3, (4-2)//1+1=3
        ),
        OutputShapeTestCase(
            name="1x3_kernel_wide_input",
            input_dimensions=(1, 3, 6),
            kernel_size=(1, 3),  # (height=1, width=3)
            n_kernels=2,
            input_activations=np.ones((1, 1, 3, 6)),
            expected_output_shape=(1, 2, 3, 4),  # (3-1)//1+1=3, (6-3)//1+1=4
        ),
        OutputShapeTestCase(
            name="2x1_kernel_tall_input",
            input_dimensions=(1, 6, 3),
            kernel_size=(2, 1),  # (height=2, width=1)
            n_kernels=3,
            input_activations=np.ones((1, 1, 6, 3)),
            expected_output_shape=(1, 3, 5, 3),  # (6-2)//1+1=5, (3-1)//1+1=3
        ),
    ],
    ids=lambda test_case: test_case.name,
)
def test_convolution_2d_output_shape(test_case: OutputShapeTestCase):
    """Test that Convolution2D layer produces correct output shapes."""
    layer = Convolution2D(
        input_dimensions=test_case.input_dimensions,
        kernel_size=test_case.kernel_size,
        n_kernels=test_case.n_kernels,
    )
    output = layer.forward_prop(input_activations=test_case.input_activations)  # type: ignore[arg-type]
    assert output.shape == test_case.expected_output_shape


@dataclass(frozen=True)
class ForwardPropTestCase:
    name: str
    input_dimensions: Dimensions
    kernel_size: int | tuple[int, int]
    n_kernels: int
    kernel_init_fn: KernelInitFn
    input_activations: np.ndarray
    expected_output: np.ndarray
    use_activation: bool = False


def _custom_kernel_init_fn(*args: object) -> ParametersType:
    """Custom kernel initialization for testing."""
    del args  # unused
    return Convolution2D.Parameters(
        weights=np.ones((1, 1, 2, 2)), biases=np.zeros(1) - 2
    )


@pytest.mark.parametrize(
    "test_case",
    [
        ForwardPropTestCase(
            name="2x2_kernel_with_relu_activation",
            input_dimensions=(1, 3, 3),
            kernel_size=2,
            n_kernels=1,
            kernel_init_fn=_custom_kernel_init_fn,
            input_activations=np.ones((1, 1, 3, 3)),
            expected_output=np.array([[[[2.0, 2.0], [2.0, 2.0]]]]),
            use_activation=True,
        ),
        ForwardPropTestCase(
            name="2x2_kernel_rectangular_4x3_input",
            input_dimensions=(1, 4, 3),
            kernel_size=2,
            n_kernels=1,
            kernel_init_fn=_custom_kernel_init_fn,
            input_activations=np.ones((1, 1, 4, 3)),
            expected_output=np.array([[[[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]]]]),
            use_activation=True,
        ),
        ForwardPropTestCase(
            name="3x2_kernel_rectangular_5x4_input",
            input_dimensions=(1, 5, 4),
            kernel_size=(3, 2),  # (height=3, width=2)
            n_kernels=1,
            kernel_init_fn=lambda *args: Convolution2D.Parameters(
                weights=np.ones((1, 1, 3, 2)),
                biases=np.zeros(1) - 6,  # (n_kernels, in_channels, height, width)
            ),
            input_activations=np.ones((1, 1, 5, 4)),
            expected_output=np.array(
                [[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]]
            ),  # Shape: (1, 1, 3, 3)
            use_activation=True,
        ),
        ForwardPropTestCase(
            name="1x3_kernel_wide_input",
            input_dimensions=(1, 3, 6),
            kernel_size=(1, 3),  # (height=1, width=3)
            n_kernels=1,
            kernel_init_fn=lambda *args: Convolution2D.Parameters(
                weights=np.ones((1, 1, 1, 3)),
                biases=np.zeros(1) - 3,  # (n_kernels, in_channels, height, width)
            ),
            input_activations=np.ones((1, 1, 3, 6)),
            expected_output=np.array(
                [[[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]]
            ),  # Shape: (1, 1, 3, 4)
            use_activation=True,
        ),
        ForwardPropTestCase(
            name="2x1_kernel_tall_input",
            input_dimensions=(1, 6, 3),
            kernel_size=(2, 1),  # (height=2, width=1)
            n_kernels=1,
            kernel_init_fn=lambda *args: Convolution2D.Parameters(
                weights=np.ones((1, 1, 2, 1)),
                biases=np.zeros(1) - 2,  # (n_kernels, in_channels, height, width)
            ),
            input_activations=np.ones((1, 1, 6, 3)),
            expected_output=np.array(
                [
                    [
                        [
                            [0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0],
                        ]
                    ]
                ]
            ),  # Shape: (1, 1, 5, 3)
            use_activation=True,
        ),
    ],
    ids=lambda test_case: test_case.name,
)
def test_convolution_2d_forward_prop(test_case: ForwardPropTestCase):
    """Test forward propagation through Convolution2D layer."""
    conv_layer = Convolution2D(
        input_dimensions=test_case.input_dimensions,
        kernel_size=test_case.kernel_size,
        n_kernels=test_case.n_kernels,
        kernel_init_fn=test_case.kernel_init_fn,
    )

    if test_case.use_activation:
        activation_layer = Activation(
            input_dimensions=conv_layer.output_dimensions, activation_fn=ReLU
        )
        output = activation_layer.forward_prop(
            input_activations=conv_layer.forward_prop(
                input_activations=Activations(test_case.input_activations)
            )
        )
    else:
        output = conv_layer.forward_prop(
            input_activations=Activations(test_case.input_activations)
        )

    assert np.allclose(output, test_case.expected_output)


@dataclass(frozen=True)
class BackwardPropTestCase:
    name: str
    input_dimensions: Dimensions
    kernel_size: int | tuple[int, int]
    n_kernels: int
    kernel_init_fn: KernelInitFn
    input_activations: np.ndarray
    expected_dX: np.ndarray


@pytest.mark.parametrize(
    "test_case",
    [
        BackwardPropTestCase(
            name="2x2_kernel_ones_initialization",
            input_dimensions=(1, 3, 3),
            kernel_size=2,
            n_kernels=1,
            kernel_init_fn=Convolution2D.Parameters.ones,
            input_activations=np.ones((1, 1, 3, 3)),
            expected_dX=np.array(
                [[[[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]]]
            ),
        ),
        BackwardPropTestCase(
            name="3x2_kernel_rectangular_non_square",
            input_dimensions=(1, 4, 5),
            kernel_size=(3, 2),  # (height=3, width=2)
            n_kernels=1,
            kernel_init_fn=Convolution2D.Parameters.ones,
            input_activations=np.ones((1, 1, 4, 5)),
            expected_dX=np.array(
                [
                    [
                        [
                            [1.0, 2.0, 2.0, 2.0, 1.0],
                            [2.0, 4.0, 4.0, 4.0, 2.0],
                            [2.0, 4.0, 4.0, 4.0, 2.0],
                            [1.0, 2.0, 2.0, 2.0, 1.0],
                        ]
                    ]
                ]
            ),
        ),
        BackwardPropTestCase(
            name="1x3_kernel_wide_kernel",
            input_dimensions=(1, 3, 5),
            kernel_size=(1, 3),  # (height=1, width=3)
            n_kernels=1,
            kernel_init_fn=Convolution2D.Parameters.ones,
            input_activations=np.ones((1, 1, 3, 5)),
            expected_dX=np.array(
                [
                    [
                        [
                            [1.0, 2.0, 3.0, 2.0, 1.0],
                            [1.0, 2.0, 3.0, 2.0, 1.0],
                            [1.0, 2.0, 3.0, 2.0, 1.0],
                        ]
                    ]
                ]
            ),
        ),
        BackwardPropTestCase(
            name="2x1_kernel_tall_kernel",
            input_dimensions=(1, 5, 3),
            kernel_size=(2, 1),  # (height=2, width=1)
            n_kernels=1,
            kernel_init_fn=Convolution2D.Parameters.ones,
            input_activations=np.ones((1, 1, 5, 3)),
            expected_dX=np.array(
                [
                    [
                        [
                            [1.0, 1.0, 1.0],
                            [2.0, 2.0, 2.0],
                            [2.0, 2.0, 2.0],
                            [2.0, 2.0, 2.0],
                            [1.0, 1.0, 1.0],
                        ]
                    ]
                ]
            ),
        ),
        BackwardPropTestCase(
            name="2x2_kernel_multiple_kernels",
            input_dimensions=(1, 3, 3),
            kernel_size=2,
            n_kernels=2,
            kernel_init_fn=Convolution2D.Parameters.ones,
            input_activations=np.ones((1, 1, 3, 3)),
            expected_dX=np.array(
                [[[[2.0, 4.0, 2.0], [4.0, 8.0, 4.0], [2.0, 4.0, 2.0]]]]
            ),
        ),
        BackwardPropTestCase(
            name="1x1_kernel_identity",
            input_dimensions=(1, 3, 3),
            kernel_size=1,
            n_kernels=1,
            kernel_init_fn=Convolution2D.Parameters.ones,
            input_activations=np.ones((1, 1, 3, 3)),
            expected_dX=np.array(
                [[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]]
            ),
        ),
        BackwardPropTestCase(
            name="3x1_kernel_very_tall",
            input_dimensions=(1, 5, 2),
            kernel_size=(3, 1),  # (height=3, width=1)
            n_kernels=1,
            kernel_init_fn=Convolution2D.Parameters.ones,
            input_activations=np.ones((1, 1, 5, 2)),
            expected_dX=np.array(
                [[[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 2.0], [1.0, 1.0]]]]
            ),
        ),
    ],
    ids=lambda test_case: test_case.name,
)
def test_convolution_2d_backward_prop(test_case: BackwardPropTestCase):
    """Test backward propagation through Convolution2D layer."""
    layer = Convolution2D(
        input_dimensions=test_case.input_dimensions,
        kernel_size=test_case.kernel_size,
        n_kernels=test_case.n_kernels,
        kernel_init_fn=test_case.kernel_init_fn,
    )
    output = layer.forward_prop(
        input_activations=Activations(test_case.input_activations)
    )
    dZ = np.ones_like(output)
    dX = layer.backward_prop(dZ=dZ)
    assert np.allclose(dX, test_case.expected_dX)  # type: ignore[arg-type]


def test_convolution_2d_gradient_operation():
    """Test that Convolution2D layer implements GradLayer interface."""
    layer = Convolution2D(input_dimensions=(1, 3, 3), kernel_size=2, n_kernels=1)
    assert isinstance(layer, GradLayer)
