from dataclasses import dataclass

import jax.numpy as jnp
import pytest

from mo_net.functions import get_activation_fn
from mo_net.model.layer import Activation
from mo_net.model.layer.convolution import (
    Convolution2D,
    KernelInitFn,
    ParametersType,
)
from mo_net.protos import Activations, Dimensions, GradLayer


@dataclass(frozen=True)
class OutputShapeTestCase:
    name: str
    input_dimensions: Dimensions
    kernel_size: int | tuple[int, int]
    n_kernels: int
    input_activations: jnp.ndarray
    expected_output_shape: tuple[int, ...]


@pytest.mark.parametrize(
    "test_case",
    [
        OutputShapeTestCase(
            name="1x1_kernel_single_channel",
            input_dimensions=(1, 3, 3),
            kernel_size=1,
            n_kernels=1,
            input_activations=jnp.ones((1, 1, 3, 3)),
            expected_output_shape=(1, 1, 3, 3),
        ),
        OutputShapeTestCase(
            name="2x2_kernel_single_channel",
            input_dimensions=(1, 3, 3),
            kernel_size=2,
            n_kernels=1,
            input_activations=jnp.ones((1, 1, 3, 3)),
            expected_output_shape=(1, 1, 2, 2),
        ),
        OutputShapeTestCase(
            name="2x2_kernel_dual_channel",
            input_dimensions=(1, 3, 3),
            kernel_size=2,
            n_kernels=2,
            input_activations=jnp.ones((1, 1, 3, 3)),
            expected_output_shape=(1, 2, 2, 2),
        ),
        OutputShapeTestCase(
            name="3x2_kernel_rectangular_input",
            input_dimensions=(1, 5, 4),
            kernel_size=(3, 2),  # (height=3, width=2)
            n_kernels=1,
            input_activations=jnp.ones((1, 1, 5, 4)),
            expected_output_shape=(1, 1, 3, 3),  # (5-3)//1+1=3, (4-2)//1+1=3
        ),
        OutputShapeTestCase(
            name="1x3_kernel_wide_input",
            input_dimensions=(1, 3, 6),
            kernel_size=(1, 3),  # (height=1, width=3)
            n_kernels=2,
            input_activations=jnp.ones((1, 1, 3, 6)),
            expected_output_shape=(1, 2, 3, 4),  # (3-1)//1+1=3, (6-3)//1+1=4
        ),
        OutputShapeTestCase(
            name="2x1_kernel_tall_input",
            input_dimensions=(1, 6, 3),
            kernel_size=(2, 1),  # (height=2, width=1)
            n_kernels=3,
            input_activations=jnp.ones((1, 1, 6, 3)),
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
        kernel_init_fn=Convolution2D.Parameters.ones,
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
    input_activations: jnp.ndarray
    expected_output: jnp.ndarray
    use_activation: bool = False


def _custom_kernel_init_fn(*args: object) -> ParametersType:
    """Custom kernel initialization for testing."""
    del args  # unused
    return Convolution2D.Parameters(
        weights=jnp.ones((1, 1, 2, 2)), biases=jnp.zeros(1) - 2
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
            input_activations=jnp.ones((1, 1, 3, 3)),
            expected_output=jnp.array([[[[2.0, 2.0], [2.0, 2.0]]]]),
            use_activation=True,
        ),
        ForwardPropTestCase(
            name="2x2_kernel_rectangular_4x3_input",
            input_dimensions=(1, 4, 3),
            kernel_size=2,
            n_kernels=1,
            kernel_init_fn=_custom_kernel_init_fn,
            input_activations=jnp.ones((1, 1, 4, 3)),
            expected_output=jnp.array([[[[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]]]]),
            use_activation=True,
        ),
        ForwardPropTestCase(
            name="3x2_kernel_rectangular_5x4_input",
            input_dimensions=(1, 5, 4),
            kernel_size=(3, 2),  # (height=3, width=2)
            n_kernels=1,
            kernel_init_fn=lambda *args: Convolution2D.Parameters(
                weights=jnp.ones((1, 1, 3, 2)),
                biases=jnp.zeros(1) - 6,  # (n_kernels, in_channels, height, width)
            ),
            input_activations=jnp.ones((1, 1, 5, 4)),
            expected_output=jnp.array(
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
                weights=jnp.ones((1, 1, 1, 3)),
                biases=jnp.zeros(1) - 3,  # (n_kernels, in_channels, height, width)
            ),
            input_activations=jnp.ones((1, 1, 3, 6)),
            expected_output=jnp.array(
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
                weights=jnp.ones((1, 1, 2, 1)),
                biases=jnp.zeros(1) - 2,  # (n_kernels, in_channels, height, width)
            ),
            input_activations=jnp.ones((1, 1, 6, 3)),
            expected_output=jnp.array(
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
            input_dimensions=conv_layer.output_dimensions,
            activation_fn=get_activation_fn("relu"),
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

    assert jnp.allclose(output, test_case.expected_output)


@dataclass(frozen=True)
class BackwardPropTestCase:
    name: str
    input_dimensions: Dimensions
    kernel_size: int | tuple[int, int]
    n_kernels: int
    kernel_init_fn: KernelInitFn
    input_activations: jnp.ndarray
    expected_dX: jnp.ndarray


@pytest.mark.parametrize(
    "test_case",
    [
        BackwardPropTestCase(
            name="2x2_kernel_ones_initialization",
            input_dimensions=(1, 3, 3),
            kernel_size=2,
            n_kernels=1,
            kernel_init_fn=Convolution2D.Parameters.ones,
            input_activations=jnp.ones((1, 1, 3, 3)),
            expected_dX=jnp.array(
                [[[[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]]]
            ),
        ),
        BackwardPropTestCase(
            name="3x2_kernel_rectangular_non_square",
            input_dimensions=(1, 4, 5),
            kernel_size=(3, 2),  # (height=3, width=2)
            n_kernels=1,
            kernel_init_fn=Convolution2D.Parameters.ones,
            input_activations=jnp.ones((1, 1, 4, 5)),
            expected_dX=jnp.array(
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
            input_activations=jnp.ones((1, 1, 3, 5)),
            expected_dX=jnp.array(
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
            input_activations=jnp.ones((1, 1, 5, 3)),
            expected_dX=jnp.array(
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
            input_activations=jnp.ones((1, 1, 3, 3)),
            expected_dX=jnp.array(
                [[[[2.0, 4.0, 2.0], [4.0, 8.0, 4.0], [2.0, 4.0, 2.0]]]]
            ),
        ),
        BackwardPropTestCase(
            name="1x1_kernel_identity",
            input_dimensions=(1, 3, 3),
            kernel_size=1,
            n_kernels=1,
            kernel_init_fn=Convolution2D.Parameters.ones,
            input_activations=jnp.ones((1, 1, 3, 3)),
            expected_dX=jnp.array(
                [[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]]
            ),
        ),
        BackwardPropTestCase(
            name="3x1_kernel_very_tall",
            input_dimensions=(1, 5, 2),
            kernel_size=(3, 1),  # (height=3, width=1)
            n_kernels=1,
            kernel_init_fn=Convolution2D.Parameters.ones,
            input_activations=jnp.ones((1, 1, 5, 2)),
            expected_dX=jnp.array(
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
    dZ = jnp.ones_like(output)
    dX = layer.backward_prop(dZ=dZ)
    assert jnp.allclose(dX, test_case.expected_dX)  # type: ignore[arg-type]


def test_convolution_2d_gradient_operation():
    """Test that Convolution2D layer implements GradLayer interface."""
    layer = Convolution2D(
        input_dimensions=(1, 3, 3),
        kernel_size=2,
        n_kernels=1,
        kernel_init_fn=Convolution2D.Parameters.ones,
    )
    assert isinstance(layer, GradLayer)


@dataclass(frozen=True)
class ParameterUpdateTestCase:
    name: str
    input_dimensions: Dimensions
    kernel_size: int | tuple[int, int]
    n_kernels: int
    kernel_init_fn: KernelInitFn
    input_activations: jnp.ndarray
    dZ: jnp.ndarray
    expected_output: jnp.ndarray
    expected_weight_gradients: jnp.ndarray
    expected_bias_gradients: jnp.ndarray


@pytest.mark.parametrize(
    "test_case",
    [
        ParameterUpdateTestCase(
            name="2x2_kernel_simple_case",
            input_dimensions=(1, 2, 2),
            kernel_size=2,
            n_kernels=1,
            kernel_init_fn=lambda *_: Convolution2D.Parameters(
                weights=jnp.array([[[[0.1, 0.2], [0.3, 0.4]]]]), biases=jnp.array([0.5])
            ),
            input_activations=jnp.array([[[[1.0, 2.0], [3.0, 4.0]]]]),
            dZ=jnp.array([[[[1.0]]]]),
            expected_output=jnp.array([[[[3.5]]]]),
            expected_weight_gradients=jnp.array([[[[1.0, 2.0], [3.0, 4.0]]]]),
            expected_bias_gradients=jnp.array([1.0]),
        ),
        ParameterUpdateTestCase(
            name="3x3_input_2x2_kernel",
            input_dimensions=(1, 3, 3),
            kernel_size=2,
            n_kernels=1,
            kernel_init_fn=lambda *_: Convolution2D.Parameters(
                weights=jnp.array([[[[1.0, 0.0], [0.0, 1.0]]]]), biases=jnp.array([0.0])
            ),
            input_activations=jnp.array(
                [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]]
            ),
            dZ=jnp.array([[[[1.0, 1.0], [1.0, 1.0]]]]),
            expected_output=jnp.array([[[[6.0, 8.0], [12.0, 14.0]]]]),
            expected_weight_gradients=jnp.array([[[[12.0, 16.0], [24.0, 28.0]]]]),
            expected_bias_gradients=jnp.array([4.0]),
        ),
        ParameterUpdateTestCase(
            name="3x2_input_2x1_kernel",
            input_dimensions=(1, 3, 2),
            kernel_size=(2, 1),
            n_kernels=1,
            kernel_init_fn=lambda *_: Convolution2D.Parameters(
                weights=jnp.array([[[[1.0], [0.5]]]]), biases=jnp.array([0.0])
            ),
            input_activations=jnp.array([[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]]),
            dZ=jnp.array([[[[1.0, 1.0], [1.0, 1.0]]]]),
            expected_output=jnp.array([[[[2.5, 4.0], [5.5, 7.0]]]]),
            expected_weight_gradients=jnp.array([[[[10.0], [18.0]]]]),
            expected_bias_gradients=jnp.array([4.0]),
        ),
    ],
    ids=lambda test_case: test_case.name,
)
def test_parameter_update_value(test_case: ParameterUpdateTestCase):
    """Test that parameter gradients are correctly computed and stored in cache."""
    layer = Convolution2D(
        input_dimensions=test_case.input_dimensions,
        kernel_size=test_case.kernel_size,
        n_kernels=test_case.n_kernels,
        kernel_init_fn=test_case.kernel_init_fn,
    )

    output = layer.forward_prop(
        input_activations=Activations(test_case.input_activations)
    )
    assert jnp.allclose(output, test_case.expected_output), (
        f"Forward prop failed: got {output}, expected {test_case.expected_output}"
    )

    layer.backward_prop(dZ=test_case.dZ)

    cached_dP = layer.cache["dP"]
    assert cached_dP is not None, "Parameter gradients not stored in cache"
    assert jnp.allclose(cached_dP.weights, test_case.expected_weight_gradients), (  # type: ignore[attr-defined]
        f"Weight gradients incorrect: got {cached_dP.weights}"  # type: ignore[attr-defined]
    )
    assert jnp.allclose(cached_dP.biases, test_case.expected_bias_gradients), (  # type: ignore[attr-defined]
        f"Bias gradients incorrect: got {cached_dP.biases}"  # type: ignore[attr-defined]
    )
