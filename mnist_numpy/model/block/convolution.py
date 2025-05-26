from mnist_numpy.functions import ReLU
from mnist_numpy.model.block.base import Hidden
from mnist_numpy.model.layer.activation import Activation
from mnist_numpy.model.layer.batch_norm import BatchNorm
from mnist_numpy.model.layer.convolution import Convolution2D
from mnist_numpy.model.layer.pool import MaxPooling2D
from mnist_numpy.model.layer.reshape import Flatten
from mnist_numpy.protos import ActivationFn, Dimensions


class Convolution(Hidden):
    def __init__(
        self,
        *,
        input_dimensions: Dimensions,
        n_kernels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        pool_size: int | tuple[int, int] = 2,
        pool_stride: int | tuple[int, int] = 1,
        activation_fn: ActivationFn = ReLU,
        flatten_output: bool = False,
    ):
        super().__init__(
            layers=(
                conv_layer := Convolution2D(
                    input_dimensions=input_dimensions,
                    n_kernels=n_kernels,
                    kernel_size=kernel_size,
                    stride=stride,
                ),
                batch_norm_layer := BatchNorm(
                    input_dimensions=conv_layer.output_dimensions,
                ),
                Activation(
                    input_dimensions=batch_norm_layer.output_dimensions,
                    activation_fn=activation_fn,
                ),
                pool_layer := MaxPooling2D(
                    input_dimensions=batch_norm_layer.output_dimensions,
                    pool_size=pool_size,
                    stride=pool_stride,
                ),
            )
        )
        if flatten_output:
            self.append_layer(Flatten(input_dimensions=pool_layer.output_dimensions))
