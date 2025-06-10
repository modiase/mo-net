from mo_net.functions import ReLU
from mo_net.model.layer.activation import Activation
from mo_net.model.layer.batch_norm import BatchNorm
from mo_net.model.layer.convolution import Convolution2D
from mo_net.model.layer.pool import MaxPooling2D
from mo_net.model.layer.reshape import Flatten
from mo_net.model.module.base import Hidden
from mo_net.protos import ActivationFn, Dimensions


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
