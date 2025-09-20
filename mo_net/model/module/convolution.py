import functools

import jax.nn

from mo_net.functions import ActivationFn, ReLU
from mo_net.model.layer.activation import Activation
from mo_net.model.layer.batch_norm.batch_norm_2d import BatchNorm2D
from mo_net.model.layer.convolution import Convolution2D
from mo_net.model.layer.pool import MaxPooling2D
from mo_net.model.layer.reshape import Flatten
from mo_net.model.module.base import Hidden
from mo_net.protos import Dimensions


class Convolution(Hidden):
    def __init__(
        self,
        *,
        activation_fn: ActivationFn = ReLU(),
        flatten_output: bool = False,
        input_dimensions: Dimensions,
        kernel_size: int | tuple[int, int],
        key: jax.Array,
        n_kernels: int,
        pool_size: int | tuple[int, int] = 2,
        pool_stride: int | tuple[int, int] = 1,
        stride: int | tuple[int, int] = 1,
    ):
        key = jax.random.split(key)[0]
        super().__init__(
            layers=(
                conv_layer := Convolution2D(
                    input_dimensions=input_dimensions,
                    n_kernels=n_kernels,
                    kernel_size=kernel_size,
                    stride=stride,
                    kernel_init_fn=functools.partial(
                        Convolution2D.Parameters.he, key=key
                    ),
                ),
                batch_norm_layer := BatchNorm2D(
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
