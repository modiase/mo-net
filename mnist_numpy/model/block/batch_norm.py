from mnist_numpy.model.block.base import Hidden
from mnist_numpy.model.layer.activation import Activation
from mnist_numpy.model.layer.batch_norm import BatchNorm as BatchNormLayer
from mnist_numpy.model.layer.dense import Dense
from mnist_numpy.types import ActivationFn


class BatchNorm(Hidden):
    def __init__(
        self,
        *,
        activation_fn: ActivationFn,
        batch_size: int,
        input_dimensions: int,
        momentum: float = 0.9,
        output_dimensions: int,
        store_output_activations: bool = False,
    ):
        super().__init__(
            layers=tuple(
                [
                    Dense(
                        output_dimensions=output_dimensions,
                        input_dimensions=input_dimensions,
                        parameters=Dense.Parameters.appropriate(
                            dim_in=input_dimensions,
                            dim_out=output_dimensions,
                            activation_fn=activation_fn,
                        ),
                        store_output_activations=store_output_activations,
                    ),
                    BatchNormLayer(
                        input_dimensions=output_dimensions,
                        output_dimensions=output_dimensions,
                        momentum=momentum,
                        batch_size=batch_size,
                        store_output_activations=store_output_activations,
                    ),
                    Activation(
                        input_dimensions=output_dimensions,
                        activation_fn=activation_fn,
                    ),
                ]
            )
        )
