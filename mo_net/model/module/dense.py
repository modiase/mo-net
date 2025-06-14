from mo_net.model.layer.activation import Activation
from mo_net.model.layer.linear import Linear
from mo_net.model.module.base import Hidden
from mo_net.protos import ActivationFn, Dimensions


class Dense(Hidden):
    def __init__(
        self,
        input_dimensions: Dimensions,
        output_dimensions: Dimensions,
        *,
        activation_fn: ActivationFn,
        store_output_activations: bool = False,
    ):
        super().__init__(
            layers=tuple(
                [
                    Linear(
                        output_dimensions=output_dimensions,
                        input_dimensions=input_dimensions,
                        parameters=Linear.Parameters.appropriate(
                            dim_in=input_dimensions,
                            dim_out=output_dimensions,
                            activation_fn=activation_fn,
                        ),
                        store_output_activations=store_output_activations,
                    ),
                    Activation(
                        input_dimensions=output_dimensions,
                        activation_fn=activation_fn,
                    ),
                ]
            )
        )
