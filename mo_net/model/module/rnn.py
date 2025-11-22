import jax

from mo_net.functions import ActivationFn, identity
from mo_net.model.layer.recurrent import Recurrent
from mo_net.model.module.base import Hidden
from mo_net.protos import Dimensions


class RNN(Hidden):
    """
    RNN module that wraps a Recurrent layer.

    This provides a convenient interface for creating recurrent neural networks,
    similar to how Dense wraps Linear + Activation.
    """

    def __init__(
        self,
        input_dimensions: Dimensions,
        hidden_dimensions: Dimensions,
        *,
        activation_fn: ActivationFn = identity,
        key: jax.Array,
        return_sequences: bool = True,
        stateful: bool = False,
        store_output_activations: bool = False,
    ):
        super().__init__(
            layers=tuple(
                [
                    Recurrent(
                        input_dimensions=input_dimensions,
                        hidden_dimensions=hidden_dimensions,
                        activation_fn=activation_fn,
                        return_sequences=return_sequences,
                        stateful=stateful,
                        parameters_init_fn=lambda *_: Recurrent.Parameters.appropriate(
                            dim_in=input_dimensions,
                            dim_hidden=hidden_dimensions,
                            activation_fn=activation_fn,
                            key=key,
                        ),
                        store_output_activations=store_output_activations,
                    ),
                ]
            )
        )
