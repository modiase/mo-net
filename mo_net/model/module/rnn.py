import jax

from mo_net.functions import ActivationFn, Tanh
from mo_net.model.layer.recurrent import Recurrent
from mo_net.model.module.base import Hidden
from mo_net.protos import Dimensions


class RNN(Hidden):
    def __init__(
        self,
        input_dimensions: Dimensions,
        hidden_dim: int,
        *,
        activation_fn: ActivationFn = Tanh(),
        key: jax.Array,
        return_sequences: bool = True,
        store_output_activations: bool = False,
    ):
        super().__init__(
            layers=(
                Recurrent(
                    input_dimensions=input_dimensions,
                    hidden_dim=hidden_dim,
                    activation_fn=activation_fn,
                    parameters_init_fn=Recurrent.Parameters.xavier,
                    return_sequences=return_sequences,
                    store_output_activations=store_output_activations,
                    key=key,
                ),
            )
        )
