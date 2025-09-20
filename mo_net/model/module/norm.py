from dataclasses import dataclass
from typing import assert_never

import jax

from mo_net.functions import ActivationFn
from mo_net.model.layer.activation import Activation
from mo_net.model.layer.batch_norm import BatchNorm
from mo_net.model.layer.layer_norm import LayerNorm
from mo_net.model.layer.linear import Linear
from mo_net.model.module.base import Hidden
from mo_net.protos import Dimensions


@dataclass(kw_only=True, frozen=True)
class BatchNormOptions:
    momentum: float


@dataclass(kw_only=True, frozen=True)
class LayerNormOptions:
    pass


class Norm(Hidden):
    def __init__(
        self,
        input_dimensions: Dimensions,
        output_dimensions: Dimensions,
        *,
        activation_fn: ActivationFn,
        options: BatchNormOptions | LayerNormOptions,
        store_output_activations: bool,
        key: jax.Array,
    ):
        norm_layer: BatchNorm | LayerNorm
        match options:
            case BatchNormOptions():
                norm_layer = BatchNorm(
                    input_dimensions=output_dimensions,
                    momentum=options.momentum,
                    store_output_activations=store_output_activations,
                )
            case LayerNormOptions():
                norm_layer = LayerNorm(
                    input_dimensions=output_dimensions,
                    store_output_activations=store_output_activations,
                )
            case never:
                assert_never(never)

        super().__init__(
            layers=tuple(
                [
                    Linear(
                        output_dimensions=output_dimensions,
                        input_dimensions=input_dimensions,
                        parameters_init_fn=lambda dim_in,
                        dim_out: Linear.Parameters.appropriate(
                            dim_in=dim_in,
                            dim_out=dim_out,
                            activation_fn=activation_fn,
                            key=jax.random.split(key)[0],
                        ),
                        store_output_activations=store_output_activations,
                        clip_gradients=True,
                        weight_max_norm=1.0,
                        bias_max_norm=0.5,
                    ),
                    norm_layer,
                    Activation(
                        input_dimensions=output_dimensions,
                        activation_fn=activation_fn,
                    ),
                ]
            )
        )
