from dataclasses import dataclass
from typing import assert_never

from more_itertools import one

from mo_net.model.layer.activation import Activation
from mo_net.model.layer.batch_norm import BatchNorm
from mo_net.model.layer.layer_norm import LayerNorm
from mo_net.model.layer.linear import Linear
from mo_net.model.module.base import Hidden
from mo_net.protos import ActivationFn, Dimensions


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
                    neurons=one(output_dimensions),
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
                        parameters=Linear.Parameters.appropriate(
                            dim_in=input_dimensions,
                            dim_out=output_dimensions,
                            activation_fn=activation_fn,
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
