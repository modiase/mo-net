from collections.abc import Callable

from mnist_numpy.functions import ActivationFn
from mnist_numpy.model.layer.activation import Activation
from mnist_numpy.model.layer.base import _Hidden
from mnist_numpy.model.layer.linear import Activations, Linear
from mnist_numpy.model.layer.linear import Parameters as LinearParameters
from mnist_numpy.model.layer.linear import ParametersType as LinearParametersType
from mnist_numpy.types import D


class Dense(_Hidden):
    def __init__(
        self,
        *,
        input_dimensions: int,
        output_dimensions: int,
        activation_fn: ActivationFn,
        parameters: LinearParametersType | None = None,
        parameters_init_fn: Callable[
            [int, int], LinearParametersType
        ] = LinearParameters.xavier,
        store_output_activations: bool = False,
    ):
        super().__init__(
            input_dimensions=input_dimensions,
            output_dimensions=output_dimensions,
        )
        self._activation_layer = Activation(
            activation_fn=activation_fn,
            input_dimensions=input_dimensions,
        )
        self._linear_layer = Linear(
            input_dimensions=input_dimensions,
            output_dimensions=output_dimensions,
            parameters=parameters,
            parameters_init_fn=parameters_init_fn,
            store_output_activations=store_output_activations,
        )

    def _forward_prop(self, *, input_activations: Activations) -> Activations:
        return self._activation_layer.forward_prop(
            input_activations=self._linear_layer.forward_prop(
                input_activations=input_activations
            )
        )

    def _backward_prop(self, *, dZ: D[Activations]) -> D[Activations]:
        return self._linear_layer.backward_prop(
            dZ=self._activation_layer.backward_prop(dZ=dZ)
        )
