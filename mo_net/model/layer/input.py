import jax.numpy as jnp

from mo_net.model.layer.base import _Base
from mo_net.protos import Activations, D, Dimensions


class Input(_Base):
    def __init__(
        self,
        *,
        input_dimensions: Dimensions,
    ):
        super().__init__(
            input_dimensions=input_dimensions,
            output_dimensions=input_dimensions,
        )
        self._input_dimensions = input_dimensions

    def forward_prop(self, input_activations: Activations) -> Activations:
        input_activations = Activations(jnp.atleast_2d(input_activations))

        if input_activations.ndim not in (2, 3):
            raise ValueError(
                f"Input layer expects 2D or 3D input, got shape {input_activations.shape}"
            )

        feature_dims = input_activations.shape[input_activations.ndim - 1 :]
        if feature_dims != self.input_dimensions:
            raise ValueError(
                f"Input activations feature dimensions {feature_dims} do not match "
                f"input dimensions {self.input_dimensions} in layer {self}."
            )

        return self._forward_prop(input_activations=input_activations)

    def _forward_prop(self, *, input_activations: Activations) -> Activations:
        return input_activations

    def _backward_prop(self, *, dZ: D[Activations]) -> D[Activations]:
        return dZ

    def backward_prop(self, dZ: D[Activations]) -> D[Activations]:
        return self._backward_prop(dZ=dZ)

    @property
    def input_dimensions(self) -> Dimensions:
        return self._input_dimensions

    @property
    def output_dimensions(self) -> Dimensions:
        return self._input_dimensions
