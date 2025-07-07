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
