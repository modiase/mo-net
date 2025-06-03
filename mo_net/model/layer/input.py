from mo_net.model.layer.base import _Base
from mo_net.protos import Activations, Dimensions


class Input(_Base):
    def __init__(
        self,
        *,
        input_dimensions: Dimensions,
    ):
        self._input_dimensions = input_dimensions

    def _forward_prop(self, *, input_activations: Activations) -> Activations:
        return input_activations

    @property
    def input_dimensions(self) -> Dimensions:
        return self._input_dimensions

    @property
    def output_dimensions(self) -> Dimensions:
        return self._input_dimensions
