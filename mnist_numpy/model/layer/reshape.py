from mnist_numpy.model.layer.base import Hidden
from mnist_numpy.protos import Activations, D


class Reshape(Hidden):
    def _forward_prop(self, *, input_activations: Activations) -> Activations:
        return input_activations.reshape(
            input_activations.shape[0], *self.output_dimensions
        )

    def _backward_prop(self, *, dZ: D[Activations]) -> D[Activations]:
        return dZ.reshape(dZ.shape[0], *self.input_dimensions)
