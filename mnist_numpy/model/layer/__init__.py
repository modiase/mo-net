from mnist_numpy.model.layer.activation import Activation
from mnist_numpy.model.layer.base import (
    Input,
    _Hidden,
)
from mnist_numpy.model.layer.dense import Linear
from mnist_numpy.model.layer.output import _OutputLayer

type NonInputLayer = _Hidden | _OutputLayer

__all__ = [
    "Activation",
    "Linear",
    "Input",
    "NonInputLayer",
]
