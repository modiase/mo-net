from mnist_numpy.model.layer.activation import Activation
from mnist_numpy.model.layer.base import (
    Input,
    Hidden,
)
from mnist_numpy.model.layer.linear import Linear
from mnist_numpy.model.layer.output import _OutputLayer

type NonInputLayer = Hidden | _OutputLayer

__all__ = [
    "Activation",
    "Linear",
    "Input",
    "NonInputLayer",
]
