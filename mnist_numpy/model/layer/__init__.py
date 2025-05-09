from mnist_numpy.model.layer.activation import Activation
from mnist_numpy.model.layer.base import Hidden
from mnist_numpy.model.layer.input import Input
from mnist_numpy.model.layer.linear import Linear
from mnist_numpy.model.layer.output import OutputLayer

type NonInputLayer = Hidden | OutputLayer
type InputLayer = Input

__all__ = [
    "Activation",
    "Linear",
    "Input",
    "NonInputLayer",
    "InputLayer",
    "OutputLayer",
]
