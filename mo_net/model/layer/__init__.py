from mo_net.model.layer.activation import Activation
from mo_net.model.layer.base import Hidden
from mo_net.model.layer.input import Input
from mo_net.model.layer.linear import Linear
from mo_net.model.layer.output import OutputLayer

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
