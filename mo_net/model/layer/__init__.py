from mo_net.model.layer.activation import Activation
from mo_net.model.layer.average import Average
from mo_net.model.layer.base import Hidden
from mo_net.model.layer.input import Input
from mo_net.model.layer.linear import Linear
from mo_net.model.layer.output import OutputLayer
from mo_net.model.layer.recurrent import Recurrent

type NonInputLayer = Hidden | OutputLayer
type InputLayer = Input

__all__ = [
    "Activation",
    "Average",
    "Linear",
    "Input",
    "NonInputLayer",
    "InputLayer",
    "OutputLayer",
    "Recurrent",
]
