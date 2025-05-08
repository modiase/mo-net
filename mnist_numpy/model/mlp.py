from __future__ import annotations

import pickle
from collections.abc import Callable, MutableSequence
from dataclasses import dataclass
from functools import reduce
from operator import itemgetter
from typing import (
    IO,
    Iterator,
    Self,
    Sequence,
    Protocol,
    runtime_checkable,
)

import numpy as np
from more_itertools import last, pairwise

from mnist_numpy.functions import (
    cross_entropy,
    Identity,
)
from itertools import chain
from mnist_numpy.model import ModelBase
from mnist_numpy.model.block.base import Hidden as HiddenBlock, Output
from mnist_numpy.model.layer.base import _Hidden as HiddenLayer
from mnist_numpy.model.block.batch_norm import BatchNorm
from mnist_numpy.model.layer.base import (
    Input,
)
from mnist_numpy.model.layer.linear import Linear
from mnist_numpy.model.layer.dense import Dense
from mnist_numpy.model.layer.output import SoftmaxOutputLayer
from mnist_numpy.types import (
    ActivationFn,
    Activations,
    D,
    GradLayer,
    HasParameterCount,
    LossContributor,
    SupportsDeserialize,
    SupportsReinitialisation,
    SupportsSerialize,
    SupportsUpdateParameters,
    UpdateGradientType,
    HasDimensions,
)


@runtime_checkable
class HiddenElement(HasDimensions, Protocol):
    def forward_prop(self, *, input_activations: Activations) -> Activations: ...
    def backward_prop(self, *, dZ: D[Activations]) -> D[Activations]: ...
    @property
    def input_dimensions(self) -> int: ...
    @property
    def output_dimensions(self) -> int: ...


@runtime_checkable
class OutputElement(Protocol):
    def forward_prop(self, *, input_activations: Activations) -> Activations: ...
    def backward_prop(self, *, Y_true: np.ndarray) -> D[Activations]: ...
    def serialize(self) -> SupportsDeserialize[OutputElement]: ...
    @property
    def input_dimensions(self) -> int: ...
    @property
    def output_dimensions(self) -> int: ...


type Element = HiddenElement | OutputElement


class MultiLayerPerceptron(ModelBase):
    @dataclass(frozen=True, kw_only=True)
    class Serialized:
        input_dimensions: int
        hidden: tuple[SupportsDeserialize, ...]
        output_element: SupportsDeserialize

    @classmethod
    def get_name(cls) -> str:
        return "mlp"

    @classmethod
    def get_description(cls) -> str:
        return "MultiLayer Perceptron"

    @property
    def elements(self) -> Sequence[Element]:
        return tuple([*self.hidden, self.output_element])

    @property
    def input_dimensions(self) -> int:
        return self.input_layer.input_dimensions

    @property
    def output_dimensions(self) -> int:
        return self.output_element.output_dimensions

    @classmethod
    def of(
        cls,
        *,
        dimensions: Sequence[int],
        activation_fn: ActivationFn = Identity,
        regularisers: Sequence[Regulariser] = (),
        batch_norm_batch_size: int | None = None,
        tracing_enabled: bool = False,
    ) -> Self:
        if len(dimensions) < 2:
            raise ValueError(f"{cls.__name__} must have at least 2 layers.")

        model_input_dimension, model_hidden_dimensions, model_output_dimension = (
            itemgetter(0, slice(1, -1), -1)(dimensions)
        )
        hidden_elements: Sequence[HiddenElement] = tuple(
            Dense(
                input_dimensions=input_dimensions,
                output_dimensions=output_dimensions,
                activation_fn=activation_fn,
                store_output_activations=tracing_enabled,
            )
            if batch_norm_batch_size is None
            else BatchNorm(
                input_dimensions=input_dimensions,
                output_dimensions=output_dimensions,
                activation_fn=activation_fn,
                batch_size=batch_norm_batch_size,
                store_output_activations=tracing_enabled,
            )
            for input_dimensions, output_dimensions in pairwise(
                [model_input_dimension, *model_hidden_dimensions]
            )
        )

        output_element = Output(
            layers=tuple(
                [
                    Linear(
                        input_dimensions=(
                            input_dimensions := last(model_hidden_dimensions)
                        ),
                        output_dimensions=model_output_dimension,
                        parameters=Linear.Parameters.xavier(
                            dim_in=input_dimensions, dim_out=model_output_dimension
                        ),
                        store_output_activations=tracing_enabled,
                    ),
                ]
            ),
            output_layer=SoftmaxOutputLayer(
                input_dimensions=model_output_dimension,
            ),
        )
        model = cls(
            input_dimensions=model_input_dimension,
            hidden_elements=hidden_elements,
            output_element=output_element,
        )
        for regulariser in regularisers:
            regulariser(model)
        return model

    def __init__(
        self,
        *,
        input_dimensions: int,
        hidden_elements: Sequence[HiddenElement],
        output_element: OutputElement,
    ):
        self._input_layer = Input(input_dimensions=input_dimensions)
        self._hidden_elements = hidden_elements
        self._output_element: OutputElement = output_element
        self._loss_contributors: MutableSequence[LossContributor] = []

    def reinitialise(self) -> None:
        for element in self.hidden:
            if isinstance(element, SupportsReinitialisation):
                element.reinitialise()
        if isinstance(self.output_element, SupportsReinitialisation):
            self.output_element.reinitialise()

    @property
    def loss_contributors(self) -> Sequence[LossContributor]:
        return self._loss_contributors

    def register_loss_contributor(self, contributor: LossContributor) -> None:
        self._loss_contributors.append(contributor)

    @property
    def hidden(self) -> Sequence[HiddenElement]:
        return self._hidden_elements

    def accept_hidden_element_visitor(
        self, visitor: Callable[[Element, int], Element]
    ) -> None:
        for index, element in enumerate(self.hidden):
            visitor(element, index)

    @property
    def input_layer(self) -> Input:
        return self._input_layer

    @property
    def output_element(self) -> OutputElement:
        return self._output_element

    def forward_prop(self, X: np.ndarray) -> Activations:
        return reduce(
            lambda A, element: element.forward_prop(input_activations=A),
            self.elements,
            Activations(X),
        )

    def backward_prop(self, *, Y_true: np.ndarray) -> None:
        reduce(
            lambda dZ, element: element.backward_prop(dZ=dZ),
            reversed(self.hidden),
            self.output_element.backward_prop(Y_true=Y_true),
        )

    def update_parameters(self) -> None:
        for hidden_element in self.hidden:
            if isinstance(hidden_element, SupportsUpdateParameters):
                hidden_element.update_parameters()

    def dump(self, io: IO[bytes]) -> None:
        pickle.dump(
            self.Serialized(
                input_dimensions=self.input_layer.input_dimensions,
                hidden=tuple(
                    hidden_element.serialize()
                    for hidden_element in self.hidden
                    if isinstance(hidden_element, SupportsSerialize)
                ),
                output_element=self.output_element.serialize(),
            ),
            io,
        )

    @classmethod
    def load(cls, source: IO[bytes], training: bool = False) -> Self:
        serialized = pickle.load(source)
        if not isinstance(serialized, cls.Serialized):
            raise ValueError(f"Invalid serialized model: {serialized}")
        return cls(
            input_dimensions=serialized.input_dimensions,
            hidden_elements=tuple(
                element.deserialize(training=training)
                for element in serialized.hidden
                if isinstance(element, SupportsDeserialize)
            ),
            output_element=serialized.output_element.deserialize(training=training),
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward_prop(X).argmax(axis=1)

    def compute_loss(self, X: np.ndarray, Y_true: np.ndarray) -> float:
        return sum(
            (loss_contributor() for loss_contributor in self.loss_contributors),
            start=1 / X.shape[0] * cross_entropy(self.forward_prop(X), Y_true),
        )

    def serialize(self) -> Serialized:
        return self.Serialized(
            input_dimensions=self.input_dimensions,
            hidden=tuple(
                hidden_element.serialize()
                for hidden_element in self.hidden
                if isinstance(hidden_element, SupportsSerialize)
            ),
            output_element=self.output_element.serialize(),
        )

    @property
    def grad_layers(self) -> Sequence[GradLayer]:
        def _grad_layers(element: Element) -> Iterator[GradLayer]:
            match element:
                case HiddenBlock():
                    return (
                        layer
                        for layer in element.layers
                        if isinstance(layer, GradLayer)
                    )
                case HiddenLayer() as layer:
                    return iter((layer,)) if isinstance(layer, GradLayer) else iter(())
                case _:
                    return iter(())

        return tuple(
            chain.from_iterable(_grad_layers(element) for element in self.elements)
        )

    def get_gradient_caches(self) -> UpdateGradientType:
        return tuple(layer.cache["dP"] for layer in self.grad_layers)

    def populate_caches(self, updates: UpdateGradientType) -> None:
        for layer, update in zip(self.grad_layers, updates, strict=True):
            layer.cache["dP"] = update

    @property
    def parameter_count(self) -> int:
        return sum(
            element.parameter_count
            for element in self.elements
            if isinstance(element, HasParameterCount)
        )

    @property
    def dimensions(self) -> Sequence[tuple[int, int]]:
        return tuple(
            (element.input_dimensions, element.output_dimensions)
            for element in self.elements
        )


type Regulariser = Callable[[MultiLayerPerceptron], None]
