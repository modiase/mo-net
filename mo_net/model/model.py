from __future__ import annotations

import pickle
from collections.abc import Callable, Mapping, MutableSequence
from dataclasses import dataclass
from functools import partial, reduce
from itertools import chain
from operator import itemgetter
from typing import (
    IO,
    Literal,
    Self,
    Sequence,
    assert_never,
    overload,
)

import numpy as np
from more_itertools import first, last, pairwise

from mo_net.functions import (
    Identity,
    cross_entropy,
)
from mo_net.model import ModelBase
from mo_net.model.layer.base import Hidden as HiddenLayer
from mo_net.model.layer.base import ParametrisedHidden
from mo_net.model.layer.dropout import Dropout
from mo_net.model.layer.input import Input
from mo_net.model.layer.linear import Linear
from mo_net.model.layer.output import OutputLayer, RawOutputLayer, SoftmaxOutputLayer
from mo_net.model.module.base import Base, Hidden, Output
from mo_net.model.module.dense import Dense
from mo_net.model.module.norm import BatchNormOptions, LayerNormOptions, Norm
from mo_net.protos import (
    ActivationFn,
    Activations,
    Dimensions,
    HasDimensions,
    LossContributor,
    NormalisationType,
    SupportsDeserialize,
    SupportsReinitialisation,
    UpdateGradientType,
)


class Model(ModelBase):
    @dataclass(frozen=True, kw_only=True)
    class Serialized:
        input_dimensions: tuple[int, ...]
        hidden_modules: tuple[SupportsDeserialize, ...]
        output_module: SupportsDeserialize

    @classmethod
    def get_name(cls) -> str:
        return "model"

    @classmethod
    def get_description(cls) -> str:
        return "Model"

    @property
    def modules(self) -> Sequence[Base]:
        return tuple([*self.hidden_modules, self.output_module])

    @property
    def input_dimensions(self) -> Dimensions:
        return self.input_layer.input_dimensions

    @property
    def output_dimensions(self) -> Dimensions:
        return self.output_module.output_layer.output_dimensions

    @overload
    @classmethod
    def mlp_of(
        cls,
        *,
        module_dimensions: Sequence[Dimensions],
        activation_fn: ActivationFn = Identity,
        regularisers: Sequence[Regulariser] = (),
        normalisation_type: Literal[NormalisationType.NONE, NormalisationType.LAYER] = (
            NormalisationType.NONE
        ),
        batch_size: None = None,
        tracing_enabled: bool = False,
    ) -> Self: ...

    @overload
    @classmethod
    def mlp_of(
        cls,
        *,
        module_dimensions: Sequence[Dimensions],
        activation_fn: ActivationFn = Identity,
        regularisers: Sequence[Regulariser] = (),
        normalisation_type: Literal[NormalisationType.BATCH],
        batch_size: int,
        tracing_enabled: bool = False,
    ) -> Self: ...

    @classmethod
    def mlp_of(
        cls,
        *,
        module_dimensions: Sequence[Dimensions],
        activation_fn: ActivationFn = Identity,
        regularisers: Sequence[Regulariser] = (),
        normalisation_type: NormalisationType = NormalisationType.NONE,
        batch_size: int | None = None,
        tracing_enabled: bool = False,
        dropout_keep_probs: Sequence[float] | None = None,
    ) -> Self:
        if len(module_dimensions) < 2:
            raise ValueError(f"{cls.__name__} must have at least 2 layers.")

        model_input_dimension: Dimensions
        model_hidden_dimensions: Sequence[Dimensions]
        model_output_dimension: Dimensions
        model_input_dimension, model_hidden_dimensions, model_output_dimension = (
            itemgetter(0, slice(1, -1), -1)(module_dimensions)
        )
        Module: Callable[[Dimensions, Dimensions], Hidden]
        match normalisation_type:
            case NormalisationType.LAYER:
                Module = partial(
                    Norm,
                    activation_fn=activation_fn,
                    store_output_activations=tracing_enabled,
                    options=LayerNormOptions(),
                )
            case NormalisationType.BATCH:
                if batch_size is None:
                    raise ValueError(
                        "Batch size must be provided when using batch normalisation."
                    )
                Module = partial(
                    Norm,
                    activation_fn=activation_fn,
                    store_output_activations=tracing_enabled,
                    options=BatchNormOptions(
                        momentum=0.9,
                    ),
                )
            case NormalisationType.NONE:
                Module = partial(
                    Dense,
                    activation_fn=activation_fn,
                    store_output_activations=tracing_enabled,
                )
            case never:
                assert_never(never)

        hidden_modules: Sequence[Hidden] = tuple(
            Module(  # type: ignore[call-arg]
                input_dimensions=input_dimensions,
                output_dimensions=output_dimensions,
            )
            for input_dimensions, output_dimensions in pairwise(
                [model_input_dimension, *model_hidden_dimensions]
            )
        )

        output_module = Output(
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
            hidden=hidden_modules,
            output=output_module,
        )
        for regulariser in regularisers:
            regulariser(model)
        if dropout_keep_probs:
            Dropout.attach_dropout_layers(  # noqa: F821
                model=model,
                keep_probs=dropout_keep_probs,
                training=True,
            )
        return model

    def __init__(
        self,
        *,
        input_dimensions: Dimensions,
        hidden: Sequence[Hidden | HiddenLayer],
        output: Output | OutputLayer | None = None,
    ):
        self._input_layer = Input(input_dimensions=input_dimensions)
        if invalid := tuple(
            module for module in hidden if not isinstance(module, (Hidden, HiddenLayer))
        ):
            raise ValueError(f"Invalid hidden modules: {invalid}")
        self._hidden_modules = tuple(
            module if isinstance(module, Hidden) else Hidden(layers=(module,))
            for module in hidden
        )
        if output is None:
            output = RawOutputLayer(
                input_dimensions=last(self._hidden_modules).output_dimensions
            )
        if not isinstance(output, (Output, OutputLayer)):
            raise ValueError(f"Invalid output module: {output}")
        self._output_module = (
            output if isinstance(output, Output) else Output(output_layer=output)
        )
        self._loss_contributors: MutableSequence[LossContributor] = []
        self._layer_id_to_parametrised_layer: Mapping[str, ParametrisedHidden] = {
            layer.layer_id: layer for layer in self.grad_layers
        }

    def reinitialise(self) -> None:
        for module in self.hidden_modules:
            if isinstance(module, SupportsReinitialisation):
                module.reinitialise()
        if isinstance(self.output_module, SupportsReinitialisation):
            self.output_module.reinitialise()

    @property
    def loss_contributors(self) -> Sequence[LossContributor]:
        return self._loss_contributors

    def register_loss_contributor(self, contributor: LossContributor) -> None:
        self._loss_contributors.append(contributor)

    @property
    def hidden_modules(self) -> Sequence[Hidden]:
        return self._hidden_modules

    def accept_hidden_module_visitor(
        self, visitor: Callable[[Hidden, int], Hidden]
    ) -> None:
        for index, module in enumerate(self.hidden_modules):
            visitor(module, index)

    @property
    def input_layer(self) -> Input:
        return self._input_layer

    @property
    def output_module(self) -> Output:
        return self._output_module

    def forward_prop(self, X: np.ndarray) -> Activations:
        return reduce(
            lambda A, module: module.forward_prop(input_activations=A),
            [*self.hidden_modules, self.output_module],
            Activations(X),
        )

    def backward_prop(self, Y_true: np.ndarray) -> None:
        reduce(
            lambda dZ, module: module.backward_prop(dZ=dZ),
            reversed(self.hidden_modules),
            self.output_module.backward_prop(Y_true=Y_true),
        )

    def update_parameters(self) -> None:
        for module in self.hidden_modules:
            module.update_parameters()
        self.output_module.update_parameters()

    def dump(self, io: IO[bytes]) -> None:
        pickle.dump(
            self.Serialized(
                input_dimensions=tuple(self.input_layer.input_dimensions),
                hidden_modules=tuple(
                    module.serialize() for module in self.hidden_modules
                ),
                output_module=self.output_module.serialize(),
            ),
            io,
        )

    @classmethod
    def load(
        cls, source: IO[bytes], training: bool = False, freeze_parameters: bool = False
    ) -> Self:
        serialized = pickle.load(source)
        if not isinstance(serialized, cls.Serialized):
            raise ValueError(f"Invalid serialized model: {serialized}")
        return cls(
            input_dimensions=serialized.input_dimensions,
            hidden=tuple(
                module.deserialize(
                    training=training, freeze_parameters=freeze_parameters
                )
                for module in serialized.hidden_modules
            ),
            output=serialized.output_module.deserialize(
                training=training, freeze_parameters=freeze_parameters
            ),
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
            input_dimensions=tuple(self.input_dimensions),
            hidden_modules=tuple(module.serialize() for module in self.hidden_modules),
            output_module=self.output_module.serialize(),
        )

    @property
    def grad_layers(self) -> Sequence[ParametrisedHidden]:
        return tuple(
            layer
            for module in tuple([*self.hidden_modules, self.output_module])
            for layer in module.layers
            if isinstance(layer, ParametrisedHidden)
        )

    def get_gradient_caches(self) -> UpdateGradientType:
        return tuple(layer.cache["dP"] for layer in self.grad_layers)

    def populate_caches(self, updates: UpdateGradientType) -> None:
        for layer, update in zip(self.grad_layers, updates, strict=True):
            layer.cache["dP"] = update

    @property
    def parameter_count(self) -> int:
        return sum(module.parameter_count for module in self.modules)

    @property
    def parameter_n_bytes(self) -> int:
        return sum(layer.parameter_nbytes for layer in self.grad_layers)

    @property
    def module_dimensions(self) -> Sequence[tuple[Dimensions, Dimensions]]:
        return tuple(HasDimensions.get_dimensions(module) for module in self.modules)

    def append_module(self, module: Hidden) -> None:
        self._hidden_modules = tuple([*self._hidden_modules, module])

    def prepend_module(self, module: Hidden) -> None:
        self._hidden_modules = tuple([module, *self._hidden_modules])

    def append_layer(self, layer: HiddenLayer) -> None:
        last(self._hidden_modules).append_layer(layer)

    def prepend_layer(self, layer: HiddenLayer) -> None:
        first(self._hidden_modules).prepend_layer(layer)

    def get_parametrised_layer(self, layer_id: str) -> ParametrisedHidden:
        return self._layer_id_to_parametrised_layer[layer_id]

    @property
    def layers(self) -> Sequence[Base]:
        return tuple(
            chain.from_iterable(
                module.layers  # type: ignore[misc]
                if isinstance(module, Base)
                else chain(module.layers, (module.output_layer,))  # type: ignore[arg-type]
                if isinstance(module, Output)  # type: ignore[arg-type]
                else (module,)
                for module in chain(
                    [self.input_layer], self.hidden_modules, [self.output_module]
                )
            )
        )

    def print(self) -> str:
        return ", ".join(f"{layer}" for layer in self.layers)


type Regulariser = Callable[[Model], None]
