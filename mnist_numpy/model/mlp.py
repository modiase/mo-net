from __future__ import annotations

import pickle
from collections.abc import Callable, MutableSequence
from dataclasses import dataclass
from functools import partial, reduce
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

from mnist_numpy.functions import (
    cross_entropy,
    Identity,
)
from mnist_numpy.model import ModelBase
from mnist_numpy.model.block.base import Base, Hidden, Output
from mnist_numpy.model.block.norm import BatchNormOptions, LayerNormOptions, Norm
from mnist_numpy.model.layer.base import Hidden as HiddenLayer
from mnist_numpy.model.layer.input import Input
from mnist_numpy.model.layer.linear import Linear
from mnist_numpy.model.block.dense import Dense
from mnist_numpy.model.layer.output import SoftmaxOutputLayer
from mnist_numpy.protos import (
    ActivationFn,
    Activations,
    Dimensions,
    GradLayer,
    HasDimensions,
    LossContributor,
    NormalisationType,
    SupportsDeserialize,
    SupportsReinitialisation,
    UpdateGradientType,
)


class MultiLayerPerceptron(ModelBase):
    @dataclass(frozen=True, kw_only=True)
    class Serialized:
        input_dimensions: tuple[int, ...]
        hidden_blocks: tuple[SupportsDeserialize, ...]
        output_block: SupportsDeserialize

    @classmethod
    def get_name(cls) -> str:
        return "mlp"

    @classmethod
    def get_description(cls) -> str:
        return "MultiLayer Perceptron"

    @property
    def blocks(self) -> Sequence[Base]:
        return tuple([*self.hidden_blocks, self.output_block])

    @property
    def input_dimensions(self) -> Dimensions:
        return self.input_layer.input_dimensions

    @property
    def output_dimensions(self) -> Dimensions:
        return self.output_block.output_layer.output_dimensions

    @overload
    @classmethod
    def of(
        cls,
        *,
        block_dimensions: Sequence[Dimensions],
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
    def of(
        cls,
        *,
        block_dimensions: Sequence[Dimensions],
        activation_fn: ActivationFn = Identity,
        regularisers: Sequence[Regulariser] = (),
        normalisation_type: Literal[NormalisationType.BATCH],
        batch_size: int,
        tracing_enabled: bool = False,
    ) -> Self: ...

    @classmethod
    def of(
        cls,
        *,
        block_dimensions: Sequence[Dimensions],
        activation_fn: ActivationFn = Identity,
        regularisers: Sequence[Regulariser] = (),
        normalisation_type: NormalisationType = NormalisationType.NONE,
        batch_size: int | None = None,
        tracing_enabled: bool = False,
    ) -> Self:
        if len(block_dimensions) < 2:
            raise ValueError(f"{cls.__name__} must have at least 2 layers.")

        model_input_dimension: Dimensions
        model_hidden_dimensions: Sequence[Dimensions]
        model_output_dimension: Dimensions
        model_input_dimension, model_hidden_dimensions, model_output_dimension = (
            itemgetter(0, slice(1, -1), -1)(block_dimensions)
        )
        Block: Callable[[Dimensions, Dimensions], Hidden]
        match normalisation_type:
            case NormalisationType.LAYER:
                Block = partial(
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
                Block = partial(
                    Norm,
                    activation_fn=activation_fn,
                    store_output_activations=tracing_enabled,
                    options=BatchNormOptions(
                        batch_size=batch_size,
                        momentum=0.9,
                    ),
                )
            case NormalisationType.NONE:
                Block = partial(
                    Dense,
                    activation_fn=activation_fn,
                    store_output_activations=tracing_enabled,
                )
            case never:
                assert_never(never)

        hidden_blocks: Sequence[Hidden] = tuple(
            Block(  # type: ignore[call-arg]
                input_dimensions=input_dimensions,
                output_dimensions=output_dimensions,
            )
            for input_dimensions, output_dimensions in pairwise(
                [model_input_dimension, *model_hidden_dimensions]
            )
        )

        output_block = Output(
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
            hidden_blocks=hidden_blocks,
            output_block=output_block,
        )
        for regulariser in regularisers:
            regulariser(model)
        return model

    def __init__(
        self,
        *,
        input_dimensions: Dimensions,
        hidden_blocks: Sequence[Hidden],
        output_block: Output,
    ):
        self._input_layer = Input(input_dimensions=input_dimensions)
        self._hidden_blocks = hidden_blocks
        self._output_block = output_block
        self._loss_contributors: MutableSequence[LossContributor] = []

    def reinitialise(self) -> None:
        for block in self.hidden_blocks:
            if isinstance(block, SupportsReinitialisation):
                block.reinitialise()
        if isinstance(self.output_block, SupportsReinitialisation):
            self.output_block.reinitialise()

    @property
    def loss_contributors(self) -> Sequence[LossContributor]:
        return self._loss_contributors

    def register_loss_contributor(self, contributor: LossContributor) -> None:
        self._loss_contributors.append(contributor)

    @property
    def hidden_blocks(self) -> Sequence[Hidden]:
        return self._hidden_blocks

    def accept_hidden_block_visitor(
        self, visitor: Callable[[Hidden, int], Hidden]
    ) -> None:
        for index, block in enumerate(self.hidden_blocks):
            visitor(block, index)

    @property
    def input_layer(self) -> Input:
        return self._input_layer

    @property
    def output_block(self) -> Output:
        return self._output_block

    def forward_prop(self, X: np.ndarray) -> Activations:
        return reduce(
            lambda A, block: block.forward_prop(input_activations=A),
            [*self.hidden_blocks, self.output_block],
            Activations(X),
        )

    def backward_prop(self, Y_true: np.ndarray) -> None:
        reduce(
            lambda dZ, block: block.backward_prop(dZ=dZ),
            reversed(self.hidden_blocks),
            self.output_block.backward_prop(Y_true=Y_true),
        )

    def update_parameters(self) -> None:
        for block in self.hidden_blocks:
            block.update_parameters()
        self.output_block.update_parameters()

    def dump(self, io: IO[bytes]) -> None:
        pickle.dump(
            self.Serialized(
                input_dimensions=tuple(self.input_layer.input_dimensions),
                hidden_blocks=tuple(block.serialize() for block in self.hidden_blocks),
                output_block=self.output_block.serialize(),
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
            hidden_blocks=tuple(
                block.deserialize(training=training)
                for block in serialized.hidden_blocks
            ),
            output_block=serialized.output_block.deserialize(training=training),
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
            hidden_blocks=tuple(block.serialize() for block in self.hidden_blocks),
            output_block=self.output_block.serialize(),
        )

    @property
    def grad_layers(self) -> Sequence[GradLayer]:
        return tuple(
            layer
            for block in tuple([*self.hidden_blocks, self.output_block])
            for layer in block.layers
            if isinstance(layer, GradLayer)
        )

    def get_gradient_caches(self) -> UpdateGradientType:
        return tuple(layer.cache["dP"] for layer in self.grad_layers)

    def populate_caches(self, updates: UpdateGradientType) -> None:
        for layer, update in zip(self.grad_layers, updates, strict=True):
            layer.cache["dP"] = update

    @property
    def parameter_count(self) -> int:
        return sum(block.parameter_count for block in self.blocks)

    @property
    def block_dimensions(self) -> Sequence[tuple[Dimensions, Dimensions]]:
        return tuple(HasDimensions.get_dimensions(block) for block in self.blocks)

    def append_block(self, block: Hidden) -> None:
        self._hidden_blocks = tuple([*self._hidden_blocks, block])

    def prepend_block(self, block: Hidden) -> None:
        self._hidden_blocks = tuple([block, *self._hidden_blocks])

    def append_layer(self, layer: HiddenLayer) -> None:
        last(self._hidden_blocks).append_layer(layer)

    def prepend_layer(self, layer: HiddenLayer) -> None:
        first(self._hidden_blocks).prepend_layer(layer)


type Regulariser = Callable[[MultiLayerPerceptron], None]
