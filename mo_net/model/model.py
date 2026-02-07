from __future__ import annotations

import contextlib
import functools
import pickle
from collections.abc import Callable, Mapping, MutableSequence
from dataclasses import dataclass
from itertools import chain
from operator import itemgetter
from pathlib import Path
from typing import (
    IO,
    Literal,
    Self,
    Sequence,
    assert_never,
    overload,
)

import jax
import jax.numpy as jnp
from more_itertools import first, last, pairwise

from mo_net.functions import (
    ActivationFn,
    LossFn,
    identity,
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
from mo_net.model.module.rnn import RNN
from mo_net.protos import (
    Activations,
    D,
    Dimensions,
    GradLayer,
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
        activation_fn: ActivationFn = identity,
        batch_size: None = None,
        dropout_keep_probs: Sequence[float] | None = None,
        key: jax.Array,
        module_dimensions: Sequence[Dimensions],
        normalisation_type: Literal[NormalisationType.NONE, NormalisationType.LAYER] = (
            NormalisationType.NONE
        ),
        regularisers: Sequence[Regulariser] = (),
        tracing_enabled: bool = False,
    ) -> Self: ...

    @overload
    @classmethod
    def mlp_of(
        cls,
        *,
        activation_fn: ActivationFn = identity,
        batch_size: int,
        dropout_keep_probs: Sequence[float] | None = None,
        key: jax.Array,
        module_dimensions: Sequence[Dimensions],
        normalisation_type: Literal[NormalisationType.BATCH],
        regularisers: Sequence[Regulariser] = (),
        tracing_enabled: bool = False,
    ) -> Self: ...

    @classmethod
    def mlp_of(
        cls,
        *,
        activation_fn: ActivationFn = identity,
        batch_size: int | None = None,
        dropout_keep_probs: Sequence[float] | None = None,
        key: jax.Array,
        module_dimensions: Sequence[Dimensions],
        normalisation_type: NormalisationType = NormalisationType.NONE,
        regularisers: Sequence[Regulariser] = (),
        tracing_enabled: bool = False,
    ) -> Self:
        if len(module_dimensions) < 2:
            raise ValueError(f"{cls.__name__} must have at least 2 layers.")

        model_input_dimension: Dimensions
        model_hidden_dimensions: Sequence[Dimensions]
        model_output_dimension: Dimensions
        model_input_dimension, model_hidden_dimensions, model_output_dimension = (
            itemgetter(0, slice(1, -1), -1)(module_dimensions)
        )
        ModuleFactory: Callable[[Dimensions, Dimensions], Hidden]
        match normalisation_type:
            case NormalisationType.LAYER:

                def _norm_factory(
                    input_dimensions: Dimensions, output_dimensions: Dimensions
                ) -> Hidden:
                    nonlocal key
                    key, subkey = jax.random.split(key)
                    return Norm(
                        input_dimensions=input_dimensions,
                        output_dimensions=output_dimensions,
                        activation_fn=activation_fn,
                        store_output_activations=tracing_enabled,
                        options=LayerNormOptions(),
                        key=subkey,
                    )

                ModuleFactory = _norm_factory
            case NormalisationType.BATCH:
                if batch_size is None:
                    raise ValueError(
                        "Batch size must be provided when using batch normalisation."
                    )

                def _batch_norm_factory(
                    input_dimensions: Dimensions, output_dimensions: Dimensions
                ) -> Hidden:
                    nonlocal key
                    key, subkey = jax.random.split(key)
                    return Norm(
                        input_dimensions=input_dimensions,
                        output_dimensions=output_dimensions,
                        activation_fn=activation_fn,
                        store_output_activations=tracing_enabled,
                        options=BatchNormOptions(
                            momentum=0.9,
                        ),
                        key=subkey,
                    )

                ModuleFactory = _batch_norm_factory
            case NormalisationType.NONE:

                def _dense_factory(
                    input_dimensions: Dimensions, output_dimensions: Dimensions
                ) -> Hidden:
                    nonlocal key
                    key, subkey = jax.random.split(key)
                    return Dense(
                        input_dimensions=input_dimensions,
                        output_dimensions=output_dimensions,
                        activation_fn=activation_fn,
                        store_output_activations=tracing_enabled,
                        key=subkey,
                    )

                ModuleFactory = _dense_factory
            case never:
                assert_never(never)

        hidden_modules: Sequence[Hidden] = tuple(
            ModuleFactory(  # type: ignore[call-arg]
                input_dimensions=input_dimensions,
                output_dimensions=output_dimensions,
            )
            for input_dimensions, output_dimensions in pairwise(
                [model_input_dimension, *model_hidden_dimensions]
            )
        )

        key, subkey = jax.random.split(key)
        output_module = Output(
            layers=tuple(
                [
                    Linear(
                        input_dimensions=(last(model_hidden_dimensions)),
                        output_dimensions=model_output_dimension,
                        parameters_init_fn=functools.partial(
                            Linear.Parameters.xavier,
                            key=subkey,
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
            key, subkey = jax.random.split(key)
            Dropout.attach_dropout_layers(  # noqa: F821
                model=model,
                keep_probs=dropout_keep_probs,
                training=True,
                key=subkey,
            )
        return model

    @classmethod
    def rnn_of(
        cls,
        *,
        activation_fn: ActivationFn = identity,
        key: jax.Array,
        module_dimensions: Sequence[Dimensions],
        regularisers: Sequence[Regulariser] = (),
        return_sequences: bool = False,
        stateful: bool = False,
        tracing_enabled: bool = False,
    ) -> Self:
        """
        Create a stacked RNN model.

        Args:
            activation_fn: Activation function for RNN layers (default: identity/tanh)
            key: JAX random key for parameter initialization
            module_dimensions: Sequence of dimensions [input_dim, hidden_dim1, ..., output_dim]
            regularisers: Sequence of regularization functions
            return_sequences: If True, all RNN layers return sequences. If False, only last layer
                            returns final hidden state
            stateful: Whether RNN layers maintain state across batches
            tracing_enabled: Whether to enable activation tracing

        Returns:
            Model with stacked RNN layers
        """
        if len(module_dimensions) < 2:
            raise ValueError(f"{cls.__name__} must have at least 2 layers.")

        model_input_dimension: Dimensions
        model_hidden_dimensions: Sequence[Dimensions]
        model_output_dimension: Dimensions
        model_input_dimension, model_hidden_dimensions, model_output_dimension = (
            itemgetter(0, slice(1, -1), -1)(module_dimensions)
        )

        # Build stacked RNN modules
        hidden_modules: list[Hidden] = []
        for i, (input_dim, hidden_dim) in enumerate(
            pairwise([model_input_dimension, *model_hidden_dimensions])
        ):
            key, subkey = jax.random.split(key)
            # For intermediate layers, we want to return sequences to feed to next layer
            # For the last RNN layer, use the return_sequences parameter
            is_last_rnn = i == len(model_hidden_dimensions) - 1
            rnn_module = RNN(
                input_dimensions=input_dim,
                hidden_dimensions=hidden_dim,
                activation_fn=activation_fn,
                key=subkey,
                return_sequences=True if not is_last_rnn else return_sequences,
                stateful=stateful,
                store_output_activations=tracing_enabled,
            )
            hidden_modules.append(rnn_module)

        # Build output module
        # The output dimension depends on whether last RNN returns sequences
        final_rnn_output_dim = (
            model_hidden_dimensions[-1]
            if model_hidden_dimensions
            else model_input_dimension
        )

        key, subkey = jax.random.split(key)
        output_module = Output(
            layers=tuple(
                [
                    Linear(
                        input_dimensions=final_rnn_output_dim,
                        output_dimensions=model_output_dimension,
                        parameters_init_fn=functools.partial(
                            Linear.Parameters.xavier,
                            key=subkey,
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

    def forward_prop(self, X: jnp.ndarray) -> Activations:
        activations = self.input_layer.forward_prop(Activations(X))
        for module in self.hidden_modules:
            activations = module.forward_prop(input_activations=activations)
        activations = self.output_module.forward_prop(input_activations=activations)
        return activations

    def forward_prop_to(self, X: jnp.ndarray, layer_index: str) -> jnp.ndarray:
        """
        Forward propagate input to a specific layer and return its activations.

        Args:
            X: Input data
            layer_index: String in format "m" (mth module) or "m:n" (nth layer in mth module)

        Returns:
            Activations from the specified layer
        """
        parts = layer_index.split(":")
        if len(parts) > 2:
            raise ValueError(
                f"Invalid layer_index format '{layer_index}'. Expected 'm' or 'm:n' where m,n are integers."
            )

        layer_idx = -1
        if len(parts) == 2:
            module_idx_str, layer_idx_str = parts
            module_idx = int(module_idx_str)
            layer_idx = int(layer_idx_str)
        else:
            module_idx = int(parts[0])
        # TODO(#34): We should find a way to forward propagate to a specific layer
        # without forwarding the entire model
        self.forward_prop(X)

        target_layer = list(chain(self.hidden_modules, (self.output_module,)))[
            module_idx
        ].layers[layer_idx]
        if hasattr(target_layer, "_cache"):
            cache = getattr(target_layer, "_cache", None)
            if cache is not None and "input_activations" in cache:
                return cache["input_activations"]

        raise ValueError(f"No cached activations found for layer {layer_index}")

    def backward_prop(self, Y_true: jnp.ndarray) -> D[Activations]:
        dZ = self.output_module.backward_prop(Y_true=Y_true)
        for module in reversed(self.hidden_modules):
            dZ = module.backward_prop(dZ=dZ)
        return self.input_layer.backward_prop(dZ)

    def update_parameters(self) -> None:
        for module in self.hidden_modules:
            module.update_parameters()
        self.output_module.update_parameters()

    @overload
    def dump(self, io: IO[bytes]) -> None: ...

    @overload
    def dump(self, io: Path) -> None: ...

    def dump(self, io: IO[bytes] | Path) -> None:
        with (
            open(io, "wb")
            if isinstance(io, Path)
            else contextlib.nullcontext(io) as file_io
        ):
            pickle.dump(
                self.Serialized(
                    input_dimensions=tuple(self.input_layer.input_dimensions),
                    hidden_modules=tuple(
                        module.serialize() for module in self.hidden_modules
                    ),
                    output_module=self.output_module.serialize(),
                ),
                file_io,
            )

    @classmethod
    def load(
        cls,
        source: IO[bytes] | Path,
        training: bool = False,
        freeze_parameters: bool = False,
    ) -> Self:
        if isinstance(source, Path):
            with open(source, "rb") as f:
                serialized = pickle.load(f)
        else:
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

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        return jnp.argmax(self.forward_prop(X), axis=1)

    def compute_loss(
        self, X: jnp.ndarray, Y_true: jnp.ndarray, loss_fn: LossFn
    ) -> float:
        Y_pred = self.forward_prop(Activations(X))
        return loss_fn(Y_pred, Y_true) + sum(
            contributor() for contributor in self.loss_contributors
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
        caches = [layer.cache["dP"] for layer in self.grad_layers]
        # Include output layer if it has gradients (e.g., HierarchicalSoftmaxOutputLayer)
        output_layer = self.output_module.output_layer
        if isinstance(output_layer, GradLayer):
            caches.append(output_layer.cache["dP"])
        return tuple(caches)

    def populate_caches(self, updates: UpdateGradientType) -> None:
        # Check if output layer has gradients
        output_layer = self.output_module.output_layer
        has_grad_output = isinstance(output_layer, GradLayer)

        expected_len = len(self.grad_layers) + (1 if has_grad_output else 0)
        if len(updates) != expected_len:
            raise ValueError(f"Expected {expected_len} updates, got {len(updates)}")

        for layer, update in zip(self.grad_layers, updates[: len(self.grad_layers)]):
            layer.cache["dP"] = update

        if has_grad_output:
            output_layer.cache["dP"] = updates[-1]

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

    @property
    def output(self) -> Output:
        return self._output_module

    def __getitem__(self, index: str):
        """
        Access layers/modules using string indexing.

        Args:
            index: String in format "m" (mth module) or "m:n" (nth layer in mth module)

        Returns:
            The requested layer or module

        Examples:
            model["0"] -> first hidden module
            model["1:2"] -> second layer in second module
        """
        parts = index.split(":")
        if len(parts) > 2:
            raise ValueError(
                f"Invalid index format '{index}'. Expected 'm' or 'm:n' where m,n are integers."
            )

        layer_idx = None
        if len(parts) == 2:
            module_idx_str, layer_idx_str = parts
            module_idx = int(module_idx_str)
            layer_idx = int(layer_idx_str)
        else:
            module_idx = int(parts[0])

        if layer_idx is None:
            return self.hidden_modules[module_idx]
        else:
            return self.hidden_modules[module_idx].layers[layer_idx]


type Regulariser = Callable[[Model], None]
