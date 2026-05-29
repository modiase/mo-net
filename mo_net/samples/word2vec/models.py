"""Word2Vec model classes (CBOW, SkipGram, PredictModel).

Extracted from ``__main__.py`` so pickled checkpoints reference a stable
module path. When ``__main__.py`` runs as ``python -m mo_net.samples.word2vec``
it gets loaded under the name ``__main__``; pickling classes defined there
records the unstable ``__main__.CBOWModel`` path, making the resulting
zips unloadable from other processes. Classes here always live at
``mo_net.samples.word2vec.models``.
"""

from __future__ import annotations

import contextlib
import pickle
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, cast

import jax
import jax.numpy as jnp

from mo_net.model.layer.average import Average
from mo_net.model.layer.base import Hidden as HiddenLayer
from mo_net.model.layer.embedding import Embedding
from mo_net.model.layer.linear import Linear
from mo_net.model.layer.output import (
    HierarchicalSoftmaxOutputLayer,
    OutputLayer,
    SparseCategoricalSoftmaxOutputLayer,
)
from mo_net.model.model import Model
from mo_net.model.module.base import Hidden, Output
from mo_net.protos import (
    Activations,
    D,
    Dimensions,
    SupportsDeserialize,
)
from mo_net.samples.word2vec.softmax_strategy import SoftmaxConfig, SoftmaxStrategy
from mo_net.samples.word2vec.vocab import Vocab


def _negative_sampling_backward(
    *,
    model: CBOWModel | SkipGramModel,
    Y_true: jnp.ndarray,
) -> D[Activations]:
    """Backward pass through positive + sampled negatives.

    Replaces Model.backward_prop's full-softmax derivative with one that only
    updates rows for the true label plus a small set of negative samples.
    Mutates model._key (advances PRNG).
    """
    output_layer = model.output_module.output_layer
    if not isinstance(output_layer, SparseCategoricalSoftmaxOutputLayer):
        raise TypeError(
            "Negative sampling requires SparseCategoricalSoftmaxOutputLayer, "
            f"got {type(output_layer).__name__}"
        )

    model._key, subkey = jax.random.split(model._key)
    Y_negative = jax.random.choice(
        subkey,
        model.embedding_layer.vocab_size,
        shape=(Y_true.shape[0] * model._negative_samples,),
        p=model._negative_sampling_dist,
    )

    dZ = output_layer.backward_prop_with_negative(Y_true=Y_true, Y_negative=Y_negative)
    for layer in reversed(model.output_module.layers):
        dZ = layer.backward_prop(dZ=dZ)
    for module in reversed(model.hidden_modules):
        dZ = module.backward_prop(dZ=dZ)
    return model.input_layer.backward_prop(dZ)


class CBOWModel(Model):
    @dataclass(frozen=True, kw_only=True)
    class Serialized:  # type: ignore[misc]  # Intentional override of parent's Serialized class
        input_dimensions: tuple[int, ...]
        hidden_modules: tuple[SupportsDeserialize, ...]
        output_module: SupportsDeserialize
        softmax_strategy: SoftmaxStrategy = SoftmaxStrategy.FULL
        negative_samples: int = 5

    def __init__(
        self,
        *,
        hidden: Sequence[Hidden | HiddenLayer],
        input_dimensions: Dimensions,
        output: Output | OutputLayer | None = None,
        softmax_strategy: SoftmaxStrategy = SoftmaxStrategy.FULL,
        negative_samples: int = 5,
        negative_sampling_dist: jnp.ndarray | None = None,
        key: jax.Array | None = None,
    ):
        super().__init__(
            input_dimensions=input_dimensions, hidden=hidden, output=output
        )
        self._softmax_strategy = softmax_strategy
        self._negative_samples = negative_samples
        self._negative_sampling_dist = negative_sampling_dist
        self._key = key if key is not None else jax.random.PRNGKey(0)

    def backward_prop(self, Y_true: jnp.ndarray) -> D[Activations]:
        if self._softmax_strategy == SoftmaxStrategy.NEGATIVE_SAMPLING and isinstance(
            self.output_module.output_layer, SparseCategoricalSoftmaxOutputLayer
        ):
            return _negative_sampling_backward(model=self, Y_true=Y_true)
        return super().backward_prop(Y_true=Y_true)

    @classmethod
    def get_name(cls) -> str:
        return "cbow"

    @classmethod
    def get_description(cls) -> str:
        return "Continuous Bag of Words Model"

    @classmethod
    def create(
        cls,
        *,
        context_size: int,
        embedding_dim: int,
        key: jax.Array,
        softmax_config: SoftmaxConfig,
        negative_sampling_dist: jnp.ndarray | None = None,
        tracing_enabled: bool = False,
        vocab: Vocab | None = None,
        vocab_size: int | None = None,
    ) -> CBOWModel:
        if vocab is not None:
            vocab_size = len(vocab)
        elif vocab_size is None:
            raise ValueError("Must provide either vocab or vocab_size")

        key1, key2, key3 = jax.random.split(key, 3)

        # Build output layer based on softmax strategy. NEGATIVE_SAMPLING and FULL
        # construct the same layers; backward_prop dispatches on strategy.
        match softmax_config.strategy:
            case SoftmaxStrategy.HIERARCHICAL:
                if vocab is None:
                    raise ValueError(
                        "vocab is required for hierarchical softmax (needed to build Huffman tree)"
                    )
                output = Output(
                    layers=(),
                    output_layer=HierarchicalSoftmaxOutputLayer(
                        input_dimensions=(embedding_dim,),
                        vocab=vocab,
                        key=key3,
                    ),
                )
            case SoftmaxStrategy.NEGATIVE_SAMPLING | SoftmaxStrategy.FULL:
                output = Output(
                    layers=(
                        Linear(
                            input_dimensions=(embedding_dim,),
                            output_dimensions=(vocab_size,),
                            parameters_init_fn=lambda dim_in, dim_out: (
                                Linear.Parameters.xavier(dim_in, dim_out, key=key2)
                            ),
                            store_output_activations=tracing_enabled,
                        ),
                    ),
                    output_layer=SparseCategoricalSoftmaxOutputLayer(
                        input_dimensions=(vocab_size,)
                    ),
                )

        return cls(
            input_dimensions=(context_size * 2,),
            hidden=(
                Hidden(
                    layers=(
                        Embedding(
                            input_dimensions=(context_size * 2,),
                            output_dimensions=(context_size * 2, embedding_dim),
                            vocab_size=vocab_size,
                            parameters_init_fn=Embedding.Parameters.word2vec,
                            store_output_activations=tracing_enabled,
                            key=key1,
                        ),
                        Average(
                            input_dimensions=(context_size * 2, embedding_dim),
                            axis=0,
                        ),
                    )
                ),
            ),
            output=output,
            softmax_strategy=softmax_config.strategy,
            negative_samples=softmax_config.negative_samples or 5,
            negative_sampling_dist=negative_sampling_dist,
            key=key,
        )

    @property
    def embedding_layer(self) -> Embedding:
        return cast(Embedding, self.hidden_modules[0].layers[0])

    @property
    def embeddings(self) -> jnp.ndarray:
        return self.embedding_layer.parameters.embeddings

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
                    softmax_strategy=self._softmax_strategy,
                    negative_samples=self._negative_samples,
                ),
                file_io,
            )

    @classmethod
    def load(
        cls,
        source: IO[bytes] | Path,
        training: bool = False,
        freeze_parameters: bool = False,
        key: jax.Array | None = None,
        negative_sampling_dist: jnp.ndarray | None = None,
    ) -> CBOWModel:
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
            softmax_strategy=getattr(
                serialized, "softmax_strategy", SoftmaxStrategy.FULL
            ),
            negative_samples=getattr(serialized, "negative_samples", 5),
            negative_sampling_dist=negative_sampling_dist,
            key=key,
        )


class SkipGramModel(Model):
    @dataclass(frozen=True, kw_only=True)
    class Serialized:  # type: ignore[misc]  # Intentional override of parent's Serialized class
        input_dimensions: tuple[int, ...]
        hidden_modules: tuple[SupportsDeserialize, ...]
        output_module: SupportsDeserialize
        negative_samples: int
        softmax_strategy: SoftmaxStrategy = SoftmaxStrategy.NEGATIVE_SAMPLING

    def __init__(
        self,
        *,
        hidden: Sequence[Hidden | HiddenLayer],
        input_dimensions: Dimensions,
        key: jax.Array,
        negative_samples: int = 5,
        negative_sampling_dist: jnp.ndarray | None = None,
        output: Output | OutputLayer | None = None,
        softmax_strategy: SoftmaxStrategy = SoftmaxStrategy.NEGATIVE_SAMPLING,
    ):
        super().__init__(
            input_dimensions=input_dimensions, hidden=hidden, output=output
        )
        self._key = key
        self._negative_samples = negative_samples
        self._negative_sampling_dist = negative_sampling_dist
        self._softmax_strategy = softmax_strategy

    @classmethod
    def get_name(cls) -> str:
        return "skipgram"

    @classmethod
    def get_description(cls) -> str:
        return "Skip-Gram Model"

    @classmethod
    def create(
        cls,
        *,
        embedding_dim: int,
        key: jax.Array,
        negative_samples: int,
        softmax_config: SoftmaxConfig,
        negative_sampling_dist: jnp.ndarray | None = None,
        tracing_enabled: bool = False,
        vocab: Vocab | None = None,
        vocab_size: int | None = None,
    ) -> SkipGramModel:
        if vocab is not None:
            vocab_size = len(vocab)
        elif vocab_size is None:
            raise ValueError("Must provide either vocab or vocab_size")

        key1, key2, key3 = jax.random.split(key, 3)

        match softmax_config.strategy:
            case SoftmaxStrategy.HIERARCHICAL:
                if vocab is None:
                    raise ValueError(
                        "vocab is required for hierarchical softmax (needed to build Huffman tree)"
                    )
                output = Output(
                    layers=(),
                    output_layer=HierarchicalSoftmaxOutputLayer(
                        input_dimensions=(embedding_dim,),
                        vocab=vocab,
                        key=key3,
                    ),
                )
            case SoftmaxStrategy.NEGATIVE_SAMPLING | SoftmaxStrategy.FULL:
                output = Output(
                    layers=(
                        Linear(
                            input_dimensions=(embedding_dim,),
                            output_dimensions=(vocab_size,),
                            parameters_init_fn=lambda dim_in, dim_out: (
                                Linear.Parameters.xavier(dim_in, dim_out, key=key2)
                            ),
                            store_output_activations=tracing_enabled,
                        ),
                    ),
                    output_layer=SparseCategoricalSoftmaxOutputLayer(
                        input_dimensions=(vocab_size,)
                    ),
                )

        return cls(
            input_dimensions=(1,),
            hidden=(
                Hidden(
                    layers=(
                        Embedding(
                            input_dimensions=(1,),
                            output_dimensions=(1, embedding_dim),
                            vocab_size=vocab_size,
                            parameters_init_fn=Embedding.Parameters.word2vec,
                            store_output_activations=tracing_enabled,
                            key=key1,
                        ),
                        Average(
                            input_dimensions=(1, embedding_dim),
                            axis=0,
                        ),
                    )
                ),
            ),
            output=output,
            key=key,
            negative_samples=negative_samples,
            softmax_strategy=softmax_config.strategy,
            negative_sampling_dist=negative_sampling_dist,
        )

    def backward_prop(self, Y_true: jnp.ndarray) -> D[Activations]:
        if self._softmax_strategy == SoftmaxStrategy.NEGATIVE_SAMPLING and isinstance(
            self.output_module.output_layer, SparseCategoricalSoftmaxOutputLayer
        ):
            return _negative_sampling_backward(model=self, Y_true=Y_true)
        return super().backward_prop(Y_true=Y_true)

    @property
    def embedding_layer(self) -> Embedding:
        return cast(Embedding, self.hidden_modules[0].layers[0])

    @property
    def embeddings(self) -> jnp.ndarray:
        return self.embedding_layer.parameters.embeddings

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
                    negative_samples=self._negative_samples,
                    softmax_strategy=self._softmax_strategy,
                ),
                file_io,
            )

    @classmethod
    def load(
        cls,
        source: IO[bytes] | Path,
        training: bool = False,
        freeze_parameters: bool = False,
        key: jax.Array | None = None,
        negative_sampling_dist: jnp.ndarray | None = None,
    ) -> SkipGramModel:
        if key is None:
            key = jax.random.PRNGKey(0)

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
            key=key,
            negative_samples=serialized.negative_samples,
            negative_sampling_dist=negative_sampling_dist,
            softmax_strategy=getattr(
                serialized, "softmax_strategy", SoftmaxStrategy.NEGATIVE_SAMPLING
            ),
        )


class PredictModel(CBOWModel):
    @classmethod
    def get_name(cls) -> str:
        return "predict"

    @classmethod
    def get_description(cls) -> str:
        return "Autoregressive Prediction Model based on CBOW"

    @classmethod
    def from_cbow(
        cls, *, cbow_model: CBOWModel, context_width: int, key: jax.Array
    ) -> PredictModel:
        """Create a PredictModel from a trained CBOW model for autoregressive generation"""
        embedding_layer = cbow_model.embedding_layer
        vocab_size = embedding_layer.vocab_size
        embedding_dim = embedding_layer.output_dimensions[1]

        new_embedding_layer = Embedding(
            input_dimensions=(context_width,),
            output_dimensions=(context_width, embedding_dim),
            key=key,
            parameters=embedding_layer.parameters,
            store_output_activations=False,
            vocab_size=vocab_size,
        )

        return cls(
            input_dimensions=(context_width,),
            hidden=(
                Hidden(
                    layers=(
                        new_embedding_layer,
                        Average(
                            input_dimensions=(context_width, embedding_dim),
                            axis=0,
                        ),
                    )
                ),
            ),
            output=cbow_model.output,
        )
