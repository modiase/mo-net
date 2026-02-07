from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Self, TypedDict, TypeVar, cast

import jax
import jax.numpy as jnp
from pyparsing import abstractmethod

from mo_net.constants import EPSILON
from mo_net.model.layer.base import _Base
from mo_net.protos import (
    Activations,
    D,
    Dimensions,
    GradLayer,
    SupportsDeserialize,
    SupportsSerialize,
    d,
)

OutputLayerT_co = TypeVar("OutputLayerT_co", bound="OutputLayer", covariant=True)


class OutputLayer(_Base, SupportsSerialize[OutputLayerT_co]):
    class Cache(TypedDict):
        output_activations: Activations | None

    Serialized: type[SupportsDeserialize[OutputLayerT_co]]

    def __init__(
        self,
        *,
        input_dimensions: Dimensions,
    ):
        super().__init__(
            input_dimensions=input_dimensions,
            output_dimensions=input_dimensions,
        )
        self._cache: OutputLayer.Cache = {
            "output_activations": None,
        }

    def backward_prop(self, *, Y_true: jnp.ndarray) -> D[Activations]:
        return self._backward_prop(Y_true=Y_true)

    @abstractmethod
    def _backward_prop(
        self,
        *,
        Y_true: jnp.ndarray,
    ) -> D[Activations]: ...

    @abstractmethod
    def serialize(self) -> SupportsDeserialize: ...


class SoftmaxOutputLayer(OutputLayer):
    @dataclass(frozen=True, kw_only=True)
    class Serialized:
        input_dimensions: tuple[int, ...]

        def deserialize(
            self,
            *,
            training: bool = False,
            freeze_parameters: bool = False,
        ) -> SoftmaxOutputLayer:
            del training, freeze_parameters  # unused
            return SoftmaxOutputLayer(
                input_dimensions=self.input_dimensions,
            )

    def _forward_prop(
        self,
        *,
        input_activations: Activations,
    ) -> Activations:
        self._cache["output_activations"] = (
            output_activations := Activations(jax.nn.softmax(input_activations))
        )
        return output_activations

    def _backward_prop(
        self,
        *,
        Y_true: jnp.ndarray,
    ) -> D[Activations]:
        if (output_activations := self._cache["output_activations"]) is None:
            raise ValueError("Output activations not set during forward pass.")
        return cast(D[Activations], jnp.atleast_1d(output_activations - Y_true))

    @property
    def output_dimensions(self) -> Dimensions:
        return self._input_dimensions

    def serialize(self) -> SoftmaxOutputLayer.Serialized:
        return self.Serialized(input_dimensions=tuple(self._input_dimensions))


class RawOutputLayer(SoftmaxOutputLayer):
    """
    Uses softmax backprop but does not apply softmax to the output.
    """

    @dataclass(frozen=True, kw_only=True)
    class Serialized:  # type: ignore[misc]  # Intentional override of parent's Serialized class
        input_dimensions: tuple[int, ...]

        def deserialize(
            self,
            *,
            training: bool = False,
            freeze_parameters: bool = False,
        ) -> RawOutputLayer:
            del training, freeze_parameters  # unused
            return RawOutputLayer(input_dimensions=self.input_dimensions)

    def _forward_prop(
        self,
        *,
        input_activations: Activations,
    ) -> Activations:
        self._cache["output_activations"] = input_activations
        return input_activations


class MseOutputLayer(OutputLayer):
    @dataclass(frozen=True, kw_only=True)
    class Serialized:
        input_dimensions: tuple[int, ...]

        def deserialize(
            self,
            *,
            training: bool = False,
            freeze_parameters: bool = False,
        ) -> MseOutputLayer:
            del freeze_parameters  # unused
            return MseOutputLayer(
                input_dimensions=self.input_dimensions,
                training=training,
            )

    def __init__(self, *, input_dimensions: Dimensions, training: bool = True):
        super().__init__(input_dimensions=input_dimensions)
        self._training = training

    def _forward_prop(
        self,
        *,
        input_activations: Activations,
    ) -> Activations:
        if self._training:
            self._cache["output_activations"] = input_activations
        return input_activations

    def _backward_prop(
        self,
        *,
        Y_true: jnp.ndarray,
    ) -> D[Activations]:
        if (output_activations := self._cache["output_activations"]) is None:
            raise ValueError("Output activations not set during forward pass.")

        batch_size = output_activations.shape[0]
        return cast(D[Activations], 2.0 * (output_activations - Y_true) / batch_size)

    @property
    def output_dimensions(self) -> Dimensions:
        return self._input_dimensions

    def serialize(self) -> MseOutputLayer.Serialized:
        return self.Serialized(input_dimensions=tuple(self._input_dimensions))


class SparseCategoricalSoftmaxOutputLayer(OutputLayer):
    @dataclass(frozen=True, kw_only=True)
    class Serialized:
        input_dimensions: tuple[int, ...]

        def deserialize(
            self,
            *,
            training: bool = False,
            freeze_parameters: bool = False,
        ) -> SparseCategoricalSoftmaxOutputLayer:
            del training, freeze_parameters  # unused
            return SparseCategoricalSoftmaxOutputLayer(
                input_dimensions=self.input_dimensions,
            )

    def _forward_prop(
        self,
        *,
        input_activations: Activations,
    ) -> Activations:
        self._cache["output_activations"] = (
            output_activations := Activations(jax.nn.softmax(input_activations))
        )
        return output_activations

    def backward_prop_with_negative(
        self,
        *,
        Y_true: jnp.ndarray,
        Y_negative: jnp.ndarray,
    ) -> D[Activations]:
        return self._backward_prop(Y_true=Y_true, Y_negative=Y_negative)

    def _backward_prop(
        self,
        *,
        Y_true: jnp.ndarray,
        Y_negative: jnp.ndarray | None = None,
    ) -> D[Activations]:
        if (output_activations := self._cache["output_activations"]) is None:
            raise ValueError("Output activations not set during forward pass.")

        if Y_negative is None:
            result = output_activations.copy()
            result = result.at[jnp.arange(Y_true.shape[0]), Y_true].add(-1.0)
        else:
            result = jnp.zeros_like(output_activations)
            result = result.at[jnp.arange(Y_true.shape[0]), Y_true].set(
                output_activations[jnp.arange(Y_true.shape[0]), Y_true] - 1.0
            )

            if Y_negative.ndim == 2:
                batch_size, num_neg = Y_negative.shape
                row_idx = jnp.repeat(jnp.arange(batch_size), num_neg)
                col_idx = Y_negative.reshape(-1)
                result = result.at[row_idx, col_idx].set(
                    output_activations[row_idx, col_idx]
                )
            elif Y_negative.ndim == 1:
                batch_size = result.shape[0]
                total = Y_negative.shape[0]
                if total == batch_size:
                    result = result.at[jnp.arange(batch_size), Y_negative].set(
                        output_activations[jnp.arange(batch_size), Y_negative]
                    )
                else:
                    if total % batch_size != 0:
                        raise ValueError(
                            "Y_negative length must be a multiple of batch size when flattened."
                        )
                    num_neg = total // batch_size
                    row_idx = jnp.repeat(jnp.arange(batch_size), num_neg)
                    result = result.at[row_idx, Y_negative].set(
                        output_activations[row_idx, Y_negative]
                    )
            else:
                raise ValueError("Y_negative must be 1D or 2D array of indices.")

        return cast(D[Activations], jnp.atleast_1d(result))

    @property
    def output_dimensions(self) -> Dimensions:
        return self._input_dimensions

    def serialize(self) -> SparseCategoricalSoftmaxOutputLayer.Serialized:
        return self.Serialized(input_dimensions=tuple(self._input_dimensions))


class HierarchicalSoftmaxOutputLayer(OutputLayer):
    """Hierarchical softmax output layer using Huffman tree.

    Uses a binary tree structure to reduce softmax computation from O(V) to O(log V).
    Each internal node in the tree has a parameter vector that determines the probability
    of going left vs right.

    Note: This layer has learnable parameters (node vectors), unlike other output layers.
    The node vectors replace the weight matrix used in standard softmax.
    """

    @dataclass(frozen=True, kw_only=True)
    class Parameters:
        """Parameters for hierarchical softmax.

        Attributes:
            node_vectors: Weight vectors for internal nodes, shape (num_internal_nodes, embedding_dim)
        """

        node_vectors: jnp.ndarray

        def __add__(self, other: Self | float | int) -> Self:
            match other:
                case HierarchicalSoftmaxOutputLayer.Parameters():
                    return self.__class__(
                        node_vectors=self.node_vectors + other.node_vectors
                    )
                case float() | int():
                    return self.__class__(node_vectors=self.node_vectors + other)
                case _:
                    return NotImplemented

        def __radd__(self, other: Self | float | int) -> Self:
            return self.__add__(other)

        def __neg__(self) -> Self:
            return self.__class__(node_vectors=-self.node_vectors)

        def __sub__(self, other: Self | float | int) -> Self:
            return self.__add__(-other)

        def __mul__(self, other: float | int | Self) -> Self:
            match other:
                case float() | int():
                    return self.__class__(node_vectors=other * self.node_vectors)
                case HierarchicalSoftmaxOutputLayer.Parameters():
                    return self.__class__(
                        node_vectors=self.node_vectors * other.node_vectors
                    )
                case _:
                    return NotImplemented

        def __rmul__(self, other: float | Self) -> Self:
            return self.__mul__(other)

        def __truediv__(self, other: Self | float | int) -> Self:
            match other:
                case HierarchicalSoftmaxOutputLayer.Parameters():
                    return self.__class__(
                        node_vectors=self.node_vectors / (other.node_vectors + EPSILON)
                    )
                case float() | int():
                    return self.__mul__(1 / other)
                case _:
                    return NotImplemented

        def __pow__(self, scalar: float | int) -> Self:
            return self.__class__(node_vectors=self.node_vectors**scalar)

    class Cache(TypedDict):
        input_activations: Activations | None
        dP: "D[HierarchicalSoftmaxOutputLayer.Parameters] | None"
        first_moment: "D[HierarchicalSoftmaxOutputLayer.Parameters] | None"
        second_moment: "D[HierarchicalSoftmaxOutputLayer.Parameters] | None"

    @dataclass(frozen=True, kw_only=True)
    class Serialized:
        input_dimensions: tuple[int, ...]
        node_vectors: jnp.ndarray
        tree_data: bytes  # Serialized Huffman tree

        def deserialize(
            self,
            *,
            training: bool = False,
            freeze_parameters: bool = False,
        ) -> HierarchicalSoftmaxOutputLayer:
            from mo_net.data_structures.huffman_tree import HuffmanTree

            tree = HuffmanTree.deserialize(self.tree_data)
            layer = HierarchicalSoftmaxOutputLayer(
                input_dimensions=self.input_dimensions,
                huffman_tree=tree,
            )
            # Restore parameters
            layer._parameters = HierarchicalSoftmaxOutputLayer.Parameters(
                node_vectors=self.node_vectors
            )
            return layer

    def __init__(
        self,
        *,
        input_dimensions: Dimensions,
        vocab: "Vocab" | None = None,  # type: ignore
        huffman_tree: "HuffmanTree" | None = None,  # type: ignore
        key: jax.Array | None = None,
    ):
        """Initialize hierarchical softmax output layer.

        Args:
            input_dimensions: Dimensions of input (typically embedding_dim,)
            vocab: Vocabulary to build Huffman tree from (preferred - tree built automatically)
            huffman_tree: Pre-built Huffman tree (alternative to vocab, for deserialization)
            key: Random key for parameter initialization

        Note:
            Provide either `vocab` (preferred) or `huffman_tree` (for deserialization).
            When `vocab` is provided, the Huffman tree is built automatically from word frequencies.
        """
        from mo_net.data_structures.huffman_tree import HuffmanTree

        super().__init__(input_dimensions=input_dimensions)

        # Build or use provided tree
        if vocab is not None and huffman_tree is not None:
            raise ValueError("Provide either vocab or huffman_tree, not both")

        if vocab is not None:
            # Build Huffman tree from vocabulary word frequencies
            word_frequencies = {
                vocab[word]: count for word, count in vocab.word_counts.items()
            }
            # Include UNK token
            word_frequencies[vocab.unknown_token_id] = 1
            self.tree = HuffmanTree.build(word_frequencies)
        elif huffman_tree is not None:
            self.tree = huffman_tree
        else:
            raise ValueError("Must provide either vocab or huffman_tree")

        self.vocab_size = self.tree.vocab_size
        self.num_internal_nodes = self.tree.num_internal_nodes

        # Initialize node vectors (one per internal node)
        if key is None:
            key = jax.random.PRNGKey(0)

        embedding_dim = int(input_dimensions[0])

        # Xavier initialization for node vectors
        scale = jnp.sqrt(2.0 / (embedding_dim + self.num_internal_nodes))
        node_vectors = (
            jax.random.normal(key, (self.num_internal_nodes, embedding_dim)) * scale
        )

        self._parameters = self.Parameters(node_vectors=node_vectors)

        # Cache for activations and gradients (including Adam optimizer state)
        self._cache: HierarchicalSoftmaxOutputLayer.Cache = {
            "input_activations": None,
            "dP": None,
            "first_moment": d(
                self.Parameters(node_vectors=jnp.zeros_like(node_vectors))
            ),
            "second_moment": d(
                self.Parameters(node_vectors=jnp.zeros_like(node_vectors))
            ),
        }

    def _forward_prop(
        self,
        *,
        input_activations: Activations,
    ) -> Activations:
        """Compute forward pass - returns log probabilities.

        For each sample in batch, computes probability for ALL words by traversing
        all paths in the tree. This is expensive but needed for full softmax during inference.

        Args:
            input_activations: Context embeddings, shape (batch_size, embedding_dim)

        Returns:
            Log probabilities for each word, shape (batch_size, vocab_size)
        """
        # Cache input for backward pass
        self._cache["input_activations"] = input_activations

        batch_size = input_activations.shape[0]
        log_probs = jnp.zeros((batch_size, self.vocab_size))

        # For each word, compute log probability by traversing its path
        for word_id in range(self.vocab_size):
            node_indices, directions = self.tree.get_path(word_id)

            if len(node_indices) == 0:
                # Edge case: single word vocabulary
                log_probs = log_probs.at[:, word_id].set(0.0)
                continue

            # Compute log P(word|context) = sum of log P(decision_i) along path
            word_log_prob = jnp.zeros(batch_size)

            for node_idx, go_left in zip(node_indices, directions):
                # Compute score: θ_node^T · h
                scores = jnp.dot(
                    input_activations, self._parameters.node_vectors[node_idx]
                )

                # Log probability of this decision
                if go_left:
                    word_log_prob += jax.nn.log_sigmoid(scores)
                else:
                    word_log_prob += jax.nn.log_sigmoid(-scores)

            log_probs = log_probs.at[:, word_id].set(word_log_prob)

        # Return probabilities (not log probs) for compatibility
        return Activations(jax.nn.softmax(log_probs))

    def _backward_prop(
        self,
        *,
        Y_true: jnp.ndarray,
    ) -> D[Activations]:
        """Compute backward pass - sparse gradients only for paths to target words.

        Args:
            Y_true: Target word indices, shape (batch_size,)

        Returns:
            Gradient w.r.t. input activations, shape (batch_size, embedding_dim)
        """
        if (input_activations := self._cache["input_activations"]) is None:
            raise ValueError("Input activations not set during forward pass.")

        batch_size, embedding_dim = input_activations.shape

        # Initialize gradients
        grad_input = jnp.zeros_like(input_activations)
        grad_node_vectors = jnp.zeros_like(self._parameters.node_vectors)

        # Process each sample in batch
        for i in range(batch_size):
            target_word = int(Y_true[i])
            h = input_activations[i]  # Context embedding for this sample

            # Get path to target word
            node_indices, directions = self.tree.get_path(target_word)

            # For each node on the path, compute gradient
            for node_idx, go_left in zip(node_indices, directions):
                # Forward: score = θ_node^T · h
                theta = self._parameters.node_vectors[node_idx]
                score = jnp.dot(theta, h)

                # Sigmoid and gradient
                sig = jax.nn.sigmoid(score)

                # Target: 1 if go_left, 0 if go_right
                target = 1.0 if go_left else 0.0

                # Gradient of negative log likelihood (loss): (sigmoid - target)
                # This is for gradient descent: we want to minimize -log P(path)
                grad = sig - target

                # Gradient w.r.t. input: ∂loss/∂h = (σ - target) · θ
                grad_input = grad_input.at[i].add(grad * theta)

                # Gradient w.r.t. node vector: ∂loss/∂θ = (σ - target) · h
                grad_node_vectors = grad_node_vectors.at[node_idx].add(grad * h)

        # Store gradient in cache for optimizer
        self._cache["dP"] = d(self.Parameters(node_vectors=grad_node_vectors))

        return cast(D[Activations], grad_input)

    def empty_gradient(self) -> D[Parameters]:
        """Return zero gradients with same shape as parameters."""
        return d(
            self.Parameters(node_vectors=jnp.zeros_like(self._parameters.node_vectors))
        )

    def update_parameters(self) -> None:
        """Update parameters with optimizer-computed gradients from cache."""
        if (dP := self._cache["dP"]) is None:
            raise ValueError("Gradient not set during backward pass.")
        self._parameters = self._parameters + dP
        self._cache["dP"] = None

    def gradient_operation(self, f: Callable[[GradLayer], None]) -> None:
        """Apply a function to this layer for gradient operations."""
        f(self)

    @property
    def cache(self) -> Cache:
        """Return the cache containing gradients."""
        return self._cache

    @property
    def parameters(self) -> Parameters:
        """Return current parameters."""
        return self._parameters

    @property
    def output_dimensions(self) -> Dimensions:
        return self._input_dimensions

    def serialize(self) -> HierarchicalSoftmaxOutputLayer.Serialized:
        return self.Serialized(
            input_dimensions=tuple(self._input_dimensions),
            node_vectors=self._parameters.node_vectors,
            tree_data=self.tree.serialize(),
        )
