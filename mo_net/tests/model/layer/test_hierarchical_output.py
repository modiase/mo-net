"""Tests for hierarchical softmax output layer."""

from typing import cast

import pytest
import jax
import jax.numpy as jnp

from mo_net.data_structures.huffman_tree import HuffmanTree
from mo_net.model.layer.output import HierarchicalSoftmaxOutputLayer
from mo_net.protos import Activations
from mo_net.samples.word2vec.vocab import Vocab


class TestHierarchicalSoftmaxOutputLayerCreation:
    """Test creating hierarchical softmax output layer."""

    def test_create_basic(self):
        """Test creating layer from vocabulary (automatic tree building)."""
        sentences = [["the", "quick", "brown", "fox"]]
        vocab, _ = Vocab.from_sentences(sentences, max_size=1000)

        layer = HierarchicalSoftmaxOutputLayer(
            input_dimensions=(16,),
            vocab=vocab,  # Tree built automatically!
            key=jax.random.PRNGKey(42),
        )

        assert layer.vocab_size == len(vocab)
        assert layer.num_internal_nodes == len(vocab) - 1
        assert layer.parameters.node_vectors.shape == (len(vocab) - 1, 16)

    def test_node_vectors_initialized(self):
        """Test that node vectors are properly initialized."""
        frequencies = {0: 100, 1: 50}
        tree = HuffmanTree.build(frequencies)

        layer = HierarchicalSoftmaxOutputLayer(
            input_dimensions=(8,), huffman_tree=tree, key=jax.random.PRNGKey(42)
        )

        # Should have parameters
        assert layer.parameters is not None
        assert layer.parameters.node_vectors.shape == (1, 8)

        # Should not be all zeros
        assert not jnp.allclose(layer.parameters.node_vectors, 0.0)


class TestHierarchicalSoftmaxForward:
    """Test forward propagation."""

    def test_forward_returns_probabilities(self):
        """Test that forward pass returns valid probabilities."""
        frequencies = {0: 100, 1: 50, 2: 25}
        tree = HuffmanTree.build(frequencies)

        layer = HierarchicalSoftmaxOutputLayer(
            input_dimensions=(8,), huffman_tree=tree, key=jax.random.PRNGKey(42)
        )

        # Create input activations
        h = Activations(jnp.ones((2, 8)))  # Batch of 2

        # Forward pass
        output = layer._forward_prop(input_activations=h)

        # Should return probabilities
        assert output.shape == (2, 3)  # (batch, vocab)

        # Probabilities should sum to ~1 (softmax output)
        assert jnp.allclose(jnp.sum(output, axis=1), 1.0, atol=1e-5)

        # All probabilities should be in [0, 1]
        assert jnp.all(output >= 0.0)
        assert jnp.all(output <= 1.0)

    def test_forward_different_inputs_different_outputs(self):
        """Test that different inputs produce different outputs."""
        frequencies = {0: 100, 1: 50}
        tree = HuffmanTree.build(frequencies)

        layer = HierarchicalSoftmaxOutputLayer(
            input_dimensions=(4,), huffman_tree=tree, key=jax.random.PRNGKey(42)
        )

        h1 = Activations(jnp.array([[1.0, 0.0, 0.0, 0.0]]))
        h2 = Activations(jnp.array([[0.0, 1.0, 0.0, 0.0]]))

        out1 = layer._forward_prop(input_activations=h1)
        out2 = layer._forward_prop(input_activations=h2)

        # Different inputs should produce different outputs
        assert not jnp.allclose(out1, out2)


class TestHierarchicalSoftmaxBackward:
    """Test backward propagation."""

    def test_backward_returns_gradient(self):
        """Test that backward pass returns valid gradient."""
        frequencies = {0: 100, 1: 50, 2: 25}
        tree = HuffmanTree.build(frequencies)

        layer = HierarchicalSoftmaxOutputLayer(
            input_dimensions=(8,), huffman_tree=tree, key=jax.random.PRNGKey(42)
        )

        # Forward pass
        h = Activations(jnp.ones((2, 8)))
        layer._forward_prop(input_activations=h)

        # Backward pass
        Y_true = jnp.array([0, 1])  # Target words
        grad = cast(jnp.ndarray, layer._backward_prop(Y_true=Y_true))

        # Gradient should have same shape as input
        assert grad.shape == h.shape

        # Gradient should not be all zeros
        assert not jnp.allclose(grad, 0.0)

    def test_backward_computes_node_gradients(self):
        """Test that backward pass computes gradients for node vectors."""
        frequencies = {0: 100, 1: 50}
        tree = HuffmanTree.build(frequencies)

        layer = HierarchicalSoftmaxOutputLayer(
            input_dimensions=(4,), huffman_tree=tree, key=jax.random.PRNGKey(42)
        )

        # Forward and backward
        h = Activations(jnp.ones((1, 4)))
        layer._forward_prop(input_activations=h)
        layer._backward_prop(Y_true=jnp.array([0]))

        # Should have node vector gradients in cache
        dP = cast(HierarchicalSoftmaxOutputLayer.Parameters, layer.cache["dP"])
        assert dP is not None
        assert dP.node_vectors.shape == layer.parameters.node_vectors.shape

    def test_gradient_sparsity(self):
        """Test that only nodes on path receive non-zero gradients."""
        # Create tree with multiple words
        frequencies = {0: 100, 1: 50, 2: 25, 3: 10, 4: 5}
        tree = HuffmanTree.build(frequencies)

        layer = HierarchicalSoftmaxOutputLayer(
            input_dimensions=(8,), huffman_tree=tree, key=jax.random.PRNGKey(42)
        )

        # Forward and backward for single target word
        h = Activations(jnp.ones((1, 8)))
        layer._forward_prop(input_activations=h)
        layer._backward_prop(Y_true=jnp.array([0]))

        # Get gradients from cache
        dP = cast(HierarchicalSoftmaxOutputLayer.Parameters, layer.cache["dP"])
        assert dP is not None
        node_grads = dP.node_vectors

        # Count non-zero gradient rows
        non_zero_rows = jnp.sum(jnp.any(node_grads != 0, axis=1))

        # Should be sparse (only nodes on path to word 0)
        path_length = tree.get_path_length(0)
        assert non_zero_rows == path_length


class TestHierarchicalSoftmaxParameterUpdate:
    """Test parameter updates."""

    def test_update_parameters(self):
        """Test updating parameters using gradient from cache."""
        frequencies = {0: 100, 1: 50}
        tree = HuffmanTree.build(frequencies)

        layer = HierarchicalSoftmaxOutputLayer(
            input_dimensions=(4,), huffman_tree=tree, key=jax.random.PRNGKey(42)
        )

        # Save original parameters
        original_params = layer.parameters.node_vectors.copy()

        # Do a forward and backward pass to generate gradients
        h = Activations(jnp.ones((1, 4)))
        layer._forward_prop(input_activations=h)
        layer._backward_prop(Y_true=jnp.array([0]))

        # Update parameters (applies dP from cache)
        layer.update_parameters()

        # Parameters should have changed
        assert not jnp.allclose(layer.parameters.node_vectors, original_params)


class TestHierarchicalSoftmaxSerialization:
    """Test serialization and deserialization."""

    def test_serialize_deserialize_roundtrip(self):
        """Test that serialize/deserialize preserves layer state."""
        frequencies = {0: 100, 1: 50, 2: 25}
        tree = HuffmanTree.build(frequencies)

        layer = HierarchicalSoftmaxOutputLayer(
            input_dimensions=(8,), huffman_tree=tree, key=jax.random.PRNGKey(42)
        )

        # Serialize
        serialized = layer.serialize()

        # Deserialize
        layer2 = serialized.deserialize()

        # Check properties match
        assert layer2.vocab_size == layer.vocab_size
        assert layer2.num_internal_nodes == layer.num_internal_nodes
        assert jnp.allclose(
            layer2.parameters.node_vectors, layer.parameters.node_vectors
        )

    def test_deserialized_layer_produces_same_output(self):
        """Test that deserialized layer produces identical output."""
        frequencies = {0: 100, 1: 50}
        tree = HuffmanTree.build(frequencies)

        layer = HierarchicalSoftmaxOutputLayer(
            input_dimensions=(4,), huffman_tree=tree, key=jax.random.PRNGKey(42)
        )

        # Get output from original
        h = Activations(jnp.ones((1, 4)))
        out1 = layer._forward_prop(input_activations=h)

        # Serialize and deserialize
        serialized = layer.serialize()
        layer2 = serialized.deserialize()

        # Get output from deserialized
        out2 = layer2._forward_prop(input_activations=h)

        # Should be identical
        assert jnp.allclose(out1, out2, atol=1e-6)


class TestHierarchicalSoftmaxEdgeCases:
    """Test edge cases."""

    def test_single_word_vocabulary(self):
        """Test with single word vocabulary."""
        frequencies = {0: 100}
        tree = HuffmanTree.build(frequencies)

        layer = HierarchicalSoftmaxOutputLayer(
            input_dimensions=(4,), huffman_tree=tree, key=jax.random.PRNGKey(42)
        )

        h = Activations(jnp.ones((1, 4)))
        output = layer._forward_prop(input_activations=h)

        # Should return probability 1.0 for the only word
        assert output.shape == (1, 1)
        assert jnp.allclose(output, 1.0, atol=1e-5)

    def test_batch_processing(self):
        """Test processing batch of samples."""
        frequencies = {0: 100, 1: 50, 2: 25}
        tree = HuffmanTree.build(frequencies)

        layer = HierarchicalSoftmaxOutputLayer(
            input_dimensions=(8,), huffman_tree=tree, key=jax.random.PRNGKey(42)
        )

        # Forward with batch
        batch_size = 5
        h = Activations(jax.random.normal(jax.random.PRNGKey(0), (batch_size, 8)))
        output = layer._forward_prop(input_activations=h)

        assert output.shape == (batch_size, 3)

        # Backward with batch
        Y_true = jnp.array([0, 1, 2, 0, 1])
        grad = cast(jnp.ndarray, layer._backward_prop(Y_true=Y_true))

        assert grad.shape == (batch_size, 8)
