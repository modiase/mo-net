"""Tests for output layers, including negative sampling."""

from typing import cast

import pytest
import jax
import jax.numpy as jnp

from mo_net.model.layer.output import (
    SparseCategoricalSoftmaxOutputLayer,
    SoftmaxOutputLayer,
    RawOutputLayer,
    MseOutputLayer,
)
from mo_net.protos import Activations


class TestSparseCategoricalSoftmaxOutputLayer:
    """Test SparseCategoricalSoftmaxOutputLayer."""

    def test_forward_prop(self):
        """Test forward propagation produces softmax probabilities."""
        layer = SparseCategoricalSoftmaxOutputLayer(input_dimensions=(10,))

        X = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]])
        output = layer.forward_prop(input_activations=Activations(X))

        # Output should be probabilities (sum to 1)
        assert jnp.isclose(jnp.sum(output), 1.0)

        # All probabilities should be in [0, 1]
        assert jnp.all(output >= 0)
        assert jnp.all(output <= 1)

    def test_backward_prop_without_negative_sampling(self):
        """Test backward propagation without negative sampling."""
        layer = SparseCategoricalSoftmaxOutputLayer(input_dimensions=(5,))

        X = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        Y_true = jnp.array([2])  # True class is index 2

        layer.forward_prop(input_activations=Activations(X))
        gradient = cast(jnp.ndarray, layer.backward_prop(Y_true=Y_true))

        # Gradient should have same shape as input
        assert gradient.shape == X.shape

        # Gradient at true class should be negative (prob - 1)
        assert gradient[0, 2] < 0


class TestNegativeSamplingBackwardProp:
    """Test backward propagation with negative sampling."""

    def test_backward_prop_with_negative_sampling_2d(self):
        """Test backward_prop_with_negative with 2D negative samples array."""
        layer = SparseCategoricalSoftmaxOutputLayer(input_dimensions=(10,))

        batch_size = 2
        num_negatives = 3

        X = jax.random.normal(jax.random.PRNGKey(0), (batch_size, 10))
        Y_true = jnp.array([2, 5])  # True labels
        Y_negative = jnp.array([[1, 3, 7], [0, 4, 8]])  # Negative samples

        layer.forward_prop(input_activations=Activations(X))
        gradient = cast(
            jnp.ndarray,
            layer.backward_prop_with_negative(Y_true=Y_true, Y_negative=Y_negative),
        )

        # Gradient should have same shape as input
        assert gradient.shape == X.shape

        # Gradient should be zero for non-sampled indices
        # Sample 0: true=2, negatives=[1,3,7] -> indices 0,4,5,6,8,9 should be zero
        assert gradient[0, 0] == 0.0
        assert gradient[0, 4] == 0.0
        assert gradient[0, 6] == 0.0

        # Gradient should be non-zero for sampled indices
        assert gradient[0, 2] != 0.0  # True sample
        assert gradient[0, 1] != 0.0  # Negative sample
        assert gradient[0, 3] != 0.0  # Negative sample

    def test_backward_prop_with_negative_sampling_1d(self):
        """Test backward_prop_with_negative with 1D negative samples array."""
        layer = SparseCategoricalSoftmaxOutputLayer(input_dimensions=(10,))

        batch_size = 2
        num_negatives = 3

        X = jax.random.normal(jax.random.PRNGKey(0), (batch_size, 10))
        Y_true = jnp.array([2, 5])
        # Flattened negative samples: [neg1, neg2, neg3, neg1, neg2, neg3]
        Y_negative = jnp.array([1, 3, 7, 0, 4, 8])

        layer.forward_prop(input_activations=Activations(X))
        gradient = cast(
            jnp.ndarray,
            layer.backward_prop_with_negative(Y_true=Y_true, Y_negative=Y_negative),
        )

        # Gradient should have same shape as input
        assert gradient.shape == X.shape

        # Gradient should be zero for non-sampled indices
        assert gradient[0, 0] == 0.0
        assert gradient[0, 6] == 0.0

        # Gradient should be non-zero for sampled indices
        assert gradient[0, 2] != 0.0  # True sample
        assert gradient[0, 1] != 0.0  # Negative sample

    def test_gradient_sparsity(self):
        """Test that gradients are sparse (mostly zeros)."""
        vocab_size = 100
        batch_size = 4
        num_negatives = 5

        layer = SparseCategoricalSoftmaxOutputLayer(input_dimensions=(vocab_size,))

        X = jax.random.normal(jax.random.PRNGKey(0), (batch_size, vocab_size))
        Y_true = jnp.array([10, 20, 30, 40])
        Y_negative = jnp.array(
            [
                [1, 2, 3, 4, 5],
                [11, 12, 13, 14, 15],
                [21, 22, 23, 24, 25],
                [31, 32, 33, 34, 35],
            ]
        )

        layer.forward_prop(input_activations=Activations(X))
        gradient = cast(
            jnp.ndarray,
            layer.backward_prop_with_negative(Y_true=Y_true, Y_negative=Y_negative),
        )

        # Count non-zero elements per sample
        for i in range(batch_size):
            non_zero = jnp.sum(gradient[i] != 0)
            # Should have at most 1 (true) + num_negatives non-zero elements
            assert non_zero <= (1 + num_negatives)

        # Most of the gradient should be zero
        total_elements = gradient.size
        non_zero_total = jnp.sum(gradient != 0)
        sparsity = non_zero_total / total_elements

        # Should be very sparse (< 10% non-zero for this test)
        assert sparsity < 0.1

    def test_true_sample_gradient_sign(self):
        """Test that true sample gets negative gradient (prob - 1)."""
        layer = SparseCategoricalSoftmaxOutputLayer(input_dimensions=(5,))

        X = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        Y_true = jnp.array([2])
        Y_negative = jnp.array([[0, 1]])

        layer.forward_prop(input_activations=Activations(X))
        gradient = cast(
            jnp.ndarray,
            layer.backward_prop_with_negative(Y_true=Y_true, Y_negative=Y_negative),
        )

        # True sample should have negative gradient (probability - 1)
        # Since probability is in [0, 1], prob - 1 is in [-1, 0]
        assert gradient[0, 2] < 0
        assert gradient[0, 2] >= -1.0

    def test_negative_sample_gradient_sign(self):
        """Test that negative samples get positive gradient (probability)."""
        layer = SparseCategoricalSoftmaxOutputLayer(input_dimensions=(5,))

        X = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        Y_true = jnp.array([4])  # True label
        Y_negative = jnp.array([[1, 2]])  # Negative samples

        layer.forward_prop(input_activations=Activations(X))
        gradient = cast(
            jnp.ndarray,
            layer.backward_prop_with_negative(Y_true=Y_true, Y_negative=Y_negative),
        )

        # Negative samples should have positive gradient (probability)
        assert gradient[0, 1] > 0
        assert gradient[0, 2] > 0

        # Non-sampled items should be zero
        assert gradient[0, 0] == 0.0
        assert gradient[0, 3] == 0.0

    def test_different_negative_counts(self):
        """Test with different numbers of negative samples."""
        layer = SparseCategoricalSoftmaxOutputLayer(input_dimensions=(20,))

        X = jax.random.normal(jax.random.PRNGKey(0), (2, 20))
        Y_true = jnp.array([5, 10])

        for num_neg in [1, 3, 5, 10]:
            Y_negative = jnp.arange(num_neg * 2).reshape(2, num_neg)

            layer.forward_prop(input_activations=Activations(X))
            gradient = cast(
                jnp.ndarray,
                layer.backward_prop_with_negative(Y_true=Y_true, Y_negative=Y_negative),
            )

            # Check sparsity
            for i in range(2):
                non_zero = jnp.sum(gradient[i] != 0)
                assert non_zero <= (1 + num_neg)

    def test_batch_processing_with_negative_sampling(self):
        """Test that batch processing works correctly with negative sampling."""
        layer = SparseCategoricalSoftmaxOutputLayer(input_dimensions=(10,))

        batch_sizes = [1, 2, 4, 8]

        for batch_size in batch_sizes:
            X = jax.random.normal(jax.random.PRNGKey(0), (batch_size, 10))
            Y_true = jnp.arange(batch_size) % 10
            Y_negative = jnp.array(
                [[i % 10 for i in range(3)] for _ in range(batch_size)]
            )

            layer.forward_prop(input_activations=Activations(X))
            gradient = cast(
                jnp.ndarray,
                layer.backward_prop_with_negative(Y_true=Y_true, Y_negative=Y_negative),
            )

            assert gradient.shape == (batch_size, 10)


class TestNegativeSamplingEdgeCases:
    """Test edge cases in negative sampling."""

    def test_negative_sampling_with_duplicates(self):
        """Test handling when negative samples contain duplicates."""
        layer = SparseCategoricalSoftmaxOutputLayer(input_dimensions=(10,))

        X = jax.random.normal(jax.random.PRNGKey(0), (1, 10))
        Y_true = jnp.array([5])
        # Duplicate negative samples
        Y_negative = jnp.array([[2, 2, 2]])

        layer.forward_prop(input_activations=Activations(X))
        gradient = cast(
            jnp.ndarray,
            layer.backward_prop_with_negative(Y_true=Y_true, Y_negative=Y_negative),
        )

        # Should handle duplicates without error
        assert gradient.shape == (1, 10)

    def test_negative_sampling_overlaps_with_true(self):
        """Test when negative samples include the true label."""
        layer = SparseCategoricalSoftmaxOutputLayer(input_dimensions=(10,))

        X = jax.random.normal(jax.random.PRNGKey(0), (1, 10))
        Y_true = jnp.array([5])
        # Include true label in negatives
        Y_negative = jnp.array([[5, 3, 7]])

        layer.forward_prop(input_activations=Activations(X))
        gradient = cast(
            jnp.ndarray,
            layer.backward_prop_with_negative(Y_true=Y_true, Y_negative=Y_negative),
        )

        # Should handle overlap (later assignment overwrites)
        assert gradient.shape == (1, 10)

    def test_invalid_negative_shape(self):
        """Test that invalid negative sample shapes raise errors."""
        layer = SparseCategoricalSoftmaxOutputLayer(input_dimensions=(10,))

        X = jax.random.normal(jax.random.PRNGKey(0), (2, 10))
        Y_true = jnp.array([2, 5])

        # 3D array should raise error
        Y_negative_3d = jnp.array([[[1, 2], [3, 4]]])

        layer.forward_prop(input_activations=Activations(X))

        with pytest.raises(ValueError):
            layer.backward_prop_with_negative(Y_true=Y_true, Y_negative=Y_negative_3d)

    def test_mismatched_batch_size(self):
        """Test error when negative samples don't match batch size."""
        layer = SparseCategoricalSoftmaxOutputLayer(input_dimensions=(10,))

        X = jax.random.normal(jax.random.PRNGKey(0), (2, 10))
        Y_true = jnp.array([2, 5])

        # Wrong number of negatives (not divisible by batch size)
        Y_negative = jnp.array([1, 2, 3, 4, 5])  # 5 is not divisible by 2

        layer.forward_prop(input_activations=Activations(X))

        with pytest.raises(ValueError):
            layer.backward_prop_with_negative(Y_true=Y_true, Y_negative=Y_negative)


class TestOtherOutputLayers:
    """Test other output layer types for completeness."""

    def test_softmax_output_layer(self):
        """Test basic SoftmaxOutputLayer."""
        layer = SoftmaxOutputLayer(input_dimensions=(5,))

        X = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        output = layer.forward_prop(input_activations=Activations(X))

        # Should produce probabilities
        assert jnp.isclose(jnp.sum(output), 1.0)

    def test_raw_output_layer(self):
        """Test RawOutputLayer (no softmax in forward)."""
        layer = RawOutputLayer(input_dimensions=(5,))

        X = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        output = layer.forward_prop(input_activations=Activations(X))

        # Should return raw values (no softmax)
        assert jnp.allclose(output, X)

    def test_mse_output_layer(self):
        """Test MSE output layer."""
        layer = MseOutputLayer(input_dimensions=(3,), training=True)

        X = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        Y_true = jnp.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]])

        output = layer.forward_prop(input_activations=Activations(X))
        gradient = cast(jnp.ndarray, layer.backward_prop(Y_true=Y_true))

        # Gradient should be proportional to (output - Y_true)
        assert gradient.shape == X.shape


class TestSerialization:
    """Test output layer serialization."""

    def test_sparse_categorical_serialization(self):
        """Test SparseCategoricalSoftmaxOutputLayer serialization."""
        layer = SparseCategoricalSoftmaxOutputLayer(input_dimensions=(10,))

        serialized = layer.serialize()
        deserialized = serialized.deserialize()

        assert deserialized.output_dimensions == layer.output_dimensions

    def test_softmax_serialization(self):
        """Test SoftmaxOutputLayer serialization."""
        layer = SoftmaxOutputLayer(input_dimensions=(5,))

        serialized = layer.serialize()
        deserialized = serialized.deserialize()

        assert deserialized.output_dimensions == layer.output_dimensions

    def test_mse_serialization(self):
        """Test MseOutputLayer serialization."""
        layer = MseOutputLayer(input_dimensions=(3,), training=True)

        serialized = layer.serialize()
        deserialized = serialized.deserialize(training=True)

        assert deserialized.output_dimensions == layer.output_dimensions
