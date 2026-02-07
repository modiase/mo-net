"""Tests for Huffman tree data structure."""

import pytest

from mo_net.data_structures.huffman_tree import HuffmanNode, HuffmanTree


class TestHuffmanNode:
    """Test HuffmanNode class."""

    def test_create_leaf_node(self):
        """Test creating a leaf node (word)."""
        node = HuffmanNode(frequency=100, word_id=5)
        assert node.frequency == 100
        assert node.word_id == 5
        assert node.left is None
        assert node.right is None
        assert node.is_leaf()

    def test_create_internal_node(self):
        """Test creating an internal node."""
        left = HuffmanNode(frequency=50, word_id=1)
        right = HuffmanNode(frequency=30, word_id=2)
        parent = HuffmanNode(frequency=80, left=left, right=right)

        assert parent.frequency == 80
        assert parent.word_id is None
        assert not parent.is_leaf()
        assert parent.left == left
        assert parent.right == right

    def test_node_comparison(self):
        """Test that nodes are compared by frequency."""
        node1 = HuffmanNode(frequency=100)
        node2 = HuffmanNode(frequency=50)
        node3 = HuffmanNode(frequency=150)

        assert node2 < node1
        assert node1 < node3
        assert not (node1 < node2)


class TestHuffmanTreeBuild:
    """Test building Huffman trees."""

    def test_build_simple_tree(self):
        """Test building tree from simple frequencies."""
        frequencies = {0: 100, 1: 50, 2: 25, 3: 10}
        tree = HuffmanTree.build(frequencies)

        assert tree.vocab_size == 4
        assert tree.num_internal_nodes == 3
        assert tree.root is not None

    def test_build_two_word_tree(self):
        """Test building tree with just two words."""
        frequencies = {0: 100, 1: 50}
        tree = HuffmanTree.build(frequencies)

        assert tree.vocab_size == 2
        assert tree.num_internal_nodes == 1

    def test_build_single_word_tree(self):
        """Test building tree with single word."""
        frequencies = {0: 100}
        tree = HuffmanTree.build(frequencies)

        assert tree.vocab_size == 1
        assert tree.num_internal_nodes == 1  # Root is one internal node
        # Single word should still be accessible
        path, code = tree.get_path(0)
        assert len(path) == 1  # One internal node (root) for single word

    def test_build_empty_raises_error(self):
        """Test that empty frequencies raises error."""
        with pytest.raises(ValueError, match="Cannot build tree from empty"):
            HuffmanTree.build({})

    def test_frequent_words_have_shorter_paths(self):
        """Test that more frequent words have shorter paths."""
        frequencies = {
            0: 1000,  # Very frequent
            1: 500,  # Frequent
            2: 100,  # Medium
            3: 10,  # Rare
            4: 1,  # Very rare
        }
        tree = HuffmanTree.build(frequencies)

        # Get path lengths
        path_0 = tree.get_path_length(0)
        path_1 = tree.get_path_length(1)
        path_2 = tree.get_path_length(2)
        path_3 = tree.get_path_length(3)
        path_4 = tree.get_path_length(4)

        # Most frequent word should have shortest or equal path
        assert path_0 <= path_1
        assert path_0 <= path_2
        assert path_0 <= path_3
        assert path_0 <= path_4

        # Rare words should have longer paths
        assert path_4 >= path_0
        assert path_3 >= path_1

    def test_equal_frequencies(self):
        """Test building tree with equal frequencies."""
        frequencies = {0: 50, 1: 50, 2: 50, 3: 50}
        tree = HuffmanTree.build(frequencies)

        # Tree should be built successfully
        assert tree.vocab_size == 4

        # All words should have paths
        for word_id in range(4):
            path, code = tree.get_path(word_id)
            assert len(path) == len(code)


class TestHuffmanTreePaths:
    """Test path extraction from Huffman tree."""

    def test_get_path_returns_valid_structure(self):
        """Test that get_path returns (nodes, directions) tuple."""
        frequencies = {0: 100, 1: 50, 2: 25}
        tree = HuffmanTree.build(frequencies)

        nodes, directions = tree.get_path(0)

        # Should return lists
        assert isinstance(nodes, list)
        assert isinstance(directions, list)

        # Should have same length
        assert len(nodes) == len(directions)

        # Directions should be booleans
        for direction in directions:
            assert isinstance(direction, bool)

    def test_get_path_for_all_words(self):
        """Test that all words have valid paths."""
        frequencies = {0: 100, 1: 50, 2: 25, 3: 10}
        tree = HuffmanTree.build(frequencies)

        for word_id in range(4):
            nodes, directions = tree.get_path(word_id)

            # Path should exist
            assert len(nodes) >= 0

            # Node indices should be valid
            for node_idx in nodes:
                assert 0 <= node_idx < tree.num_internal_nodes

    def test_get_path_invalid_word_raises_error(self):
        """Test that invalid word ID raises KeyError."""
        frequencies = {0: 100, 1: 50}
        tree = HuffmanTree.build(frequencies)

        with pytest.raises(KeyError, match="Word ID 999 not in tree"):
            tree.get_path(999)

    def test_paths_are_consistent(self):
        """Test that paths don't change between calls."""
        frequencies = {0: 100, 1: 50, 2: 25}
        tree = HuffmanTree.build(frequencies)

        # Get path twice
        path1, code1 = tree.get_path(0)
        path2, code2 = tree.get_path(0)

        # Should be identical
        assert path1 == path2
        assert code1 == code2


class TestHuffmanTreePathLength:
    """Test path length calculations."""

    def test_get_path_length(self):
        """Test getting path length for a word."""
        frequencies = {0: 100, 1: 50, 2: 25, 3: 10}
        tree = HuffmanTree.build(frequencies)

        for word_id in range(4):
            path_len = tree.get_path_length(word_id)
            nodes, _ = tree.get_path(word_id)

            # Path length should match node list length
            assert path_len == len(nodes)

    def test_average_path_length(self):
        """Test calculating average path length."""
        frequencies = {0: 100, 1: 50, 2: 25, 3: 10}
        tree = HuffmanTree.build(frequencies)

        avg_len = tree.get_average_path_length(frequencies)

        # Average should be positive and reasonable
        assert avg_len > 0
        assert avg_len < tree.vocab_size  # Can't be longer than vocab size

    def test_average_path_length_uniform(self):
        """Test average path length with uniform frequencies."""
        frequencies = {0: 100, 1: 100, 2: 100, 3: 100}
        tree = HuffmanTree.build(frequencies)

        avg_len = tree.get_average_path_length(frequencies)

        # With uniform frequencies, should be close to log2(n)
        import math

        expected_len = math.log2(4)
        assert abs(avg_len - expected_len) < 1.0  # Reasonable tolerance

    def test_average_path_length_skewed(self):
        """Test that skewed frequencies give lower average path length."""
        # Skewed distribution (Zipf-like)
        skewed = {0: 1000, 1: 500, 2: 250, 3: 125, 4: 62}
        # Uniform distribution
        uniform = {0: 287, 1: 287, 2: 287, 3: 287, 4: 287}

        tree_skewed = HuffmanTree.build(skewed)
        tree_uniform = HuffmanTree.build(uniform)

        avg_skewed = tree_skewed.get_average_path_length(skewed)
        avg_uniform = tree_uniform.get_average_path_length(uniform)

        # Skewed should have shorter average (frequent words dominate)
        assert avg_skewed <= avg_uniform


class TestHuffmanTreeSerialization:
    """Test tree serialization and deserialization."""

    def test_serialize_deserialize_roundtrip(self):
        """Test that serialize/deserialize preserves paths."""
        frequencies = {0: 100, 1: 50, 2: 25, 3: 10}
        tree = HuffmanTree.build(frequencies)

        # Serialize
        data = tree.serialize()
        assert isinstance(data, bytes)

        # Deserialize
        tree2 = HuffmanTree.deserialize(data)

        # Check that vocab size matches
        assert tree2.vocab_size == tree.vocab_size

        # Check that all paths match
        for word_id in range(tree.vocab_size):
            path1, code1 = tree.get_path(word_id)
            path2, code2 = tree2.get_path(word_id)

            assert path1 == path2
            assert code1 == code2

    def test_serialize_single_word_tree(self):
        """Test serializing tree with single word."""
        frequencies = {0: 100}
        tree = HuffmanTree.build(frequencies)

        data = tree.serialize()
        tree2 = HuffmanTree.deserialize(data)

        assert tree2.vocab_size == 1
        path, code = tree2.get_path(0)
        assert len(path) == len(code)

    def test_serialize_large_tree(self):
        """Test serializing tree with many words."""
        # Create tree with 100 words
        frequencies = {i: 1000 - i * 10 for i in range(100)}
        tree = HuffmanTree.build(frequencies)

        data = tree.serialize()
        tree2 = HuffmanTree.deserialize(data)

        assert tree2.vocab_size == 100

        # Spot check a few paths
        for word_id in [0, 50, 99]:
            path1, code1 = tree.get_path(word_id)
            path2, code2 = tree2.get_path(word_id)
            assert path1 == path2
            assert code1 == code2


class TestHuffmanTreeProperties:
    """Test tree properties and invariants."""

    def test_all_words_have_unique_paths(self):
        """Test that every word has a unique path."""
        frequencies = {0: 100, 1: 50, 2: 25, 3: 10}
        tree = HuffmanTree.build(frequencies)

        paths_set = set()
        for word_id in range(4):
            path, code = tree.get_path(word_id)
            path_tuple = (tuple(path), tuple(code))
            assert path_tuple not in paths_set  # Should be unique
            paths_set.add(path_tuple)

    def test_tree_has_correct_number_of_internal_nodes(self):
        """Test that tree has vocab_size - 1 internal nodes."""
        for vocab_size in [2, 5, 10, 20]:
            frequencies = {i: 100 - i for i in range(vocab_size)}
            tree = HuffmanTree.build(frequencies)

            assert tree.num_internal_nodes == vocab_size - 1

            # Verify by checking maximum node index in any path
            max_node_idx = -1
            for word_id in range(vocab_size):
                nodes, _ = tree.get_path(word_id)
                if nodes:  # Skip empty paths (single word case)
                    max_node_idx = max(max_node_idx, max(nodes))

            # Max node index should be < num_internal_nodes
            if max_node_idx >= 0:  # If there are any internal nodes
                assert max_node_idx < tree.num_internal_nodes
