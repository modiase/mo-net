"""
Huffman tree for hierarchical softmax in word2vec.

Builds a binary tree from word frequencies where frequent words have shorter paths.
Used to reduce softmax computation from O(V) to O(log V).
"""

from __future__ import annotations

import heapq
import msgpack
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass
class HuffmanNode:
    """A node in the Huffman tree.

    Attributes:
        frequency: Combined frequency of all words in subtree
        word_id: Word ID if this is a leaf node, None for internal nodes
        left: Left child node
        right: Right child node
    """

    frequency: int
    word_id: int | None = None
    left: HuffmanNode | None = None
    right: HuffmanNode | None = None

    def __lt__(self, other: HuffmanNode) -> bool:
        """Compare nodes by frequency for priority queue."""
        return self.frequency < other.frequency

    def is_leaf(self) -> bool:
        """Check if this is a leaf node (represents a word)."""
        return self.word_id is not None


class HuffmanTree:
    """Binary Huffman tree for hierarchical softmax.

    Organizes vocabulary words in a binary tree based on their frequencies.
    Frequent words get shorter paths from root, optimizing training speed.

    Attributes:
        root: Root node of the tree
        vocab_size: Number of words (leaf nodes)
        paths: Precomputed paths for each word
        codes: Precomputed binary codes for each word
    """

    def __init__(self, root: HuffmanNode, vocab_size: int):
        """Initialize tree with root node.

        Args:
            root: Root node of the Huffman tree
            vocab_size: Number of vocabulary words
        """
        self.root = root
        self.vocab_size = vocab_size
        self.num_internal_nodes = vocab_size - 1

        # Precompute paths and codes for all words
        self._paths: dict[int, list[int]] = {}
        self._codes: dict[int, list[bool]] = {}
        self._build_paths()

    @classmethod
    def build(cls, word_frequencies: Mapping[int, int]) -> HuffmanTree:
        """Build Huffman tree from word frequencies.

        Args:
            word_frequencies: Mapping from word_id to frequency count

        Returns:
            HuffmanTree with frequent words having shorter paths

        Example:
            >>> frequencies = {0: 100, 1: 50, 2: 25, 3: 10}
            >>> tree = HuffmanTree.build(frequencies)
            >>> # Word 0 (freq=100) will have shorter path than word 3 (freq=10)
        """
        if not word_frequencies:
            raise ValueError("Cannot build tree from empty frequencies")

        # Create leaf nodes for each word
        heap: list[HuffmanNode] = []
        for word_id, freq in word_frequencies.items():
            node = HuffmanNode(frequency=freq, word_id=word_id)
            heapq.heappush(heap, node)

        vocab_size = len(word_frequencies)

        # Special case: single word vocabulary
        if vocab_size == 1:
            # For single word, just return the leaf as root
            # This means num_internal_nodes will be 1 (the root itself)
            leaf = heap[0]
            root = HuffmanNode(frequency=leaf.frequency, left=leaf)
            tree = cls.__new__(cls)
            tree.root = root
            tree.vocab_size = vocab_size
            tree.num_internal_nodes = 1  # One internal node (root) for single word
            tree._paths = {}
            tree._codes = {}
            tree._build_paths()
            return tree

        # Build tree bottom-up using priority queue
        while len(heap) > 1:
            # Pop two nodes with smallest frequencies
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)

            # Create parent node with combined frequency
            parent = HuffmanNode(
                frequency=left.frequency + right.frequency, left=left, right=right
            )

            # Push parent back to heap
            heapq.heappush(heap, parent)

        # Last node is the root
        root = heap[0]
        return cls(root=root, vocab_size=vocab_size)

    def _build_paths(self) -> None:
        """Build path lookup tables for all words.

        For each word, stores:
        - List of internal node indices from root to word
        - List of binary decisions (True=left, False=right)
        """

        def traverse(
            node: HuffmanNode | None,
            path: list[int],
            code: list[bool],
            node_index: int,
        ) -> int:
            """Recursively traverse tree and build paths.

            Args:
                node: Current node
                path: Current path (node indices)
                code: Current code (left/right decisions)
                node_index: Index to assign to next internal node

            Returns:
                Next available node index
            """
            if node is None:
                return node_index

            if node.is_leaf():
                # Found a word - save its path and code
                assert node.word_id is not None
                self._paths[node.word_id] = path.copy()
                self._codes[node.word_id] = code.copy()
                return node_index

            # Internal node - assign index and traverse children
            current_index = node_index
            next_index = node_index + 1

            # Traverse left (code = True)
            if node.left is not None:
                next_index = traverse(
                    node.left, path + [current_index], code + [True], next_index
                )

            # Traverse right (code = False)
            if node.right is not None:
                next_index = traverse(
                    node.right, path + [current_index], code + [False], next_index
                )

            return next_index

        # Start traversal from root with index 0
        traverse(self.root, [], [], 0)

    def get_path(self, word_id: int) -> tuple[list[int], list[bool]]:
        """Get path from root to word.

        Args:
            word_id: ID of the word

        Returns:
            Tuple of (node_indices, directions):
            - node_indices: List of internal node indices from root to word
            - directions: List of booleans (True=left, False=right)

        Example:
            >>> tree = HuffmanTree.build({0: 100, 1: 50})
            >>> nodes, dirs = tree.get_path(0)
            >>> # nodes might be [0], dirs might be [True]
            >>> # meaning: from root (node 0), go left to reach word 0
        """
        if word_id not in self._paths:
            raise KeyError(f"Word ID {word_id} not in tree")

        return self._paths[word_id], self._codes[word_id]

    def get_path_length(self, word_id: int) -> int:
        """Get length of path from root to word.

        Args:
            word_id: ID of the word

        Returns:
            Number of internal nodes in path
        """
        path, _ = self.get_path(word_id)
        return len(path)

    def get_average_path_length(self, word_frequencies: Mapping[int, int]) -> float:
        """Calculate average path length weighted by word frequencies.

        Args:
            word_frequencies: Mapping from word_id to frequency

        Returns:
            Average path length, weighted by frequency
        """
        total_path_cost = 0
        total_frequency = 0

        for word_id, freq in word_frequencies.items():
            path_length = self.get_path_length(word_id)
            total_path_cost += path_length * freq
            total_frequency += freq

        return total_path_cost / total_frequency if total_frequency > 0 else 0.0

    def serialize(self) -> bytes:
        """Serialize tree to bytes for saving.

        Returns:
            Msgpack-encoded tree structure
        """
        data = {
            "vocab_size": self.vocab_size,
            "paths": {str(k): v for k, v in self._paths.items()},
            "codes": {str(k): v for k, v in self._codes.items()},
        }
        return msgpack.packb(data, use_bin_type=True)

    @classmethod
    def deserialize(cls, data: bytes) -> HuffmanTree:
        """Deserialize tree from bytes.

        Args:
            data: Msgpack-encoded tree structure

        Returns:
            Reconstructed HuffmanTree

        Note:
            This recreates the tree structure from saved paths/codes.
            The actual tree structure isn't saved, only the path information needed for inference.
        """
        decoded = msgpack.unpackb(data, raw=False)
        vocab_size = decoded["vocab_size"]

        # Reconstruct a minimal tree (just needs to support get_path)
        # We don't need the full tree structure for inference
        root = HuffmanNode(frequency=0)  # Dummy root
        tree = cls.__new__(cls)
        tree.root = root
        tree.vocab_size = vocab_size
        tree.num_internal_nodes = vocab_size - 1

        # Restore paths and codes
        tree._paths = {int(k): v for k, v in decoded["paths"].items()}
        tree._codes = {int(k): v for k, v in decoded["codes"].items()}

        return tree
