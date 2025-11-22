"""
Softmax strategy configuration for word2vec models.

Provides three approaches for computing output probabilities:
- FULL: Standard softmax over entire vocabulary (slow but simple)
- NEGATIVE_SAMPLING: Sample k negative examples (fast, good quality)
- HIERARCHICAL: Binary tree-based softmax (fast for large vocabularies)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING


class SoftmaxStrategy(Enum):
    """Strategy for computing output probabilities in word2vec models.

    FULL: Standard softmax over entire vocabulary
        - Complexity: O(V) per sample
        - Best for: Small vocabularies (<5K words)
        - Quality: Baseline

    NEGATIVE_SAMPLING: Sample k negative examples
        - Complexity: O(k) per sample, typically k=5-20
        - Best for: Medium to large vocabularies, quality-focused training
        - Quality: High, original word2vec approach

    HIERARCHICAL: Binary tree (Huffman) based softmax
        - Complexity: O(log V) per sample
        - Best for: Very large vocabularies (>50K words), speed-focused training
        - Quality: Good, frequent words optimized with shorter paths
    """

    FULL = "full"
    NEGATIVE_SAMPLING = "negative_sampling"
    HIERARCHICAL = "hierarchical"


@dataclass(frozen=True)
class SoftmaxConfig:
    """Configuration for softmax computation strategy.

    Attributes:
        strategy: Which softmax approach to use
        negative_samples: Number of negative samples (required for NEGATIVE_SAMPLING)

    Examples:
        >>> # Full softmax (simple but slow)
        >>> config = SoftmaxConfig(strategy=SoftmaxStrategy.FULL)

        >>> # Negative sampling with 5 samples
        >>> config = SoftmaxConfig(
        ...     strategy=SoftmaxStrategy.NEGATIVE_SAMPLING,
        ...     negative_samples=5
        ... )

        >>> # Hierarchical softmax (fast for large vocab)
        >>> config = SoftmaxConfig(strategy=SoftmaxStrategy.HIERARCHICAL)
    """

    strategy: SoftmaxStrategy
    negative_samples: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.strategy == SoftmaxStrategy.NEGATIVE_SAMPLING:
            if self.negative_samples is None:
                raise ValueError(
                    "negative_samples must be specified when using "
                    "SoftmaxStrategy.NEGATIVE_SAMPLING"
                )
            if self.negative_samples < 1:
                raise ValueError(
                    f"negative_samples must be >= 1, got {self.negative_samples}"
                )

        if self.strategy != SoftmaxStrategy.NEGATIVE_SAMPLING:
            if self.negative_samples is not None:
                raise ValueError(
                    f"negative_samples should only be specified for "
                    f"NEGATIVE_SAMPLING strategy, got strategy={self.strategy.value}"
                )

    @classmethod
    def full_softmax(cls) -> SoftmaxConfig:
        """Create config for full softmax (standard approach)."""
        return cls(strategy=SoftmaxStrategy.FULL)

    @classmethod
    def negative_sampling(cls, k: int = 5) -> SoftmaxConfig:
        """Create config for negative sampling.

        Args:
            k: Number of negative samples per positive sample (default: 5)
        """
        return cls(strategy=SoftmaxStrategy.NEGATIVE_SAMPLING, negative_samples=k)

    @classmethod
    def hierarchical_softmax(cls) -> SoftmaxConfig:
        """Create config for hierarchical (Huffman tree) softmax."""
        return cls(strategy=SoftmaxStrategy.HIERARCHICAL)
