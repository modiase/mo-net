"""
Word2Vec evaluation utilities.

Provides intrinsic evaluation methods for word embeddings:
- Word analogies (semantic and syntactic)
- Nearest neighbors
- Word similarity correlation
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
from loguru import logger

if TYPE_CHECKING:
    from mo_net.samples.word2vec.__main__ import CBOWModel, SkipGramModel
    from mo_net.samples.word2vec.vocab import Vocab


@dataclass
class AnalogyExample:
    """Word analogy: a is to b as c is to ?"""

    a: str
    b: str
    c: str
    expected: str
    category: str = "semantic"


@dataclass
class AnalogyResult:
    """Result of analogy evaluation"""

    example: AnalogyExample
    predictions: list[str]
    rank: int | None
    correct: bool


def cosine_similarity(v1: jnp.ndarray, v2: jnp.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(jnp.dot(v1, v2) / (jnp.linalg.norm(v1) * jnp.linalg.norm(v2) + 1e-10))


def get_nearest_neighbors(
    word: str,
    model: CBOWModel | SkipGramModel,
    vocab: Vocab,
    k: int = 10,
    exclude_self: bool = True,
) -> list[tuple[str, float]]:
    """
    Find k nearest neighbors to a word based on cosine similarity.

    Args:
        word: The query word
        model: Trained word2vec model
        vocab: Vocabulary
        k: Number of neighbors to return
        exclude_self: Whether to exclude the query word itself

    Returns:
        List of (word, similarity) tuples, sorted by similarity descending
    """
    if word not in vocab.vocab:
        raise ValueError(f"Word '{word}' not in vocabulary")

    word_idx = vocab[word]
    word_vec = model.embeddings[word_idx]

    similarities = []
    for other_word in vocab.vocab:
        if exclude_self and other_word == word:
            continue
        other_idx = vocab[other_word]
        other_vec = model.embeddings[other_idx]
        sim = cosine_similarity(word_vec, other_vec)
        similarities.append((other_word, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:k]


def evaluate_analogy(
    example: AnalogyExample,
    model: CBOWModel | SkipGramModel,
    vocab: Vocab,
    top_k: int = 5,
) -> AnalogyResult:
    """
    Evaluate a single word analogy: a - b + c ≈ expected

    Args:
        example: Analogy example
        model: Trained word2vec model
        vocab: Vocabulary
        top_k: Number of top predictions to consider

    Returns:
        AnalogyResult with predictions and correctness
    """
    # Check all words are in vocabulary
    missing = [
        w
        for w in [example.a, example.b, example.c, example.expected]
        if w not in vocab.vocab
    ]
    if missing:
        logger.warning(f"Words not in vocabulary: {missing}")
        return AnalogyResult(example=example, predictions=[], rank=None, correct=False)

    # Compute target vector: a - b + c
    a_vec = model.embeddings[vocab[example.a]]
    b_vec = model.embeddings[vocab[example.b]]
    c_vec = model.embeddings[vocab[example.c]]
    target_vec = a_vec - b_vec + c_vec

    # Find most similar words (excluding a, b, c)
    exclude = {example.a, example.b, example.c}
    similarities = []
    for word in vocab.vocab:
        if word in exclude:
            continue
        word_vec = model.embeddings[vocab[word]]
        sim = cosine_similarity(target_vec, word_vec)
        similarities.append((word, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    predictions = [word for word, _ in similarities[:top_k]]

    # Check if expected word is in top k
    correct = example.expected in predictions
    rank = predictions.index(example.expected) + 1 if correct else None

    return AnalogyResult(
        example=example, predictions=predictions, rank=rank, correct=correct
    )


def evaluate_analogies(
    examples: Sequence[AnalogyExample],
    model: CBOWModel | SkipGramModel,
    vocab: Vocab,
    top_k: int = 5,
) -> dict[str, float]:
    """
    Evaluate multiple word analogies and compute accuracy.

    Args:
        examples: List of analogy examples
        model: Trained word2vec model
        vocab: Vocabulary
        top_k: Number of top predictions to consider

    Returns:
        Dictionary with accuracy metrics by category
    """
    results = [evaluate_analogy(ex, model, vocab, top_k) for ex in examples]

    # Compute overall accuracy
    total = len(results)
    correct = sum(1 for r in results if r.correct)
    overall_acc = correct / total if total > 0 else 0.0

    # Compute accuracy by category
    category_stats = {}
    for category in set(ex.category for ex in examples):
        cat_results = [r for r in results if r.example.category == category]
        cat_total = len(cat_results)
        cat_correct = sum(1 for r in cat_results if r.correct)
        category_stats[category] = cat_correct / cat_total if cat_total > 0 else 0.0

    return {
        "overall_accuracy": overall_acc,
        "total_examples": total,
        "correct_predictions": correct,
        **{f"{cat}_accuracy": acc for cat, acc in category_stats.items()},
    }


def get_default_analogies() -> list[AnalogyExample]:
    """
    Get a default set of word analogies for testing.

    Returns semantic and syntactic analogy examples.
    """
    analogies = [
        # Semantic analogies - capitals
        AnalogyExample("paris", "france", "london", "england", "capital"),
        AnalogyExample("paris", "france", "berlin", "germany", "capital"),
        AnalogyExample("paris", "france", "rome", "italy", "capital"),
        # Semantic analogies - gender
        AnalogyExample("man", "woman", "king", "queen", "gender"),
        AnalogyExample("man", "woman", "brother", "sister", "gender"),
        AnalogyExample("man", "woman", "father", "mother", "gender"),
        # Semantic analogies - currency
        AnalogyExample("dollar", "usa", "euro", "europe", "currency"),
        AnalogyExample("dollar", "usa", "yen", "japan", "currency"),
        # Syntactic analogies - plural
        AnalogyExample("car", "cars", "dog", "dogs", "plural"),
        AnalogyExample("cat", "cats", "bird", "birds", "plural"),
        # Syntactic analogies - tense
        AnalogyExample("walk", "walked", "talk", "talked", "tense"),
        AnalogyExample("run", "running", "swim", "swimming", "tense"),
        # Syntactic analogies - comparatives
        AnalogyExample("good", "better", "bad", "worse", "comparative"),
        AnalogyExample("big", "bigger", "small", "smaller", "comparative"),
    ]
    return analogies


def print_analogy_results(results: list[AnalogyResult], max_display: int = 10) -> None:
    """Print formatted analogy evaluation results."""
    logger.info("\n=== Word Analogy Evaluation ===")

    for i, result in enumerate(results[:max_display]):
        ex = result.example
        status = "✓" if result.correct else "✗"
        logger.info(
            f"{status} {ex.a} - {ex.b} + {ex.c} = {ex.expected} "
            f"(predicted: {result.predictions[0] if result.predictions else 'N/A'})"
        )
        if result.correct and result.rank:
            logger.info(f"   Rank: {result.rank}")

    if len(results) > max_display:
        logger.info(f"... and {len(results) - max_display} more")


def evaluate_model(
    model: CBOWModel | SkipGramModel,
    vocab: Vocab,
    analogies: Sequence[AnalogyExample] | None = None,
) -> dict[str, float]:
    """
    Comprehensive model evaluation.

    Args:
        model: Trained word2vec model
        vocab: Vocabulary
        analogies: Custom analogy examples (uses defaults if None)

    Returns:
        Dictionary of evaluation metrics
    """
    if analogies is None:
        analogies = get_default_analogies()

    # Evaluate analogies
    metrics = evaluate_analogies(analogies, model, vocab, top_k=5)

    # Sample nearest neighbors for common words
    sample_words = ["king", "queen", "good", "bad", "big", "small"]
    sample_words = [w for w in sample_words if w in vocab.vocab]

    logger.info("\n=== Sample Nearest Neighbors ===")
    for word in sample_words[:3]:
        neighbors = get_nearest_neighbors(word, model, vocab, k=5)
        neighbor_str = ", ".join([f"{w}({s:.3f})" for w, s in neighbors])
        logger.info(f"{word}: {neighbor_str}")

    return metrics
