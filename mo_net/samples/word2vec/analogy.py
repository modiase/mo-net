"""Pure-math helpers shared by the CLI and tests.

Windowing of tokenised sentences, cosine similarity, and the expression
parser used by ``calculate`` / analogy probes. No click, no trainer, no
archive — these are leaf utilities that can be imported anywhere without
dragging the CLI surface.
"""

from __future__ import annotations

from collections.abc import Collection, Iterator, Mapping, Sequence
from typing import cast

import jax.numpy as jnp
from more_itertools import windowed

from mo_net.samples.word2vec.models import CBOWModel, SkipGramModel
from mo_net.samples.word2vec.vocab import Vocab


def all_windows(
    sentences: Collection[Sequence[int]], window_size: int
) -> Iterator[Sequence[int]]:
    return cast(
        Iterator[Sequence[int]],
        (
            window
            for sentence in sentences
            for window in windowed(sentence, window_size)
            if all(item is not None for item in window)
        ),
    )


def compute_similarity(vector1: jnp.ndarray, vector2: jnp.ndarray) -> float:
    return float(
        jnp.dot(vector1, vector2)
        / (jnp.linalg.norm(vector1) * jnp.linalg.norm(vector2))
    )


def parse_and_calculate_expression(
    expr: str, model: CBOWModel | SkipGramModel, vocab: Vocab
) -> tuple[jnp.ndarray, str]:
    if "=" not in expr:
        raise ValueError("Expression must contain '=' to specify target word")

    left_side, target_word = expr.split("=", 1)
    target_word = target_word.strip().lower()
    tokens = [token.lower() for token in left_side.strip().split()]
    if len(tokens) < 2:
        raise ValueError("Expression must have at least one word and one operation")

    if tokens[0] not in vocab.vocab:
        raise ValueError(f"Word '{tokens[0]}' not found in vocabulary")

    result_vector = model.embeddings[vocab[tokens[0]]]

    i = 1
    while i < len(tokens):
        if i + 1 >= len(tokens):
            raise ValueError("Operation must be followed by a word")

        op, word = tokens[i], tokens[i + 1]
        if word not in vocab.vocab:
            raise ValueError(f"Word '{word}' not found in vocabulary")

        word_vector = model.embeddings[vocab[word]]
        match op:
            case "+":
                result_vector = result_vector + word_vector
            case "-":
                result_vector = result_vector - word_vector
            case _:
                raise ValueError(
                    f"Unsupported operation: '{op}'. Only '+' and '-' are supported"
                )

        i += 2

    return result_vector, target_word


def find_most_similar_words(
    target_vector: jnp.ndarray,
    model: CBOWModel | SkipGramModel,
    vocab: Vocab,
) -> Mapping[str, tuple[int, float]]:
    similarities = [
        (word, compute_similarity(target_vector, model.embeddings[vocab[word]]))
        for word in vocab.vocab
    ]
    similarities.sort(key=lambda x: x[1], reverse=True)
    return {
        word: (i, similarity) for i, (word, similarity) in enumerate(similarities, 1)
    }
