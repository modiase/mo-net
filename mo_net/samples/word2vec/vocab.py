from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from re import Pattern
from typing import Final, Self, cast

import jax.numpy as jnp
import msgpack  # type: ignore[import-untyped]
import numpy as np

from mo_net.resources import get_resource


type Sentence = Sequence[str]
type TokenizedSentence = Sequence[int]

_NON_PRINTABLE: Final[Pattern] = re.compile(r"[^\x20-\x7E]")
_PUNCTUATION: Final[Pattern] = re.compile(r"[^\w\s]")

ENGLISH_SENTENCES_URL: Final[str] = "s3://mo-net-resources/english-sentences.txt"


def tokenize_line(line: str) -> list[str]:
    """Tokenise a sentence the way the trainer wants to see it.

    Treats any non-word, non-whitespace character as a token delimiter so
    ``"...animals,Chicago has..."`` yields ``["animals", "chicago", "has"]``
    rather than ``["animalschicago", "has"]``.
    """
    line = _NON_PRINTABLE.sub("", line)
    line = _PUNCTUATION.sub(" ", line)
    return line.lower().split()


@dataclass(slots=True, frozen=True)
class Vocab:
    vocab: tuple[str, ...]
    token_to_id: dict[str, int]
    id_to_token: dict[int, str]
    unknown_token_id: int
    word_counts: dict[str, int] | None = None

    def get_negative_sampling_distribution(self, power: float = 0.75) -> jnp.ndarray:
        """
        Compute sampling probabilities for negative sampling using unigram^power distribution.
        Default power=0.75 as per Mikolov et al. (2013).
        """
        vocab_size = len(self)  # Includes unknown token

        if self.word_counts is None:
            return jnp.ones(vocab_size) / vocab_size

        # Get frequencies for known words + unknown token
        freqs = jnp.array([self.word_counts.get(word, 1) for word in self.vocab] + [1])
        powered_freqs = jnp.power(freqs, power)
        return powered_freqs / jnp.sum(powered_freqs)

    def serialize(self) -> bytes:
        return cast(
            bytes,
            msgpack.packb(
                {
                    "vocab": list(self.vocab),
                    "token_to_id": self.token_to_id,
                    "unknown_token_id": self.unknown_token_id,
                    "word_counts": self.word_counts,
                }
            ),
        )

    @classmethod
    def deserialize(cls, path: Path) -> Vocab:
        with open(path, "rb") as f:
            data = msgpack.unpackb(f.read())
            return cls(
                vocab=tuple(data["vocab"]),
                token_to_id=data["token_to_id"],
                id_to_token=defaultdict(
                    lambda: "<unknown>",
                    {i: token for token, i in data["token_to_id"].items()},
                ),
                unknown_token_id=data.get("unknown_token_id", len(data["vocab"])),
                word_counts=data.get("word_counts"),
            )

    @classmethod
    def from_vocab(cls, vocab: Collection[str]) -> Vocab:
        vocab_tuple = tuple(vocab)
        return cls(
            vocab=vocab_tuple,
            token_to_id={token: i for i, token in enumerate(vocab_tuple)},
            id_to_token=defaultdict(
                lambda: "<unknown>",
                {i: token for i, token in enumerate(vocab_tuple)},
            ),
            unknown_token_id=len(vocab_tuple),
        )

    @classmethod
    def from_sentences(
        cls,
        sentences: Collection[Sentence],
        max_size: int,
        forced_words: Collection[str] = (),
    ) -> tuple[Vocab, Collection[TokenizedSentence]]:
        word_counter = Counter(
            token for sentence in sentences if sentence for token in sentence
        )
        most_common_tokens = {token for token, _ in word_counter.most_common(max_size)}
        for word in forced_words:
            most_common_tokens.add(word)

        vocab_tuple = tuple(most_common_tokens)
        unknown_token_id = len(vocab_tuple)
        word_counts = {token: word_counter[token] for token in vocab_tuple}

        return (
            (
                vocab := cls(
                    vocab=vocab_tuple,
                    token_to_id={token: i for i, token in enumerate(vocab_tuple)},
                    id_to_token=defaultdict(
                        lambda: "<unknown>",
                        {i: token for i, token in enumerate(vocab_tuple)},
                    ),
                    unknown_token_id=unknown_token_id,
                    word_counts=word_counts,
                )
            ),
            [
                [vocab[token] for token in sentence]
                for sentence in sentences
                if sentence
            ],
        )

    def __len__(self) -> int:
        return len(self.vocab) + 1

    def __getitem__(self, token: str) -> int:
        """Get token ID, returning unknown_token_id if token not in vocabulary"""
        return self.token_to_id.get(token, self.unknown_token_id)

    @staticmethod
    def clean_token(token: str) -> str:
        """
        Remove non-printable characters and punctuation.

        Note: prefer :func:`tokenize_line` for new code — applying this to a
        whitespace-split token glues neighbours together when punctuation
        sits between them with no space (e.g. ``"animals,Chicago"`` becomes
        ``"animalschicago"``).
        """
        return re.sub(r"[^\w\s]|[^\x20-\x7E]", "", token).lower().strip()

    @classmethod
    def english_sentences(
        cls,
        *,
        limit: int | None = None,
        max_vocab_size: int = 1000,
        forced_words: Collection[str] = (),
        corpus_url: str = ENGLISH_SENTENCES_URL,
    ) -> tuple[Vocab, Collection[TokenizedSentence]]:
        return cls.from_sentences(
            get_english_sentences(limit, url=corpus_url),
            max_size=max_vocab_size,
            forced_words=forced_words,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> Self:
        obj = msgpack.unpackb(data)
        return cls(
            vocab=tuple(obj["vocab"]),
            token_to_id=obj["token_to_id"],
            id_to_token=defaultdict(
                lambda: "<unknown>",
                {i: token for token, i in obj["token_to_id"].items()},
            ),
            unknown_token_id=obj.get("unknown_token_id", len(obj["vocab"])),
            word_counts=obj.get("word_counts"),
        )


def subsample_tokenized_sentences(
    tokenized: Collection[TokenizedSentence],
    vocab: Vocab,
    t: float,
    seed: int = 42,
) -> list[TokenizedSentence]:
    """Mikolov 2013 frequent-word subsampling.

    Each occurrence of an in-vocab word ``w`` is dropped with probability
    ``1 - sqrt(t / f(w))`` where ``f(w)`` is its corpus frequency. OOV
    tokens (the catch-all ``unknown_token_id``) are never dropped — they
    represent the long tail we already chose not to learn, and dropping
    them only shortens windows for no signal gain. ``t = 0`` disables.
    """
    if t <= 0 or vocab.word_counts is None:
        return list(tokenized)
    total = sum(vocab.word_counts.values())
    if total <= 0:
        return list(tokenized)
    sqrt_t = t**0.5
    discard_prob: dict[int, float] = {}
    for word, count in vocab.word_counts.items():
        if count <= 0:
            continue
        freq = count / total
        prob = 1.0 - sqrt_t / (freq**0.5)
        if prob > 0:
            discard_prob[vocab[word]] = prob
    rng = random.Random(seed)
    return [
        [tok for tok in sentence if rng.random() >= discard_prob.get(tok, 0.0)]
        for sentence in tokenized
    ]


def get_training_set(
    tokenized_sentences: Collection[TokenizedSentence], context_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, Y) training arrays from tokenised sentences.

    Two-pass: count windows, preallocate X/Y, then fill per-sentence via
    vectorised slicing. Peak memory ≈ output size; the previous
    list-of-tuples + ``zip(*…)`` + ``np.array(...)`` pipeline overshot
    by ~10× and OOM'd at FineWeb scale.
    """
    n_pairs = sum(max(0, len(s) - 2 * context_size) for s in tokenized_sentences)
    X = np.empty((n_pairs, 2 * context_size), dtype=np.int32)
    Y = np.empty(n_pairs, dtype=np.int32)
    pos = 0
    for sentence in tokenized_sentences:
        n = len(sentence) - 2 * context_size
        if n <= 0:
            continue
        s = np.asarray(sentence, dtype=np.int32)
        centers = np.arange(context_size, context_size + n)
        for j in range(context_size):
            X[pos : pos + n, j] = s[centers - context_size + j]
            X[pos : pos + n, context_size + j] = s[centers + 1 + j]
        Y[pos : pos + n] = s[centers]
        pos += n
    return X, Y


def get_stop_words() -> Collection[str]:
    return {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "by",
        "for",
        "from",
        "has",
        "he",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "that",
        "the",
        "to",
        "was",
        "will",
        "with",
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "ours",
        "ourselves",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "him",
        "his",
        "himself",
        "she",
        "her",
        "hers",
        "herself",
        "they",
        "them",
        "their",
        "theirs",
        "themselves",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "these",
        "those",
        "am",
        "were",
        "being",
        "have",
        "had",
        "having",
        "do",
        "does",
        "did",
        "doing",
        "would",
        "should",
        "could",
        "can",
        "may",
        "might",
        "must",
        "shall",
        "ought",
        "but",
        "if",
        "or",
        "because",
        "until",
        "while",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "up",
        "down",
        "out",
        "off",
        "over",
        "under",
        "again",
        "further",
        "the",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "s",
        "t",
        "just",
        "now",
        "also",
        "however",
        "although",
        "though",
        "since",
        "unless",
        "whether",
        "many",
    }


def get_english_sentences(
    limit: int | None = None,
    *,
    url: str = ENGLISH_SENTENCES_URL,
) -> Collection[Sentence]:
    """Tokenise the corpus at ``url``, dropping stop words, capped at ``limit`` rows.

    Each row's ``text`` column becomes one "sentence" — one line for
    ``.txt``, one document for HF datasets. :func:`get_training_set`
    slides a fixed window inside each, so longer documents are fine.
    """
    stop_words = get_stop_words()
    ds = get_resource(url)
    texts: list[str] = ds["text"]
    if limit is not None:
        texts = texts[:limit]
    return [
        [token for token in tokenize_line(text) if token not in stop_words]
        for text in texts
    ]
