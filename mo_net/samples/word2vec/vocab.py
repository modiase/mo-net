from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Self, cast

import jax.numpy as jnp
import msgpack  # type: ignore[import-untyped]
import numpy as np
from loguru import logger

from mo_net.resources import get_resource

ENGLISH_SENTENCES_URL = "s3://mo-net-resources/english-sentences.txt"

type Sentence = Sequence[str]
type TokenizedSentence = Sequence[int]


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
        Remove non-printable characters and punctuation
        """
        return re.sub(r"[^\w\s]|[^\x20-\x7E]", "", token).lower().strip()

    @classmethod
    def english_sentences(
        cls,
        *,
        limit: int = 100000,
        max_vocab_size: int = 1000,
        forced_words: Collection[str] = (),
    ) -> tuple[Vocab, Collection[TokenizedSentence]]:
        return cls.from_sentences(
            get_english_sentences(limit),
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


def get_training_set(
    tokenized_sentences: Collection[TokenizedSentence], context_size: int
) -> tuple[np.ndarray, np.ndarray]:
    context, target = zip(
        *[
            (
                tuple(
                    chain(
                        sentence[i - context_size : i],
                        sentence[i + 1 : i + context_size + 1],
                    )
                ),
                sentence[i],
            )
            for sentence in tokenized_sentences
            for i in range(context_size, len(sentence) - context_size)
        ],
        strict=True,
    )
    return np.array(context), np.array(list(target))


def _pairs_cache_key(
    corpus_path: Path,
    limit: int,
    max_vocab_size: int,
    forced_words: Collection[str],
    context_size: int,
) -> str:
    h = hashlib.sha256()
    # corpus_path.name is the etag-prefixed cache filename — captures corpus identity
    h.update(corpus_path.name.encode())
    h.update(f"|{limit}|{max_vocab_size}|{context_size}|".encode())
    h.update(",".join(sorted(forced_words)).encode())
    return h.hexdigest()[:16]


def cached_english_training_set(
    *,
    limit: int = 100000,
    max_vocab_size: int = 1000,
    forced_words: Collection[str] = (),
    context_size: int,
    cache_dir: Path | None = None,
) -> tuple[Vocab, np.ndarray, np.ndarray]:
    """Materialise (vocab, X, Y) for the English-sentences corpus, with cache.

    Caches X and Y as separate uncompressed ``.npy`` files so subsequent
    loads use ``mmap_mode='r'`` — the OS page cache is then shared between
    concurrent jobs reading the same file, and the trainer pages in only
    the rows it touches per batch.

    Read-only mounts are tolerated: cache reads are best-effort, write
    failures (``OSError``) silently fall back to no-op.

    The (X, Y) shape is the same for CBOW and SkipGram — the X/Y swap
    that distinguishes the two models happens at the call-site after
    this returns, so a single cache file serves both.
    """
    corpus_path = get_resource(ENGLISH_SENTENCES_URL)

    x_path: Path | None = None
    y_path: Path | None = None
    vocab_path: Path | None = None
    if cache_dir is not None:
        key = _pairs_cache_key(
            corpus_path, limit, max_vocab_size, forced_words, context_size
        )
        x_path = cache_dir / f"w2v-pairs-{key}.X.npy"
        y_path = cache_dir / f"w2v-pairs-{key}.Y.npy"
        vocab_path = cache_dir / f"w2v-pairs-{key}.vocab.msgpack"
        if x_path.exists() and y_path.exists() and vocab_path.exists():
            logger.info(f"mmap-loading cached training set from {x_path.parent}")
            return (
                Vocab.from_bytes(vocab_path.read_bytes()),
                np.load(x_path, mmap_mode="r"),
                np.load(y_path, mmap_mode="r"),
            )

    vocab, tokenized = Vocab.english_sentences(
        limit=limit,
        max_vocab_size=max_vocab_size,
        forced_words=forced_words,
    )
    X, Y = get_training_set(tokenized, context_size)

    if x_path is not None and y_path is not None and vocab_path is not None:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
            np.save(x_path, np.asarray(X))
            np.save(y_path, np.asarray(Y))
            vocab_path.write_bytes(vocab.serialize())
            logger.info(f"cached training set to {x_path.parent}")
        except OSError as e:
            logger.debug(f"could not write training-set cache to {x_path}: {e}")

    return vocab, np.asarray(X), np.asarray(Y)


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


def get_english_sentences(limit: int = 100000) -> Collection[Sentence]:
    stop_words = get_stop_words()
    return [
        [
            cleaned
            for word in sentence.split()
            if (cleaned := Vocab.clean_token(word)) and cleaned not in stop_words
        ]
        for sentence in (
            get_resource("s3://mo-net-resources/english-sentences.txt")
            .read_text()
            .split("\n")[:limit]
        )
    ]
