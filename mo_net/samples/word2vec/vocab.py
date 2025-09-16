from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Counter, Self

import jax.numpy as jnp
import msgpack  # type: ignore[import-untyped]

from mo_net.resources import get_resource

type Sentence = Sequence[str]
type TokenizedSentence = Sequence[int]


@dataclass(slots=True, frozen=True)
class Vocab:
    vocab: tuple[str, ...]
    token_to_id: dict[str, int]
    id_to_token: dict[int, str]
    unknown_token_id: int

    def serialize(self) -> bytes:
        return msgpack.packb(
            {
                "vocab": list(self.vocab),
                "token_to_id": self.token_to_id,
                "unknown_token_id": self.unknown_token_id,
            }
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
        most_common_tokens = {
            token
            for token, _ in Counter(
                token for sentence in sentences if sentence for token in sentence
            ).most_common(max_size)
        }
        for word in forced_words:
            most_common_tokens.add(word)

        vocab_tuple = tuple(most_common_tokens)
        unknown_token_id = len(vocab_tuple)
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
        )


def get_training_set(
    tokenized_sentences: Collection[TokenizedSentence], context_size: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
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
    return jnp.array(context), jnp.array(list(target))


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
        [word for word in sentence.split() if word not in stop_words]
        for sentence in (
            get_resource("s3://mo-net-resources/english-sentences.txt")
            .read_text()
            .split("\n")[:limit]
        )
    ]
