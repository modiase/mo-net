from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from re import Pattern
from typing import Final, Self, cast

import jax.numpy as jnp
import msgpack  # type: ignore[import-untyped]


type Sentence = Sequence[str]
type TokenizedSentence = Sequence[int]

_NON_PRINTABLE: Final[Pattern] = re.compile(r"[^\x20-\x7E]")
_PUNCTUATION: Final[Pattern] = re.compile(r"[^\w\s]")


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
        """Sampling probabilities under the unigram^power distribution.

        Default ``power=0.75`` per Mikolov et al. (2013).
        """
        vocab_size = len(self)  # includes unknown token

        if self.word_counts is None:
            return jnp.ones(vocab_size) / vocab_size

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
        """Get token ID, returning ``unknown_token_id`` if not in vocab."""
        return self.token_to_id.get(token, self.unknown_token_id)

    @staticmethod
    def clean_token(token: str) -> str:
        """Strip non-printable characters and punctuation; lowercase.

        Prefer :func:`tokenize_line` for new code — applying this to a
        whitespace-split token glues neighbours together when punctuation
        sits between them with no space (``"animals,Chicago"`` becomes
        ``"animalschicago"``).
        """
        return re.sub(r"[^\w\s]|[^\x20-\x7E]", "", token).lower().strip()

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
