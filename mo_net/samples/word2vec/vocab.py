from __future__ import annotations

import hashlib
import random
import re
from collections import Counter, defaultdict
from collections.abc import Collection, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from re import Pattern
from typing import Final, Self, cast

import jax.numpy as jnp
import msgpack  # type: ignore[import-untyped]
import numpy as np
from loguru import logger

from mo_net.resources import get_resource, iter_text_rows


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


def _pairs_cache_key(
    corpus_url: str,
    limit: int | None,
    max_vocab_size: int,
    forced_words: Collection[str],
    context_size: int,
    subsample_t: float,
) -> str:
    h = hashlib.sha256()
    h.update(corpus_url.encode())
    h.update(f"|{limit}|{max_vocab_size}|{context_size}|{subsample_t}|".encode())
    h.update(",".join(sorted(forced_words)).encode())
    return h.hexdigest()[:16]


def _vocab_cache_key(
    corpus_url: str,
    limit: int | None,
    max_vocab_size: int,
    forced_words: Collection[str],
) -> str:
    """Vocab is invariant to context_size / subsample_t — give it its own key so
    hyperparameter tweaks on those axes don't redo the expensive vocab pass."""
    h = hashlib.sha256()
    h.update(corpus_url.encode())
    h.update(f"|{limit}|{max_vocab_size}|".encode())
    h.update(",".join(sorted(forced_words)).encode())
    return h.hexdigest()[:16]


def cached_english_training_set(
    *,
    limit: int | None = None,
    max_vocab_size: int = 1000,
    forced_words: Collection[str] = (),
    context_size: int,
    cache_dir: Path | None = None,
    subsample_t: float = 1e-5,
    corpus_url: str = ENGLISH_SENTENCES_URL,
) -> tuple[Vocab, np.ndarray, np.ndarray]:
    """Materialise (vocab, X, Y) for an English-text corpus, with cache.

    Caches X and Y as separate uncompressed ``.npy`` files so subsequent
    loads use ``mmap_mode='r'`` — the OS page cache is then shared between
    concurrent jobs reading the same file, and the trainer pages in only
    the rows it touches per batch.

    Read-only mounts are tolerated: cache reads are best-effort, write
    failures (``OSError``) silently fall back to no-op.

    The (X, Y) shape is the same for CBOW and SkipGram — the X/Y swap
    that distinguishes the two models happens at the call-site after
    this returns, so a single cache file serves both.

    ``corpus_url`` is the pair-cache key, independent of the byte-level
    resource cache.
    """
    x_path: Path | None = None
    y_path: Path | None = None
    vocab_path: Path | None = None
    if cache_dir is not None:
        key = _pairs_cache_key(
            corpus_url,
            limit,
            max_vocab_size,
            forced_words,
            context_size,
            subsample_t,
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
        corpus_url=corpus_url,
    )
    if subsample_t > 0:
        before = sum(len(s) for s in tokenized)
        tokenized = subsample_tokenized_sentences(tokenized, vocab, subsample_t)
        after = sum(len(s) for s in tokenized)
        logger.info(
            f"frequent-word subsampling (t={subsample_t}): "
            f"{before:,} → {after:,} tokens "
            f"({100 * (1 - after / max(before, 1)):.1f}% dropped)"
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


def build_streaming_training_set(
    *,
    corpus_url: str,
    limit: int | None = None,
    max_vocab_size: int = 1000,
    forced_words: Collection[str] = (),
    context_size: int,
    cache_dir: Path,
    subsample_t: float = 1e-5,
    subsample_seed: int = 42,
) -> tuple[Vocab, np.ndarray, np.ndarray]:
    """Three-pass streaming pair build for arbitrarily large corpora.

    Peak RAM ≈ vocab Counter + one row's tokens. Pair arrays land
    directly on memmap-backed ``.npy`` files; subsequent runs hit the
    same cache path as :func:`cached_english_training_set` so the
    in-memory and streaming paths interchange.

    Three passes are unavoidable: pass 1 needs the full token Counter
    before vocab can be frozen, pass 2 needs the final vocab to size
    the output arrays, pass 3 fills them. Each pass re-reads the same
    HF parquet shards / cached file — after pass 1 the OS page cache
    keeps them hot, so passes 2 and 3 are I/O-cheap.
    """
    stop_words = get_stop_words()

    key = _pairs_cache_key(
        corpus_url, limit, max_vocab_size, forced_words, context_size, subsample_t
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

    def _rows() -> Iterator[str]:
        for n, text in enumerate(iter_text_rows(corpus_url)):
            if limit is not None and n >= limit:
                break
            yield text

    vocab_only_key = _vocab_cache_key(corpus_url, limit, max_vocab_size, forced_words)
    vocab_only_path = cache_dir / f"w2v-vocab-{vocab_only_key}.msgpack"
    if vocab_only_path.exists():
        logger.info(f"vocab cache hit: {vocab_only_path}")
        vocab = Vocab.from_bytes(vocab_only_path.read_bytes())
    else:
        logger.info("streaming pass 1/3: building vocab")
        counter: Counter[str] = Counter()
        n_rows = 0
        for text in _rows():
            counter.update(t for t in tokenize_line(text) if t not in stop_words)
            n_rows += 1
            if n_rows % 100_000 == 0:
                logger.info(
                    f"  pass 1: {n_rows:,} rows, {len(counter):,} unique tokens"
                )
        logger.info(f"  pass 1 done: {n_rows:,} rows, {len(counter):,} unique tokens")

        most_common = {tok for tok, _ in counter.most_common(max_vocab_size)}
        for w in forced_words:
            most_common.add(w)
        vocab_tuple = tuple(most_common)
        word_counts = {tok: counter[tok] for tok in vocab_tuple}
        vocab = Vocab(
            vocab=vocab_tuple,
            token_to_id={t: i for i, t in enumerate(vocab_tuple)},
            id_to_token=defaultdict(
                lambda: "<unknown>",
                {i: t for i, t in enumerate(vocab_tuple)},
            ),
            unknown_token_id=len(vocab_tuple),
            word_counts=word_counts,
        )
        del counter
        cache_dir.mkdir(parents=True, exist_ok=True)
        vocab_only_path.write_bytes(vocab.serialize())
        logger.info(f"cached vocab to {vocab_only_path}")

    word_counts = vocab.word_counts or {}
    discard_prob: dict[int, float] = {}
    if subsample_t > 0:
        total = sum(word_counts.values())
        if total > 0:
            sqrt_t = subsample_t**0.5
            for tok, count in word_counts.items():
                freq = count / total
                p = 1.0 - sqrt_t / (freq**0.5)
                if p > 0:
                    discard_prob[vocab[tok]] = p

    def _tokenise_subsample(text: str, rng: random.Random) -> list[int]:
        ids = [vocab[t] for t in tokenize_line(text) if t not in stop_words]
        if not discard_prob:
            return ids
        return [tok for tok in ids if rng.random() >= discard_prob.get(tok, 0.0)]

    logger.info("streaming pass 2/3: counting training pairs")
    rng = random.Random(subsample_seed)
    n_pairs = 0
    for text in _rows():
        ids = _tokenise_subsample(text, rng)
        n_pairs += max(0, len(ids) - 2 * context_size)
    logger.info(f"  pass 2 done: {n_pairs:,} training pairs")

    logger.info("streaming pass 3/3: writing memmap arrays")
    cache_dir.mkdir(parents=True, exist_ok=True)
    X = np.lib.format.open_memmap(
        x_path, mode="w+", dtype=np.int32, shape=(n_pairs, 2 * context_size)
    )
    Y = np.lib.format.open_memmap(y_path, mode="w+", dtype=np.int32, shape=(n_pairs,))
    rng = random.Random(subsample_seed)
    pos = 0
    for text in _rows():
        ids = _tokenise_subsample(text, rng)
        n = len(ids) - 2 * context_size
        if n <= 0:
            continue
        s = np.asarray(ids, dtype=np.int32)
        centers = np.arange(context_size, context_size + n)
        for j in range(context_size):
            X[pos : pos + n, j] = s[centers - context_size + j]
            X[pos : pos + n, context_size + j] = s[centers + 1 + j]
        Y[pos : pos + n] = s[centers]
        pos += n
    X.flush()
    Y.flush()
    vocab_path.write_bytes(vocab.serialize())
    logger.info(f"  pass 3 done; cached at {x_path.parent}")

    return (
        vocab,
        np.load(x_path, mmap_mode="r"),
        np.load(y_path, mmap_mode="r"),
    )


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
