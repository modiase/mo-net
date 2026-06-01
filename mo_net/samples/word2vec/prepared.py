"""Tokenised-corpus artifact + training-time derivation.

A prepared dataset (produced by ``mo-net-build-w2v-dataset``) is an
on-disk directory holding the corpus tokenised once against a *full*
vocab — no top-K filter, no subsampling, no windowing. The expensive
work (tokenisation) is done once; the cheap work (top-K vocab,
remap, subsample, windowing) is derived at training time and cached
inside the artifact at ``derived/<args_hash>/``.

Layout::

    <artifact>/
    ├── data/                  # HF Dataset.save_to_disk; column `tokens: list[int32]`
    ├── full_vocab.msgpack     # str → full_id (no top-K)
    ├── freq.npy               # int64[full_vocab_size]; freq[full_id]
    ├── meta.json              # prep provenance + content_hash
    └── derived/
        └── <args_hash>/
            ├── X.npy          # int32 (n_pairs, 2·ctx)
            ├── Y.npy          # int32 (n_pairs,)
            ├── vocab.msgpack  # trainer-side Vocab (top-K + forced)
            └── meta.json      # derivation provenance + n_pairs
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
from collections import defaultdict
from collections.abc import Collection
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

import msgpack  # type: ignore[import-untyped]
import numpy as np
from loguru import logger
from packaging.version import Version

from mo_net.resources import file_lock
from mo_net.samples.word2vec.vocab import Vocab

ARTIFACT_DATA_DIR: Final[str] = "data"
ARTIFACT_VOCAB_FILE: Final[str] = "full_vocab.msgpack"
ARTIFACT_FREQ_FILE: Final[str] = "freq.npy"
ARTIFACT_META_FILE: Final[str] = "meta.json"

DERIVED_DIR: Final[str] = "derived"
DERIVED_X_FILE: Final[str] = "X.npy"
DERIVED_Y_FILE: Final[str] = "Y.npy"
DERIVED_VOCAB_FILE: Final[str] = "vocab.msgpack"
DERIVED_META_FILE: Final[str] = "meta.json"

PREP_ARTIFACT_VERSION: Final[str] = "2.0.0"
DERIVED_ARTIFACT_VERSION: Final[str] = "1.0.0"


@dataclass(frozen=True, slots=True)
class PrepMeta:
    corpus_url: str
    limit: int | None
    n_rows: int
    n_tokens: int
    full_vocab_size: int
    tokeniser_hash: str
    version: str
    prep_completed_at: str
    content_hash: str

    def to_json(self) -> str:
        return json.dumps(
            {
                "corpus_url": self.corpus_url,
                "limit": self.limit,
                "n_rows": self.n_rows,
                "n_tokens": self.n_tokens,
                "full_vocab_size": self.full_vocab_size,
                "tokeniser_hash": self.tokeniser_hash,
                "version": self.version,
                "prep_completed_at": self.prep_completed_at,
                "content_hash": self.content_hash,
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, text: str) -> PrepMeta:
        d = json.loads(text)
        try:
            return cls(**d)
        except TypeError as e:
            raise ValueError(
                f"Could not parse prep artifact metadata "
                f"(likely from an older artifact version): {e}"
            ) from e


def args_hash(
    *,
    content_hash: str,
    vocab_size: int,
    context_size: int,
    subsample_t: float,
    subsample_seed: int,
    forced_words: Collection[str],
) -> str:
    """Stable cache key for a (prep-artifact, derivation-args) combo."""
    h = hashlib.sha256()
    h.update(content_hash.encode())
    h.update(f"|{vocab_size}|{context_size}|{subsample_t}|{subsample_seed}|".encode())
    h.update(",".join(sorted(forced_words)).encode())
    return h.hexdigest()[:16]


def compute_content_hash(vocab_bytes: bytes, freq_bytes: bytes) -> str:
    """Content hash for the prep artifact.

    Hashes the full-vocab table + frequency array. Token IDs in the
    HF dataset are deterministic functions of these, so changes to
    the tokens propagate to ``freq.npy`` and the hash moves.
    """
    h = hashlib.sha256()
    h.update(vocab_bytes)
    h.update(freq_bytes)
    return h.hexdigest()[:32]


class PreparedDataset:
    """Read handle on a prep-artifact directory."""

    def __init__(self, path: Path):
        self.path = path
        meta_path = path / ARTIFACT_META_FILE
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Not a prepared dataset directory (no {ARTIFACT_META_FILE}): {path}"
            )
        self.meta = PrepMeta.from_json(meta_path.read_text())
        if Version(self.meta.version).major != Version(PREP_ARTIFACT_VERSION).major:
            raise ValueError(
                f"Prepared dataset at {path} has version={self.meta.version!r}; "
                f"this build reads {PREP_ARTIFACT_VERSION!r}. Re-run "
                f"`mo-net-build-w2v-dataset`."
            )

    @property
    def data_dir(self) -> Path:
        return self.path / ARTIFACT_DATA_DIR

    @property
    def full_vocab_path(self) -> Path:
        return self.path / ARTIFACT_VOCAB_FILE

    @property
    def freq_path(self) -> Path:
        return self.path / ARTIFACT_FREQ_FILE

    def load_full_vocab(self) -> dict[str, int]:
        return msgpack.unpackb(self.full_vocab_path.read_bytes())

    def load_freq(self) -> np.ndarray:
        return np.load(self.freq_path, mmap_mode="r")

    def derive(
        self,
        *,
        vocab_size: int,
        context_size: int,
        subsample_t: float,
        subsample_seed: int = 42,
        forced_words: Collection[str] = (),
    ) -> tuple[Vocab, np.ndarray, np.ndarray]:
        """Materialise (vocab, X, Y) for the given args; cache the result.

        Cache key is ``(content_hash, vocab_size, context_size,
        subsample_t, subsample_seed, sorted forced_words)``. Cached
        ``X`` / ``Y`` are mmap'd from disk; ``vocab`` is decoded fresh.

        Concurrent callers with the same args serialise on a per-key
        lock — the second arrival re-checks the cache after acquiring
        it and short-circuits to the hit path. The build itself writes
        into a sibling ``.tmp`` directory and ``os.replace``-renames on
        success, so a partial directory never appears at the canonical
        path even if the worker is killed mid-build.
        """
        key = args_hash(
            content_hash=self.meta.content_hash,
            vocab_size=vocab_size,
            context_size=context_size,
            subsample_t=subsample_t,
            subsample_seed=subsample_seed,
            forced_words=forced_words,
        )
        derived_path = self.path / DERIVED_DIR / key
        if self._derived_complete(derived_path):
            logger.info(f"derive cache hit: {derived_path}")
            return self._load_derived(derived_path)

        lock_path = self.path / DERIVED_DIR / f".{key}.lock"
        with file_lock(lock_path):
            if self._derived_complete(derived_path):
                logger.info(f"derive cache hit (after waiting on peer): {derived_path}")
                return self._load_derived(derived_path)
            logger.info(f"derive cache miss; building {derived_path}")
            return self._build_derived(
                derived_path=derived_path,
                vocab_size=vocab_size,
                context_size=context_size,
                subsample_t=subsample_t,
                subsample_seed=subsample_seed,
                forced_words=forced_words,
            )

    @staticmethod
    def _derived_complete(derived_path: Path) -> bool:
        return all(
            (derived_path / f).exists()
            for f in (
                DERIVED_META_FILE,
                DERIVED_X_FILE,
                DERIVED_Y_FILE,
                DERIVED_VOCAB_FILE,
            )
        )

    def _load_derived(self, derived_path: Path) -> tuple[Vocab, np.ndarray, np.ndarray]:
        meta = json.loads((derived_path / DERIVED_META_FILE).read_text())
        n_pairs = int(meta["n_pairs"])
        vocab = Vocab.from_bytes((derived_path / DERIVED_VOCAB_FILE).read_bytes())
        X = np.load(derived_path / DERIVED_X_FILE, mmap_mode="r")[:n_pairs]
        Y = np.load(derived_path / DERIVED_Y_FILE, mmap_mode="r")[:n_pairs]
        return vocab, X, Y

    def _build_derived(
        self,
        *,
        derived_path: Path,
        vocab_size: int,
        context_size: int,
        subsample_t: float,
        subsample_seed: int,
        forced_words: Collection[str],
    ) -> tuple[Vocab, np.ndarray, np.ndarray]:
        import datasets

        full_vocab = self.load_full_vocab()
        freq = self.load_freq()
        n_tokens = int(freq.sum())

        # Top-K full-vocab ids by frequency (descending). argpartition is
        # O(N); we only need the top-K, not full sort.
        full_vocab_size = len(full_vocab)
        k = min(vocab_size, full_vocab_size)
        if k < full_vocab_size:
            top_full_ids = np.argpartition(freq, -k)[-k:]
        else:
            top_full_ids = np.arange(full_vocab_size, dtype=np.int64)

        full_id_to_word: dict[int, str] = {
            fid: word for word, fid in full_vocab.items()
        }
        kept_words = [full_id_to_word[int(fid)] for fid in top_full_ids]
        for w in forced_words:
            if w in full_vocab and full_vocab[w] not in set(top_full_ids.tolist()):
                kept_words.append(w)

        # Trainer-side Vocab with sequential IDs.
        vocab_tuple = tuple(kept_words)
        word_counts = {w: int(freq[full_vocab[w]]) for w in vocab_tuple}
        unknown_token_id = len(vocab_tuple)
        vocab = Vocab(
            vocab=vocab_tuple,
            token_to_id={w: i for i, w in enumerate(vocab_tuple)},
            id_to_token=defaultdict(
                lambda: "<unknown>",
                {i: w for i, w in enumerate(vocab_tuple)},
            ),
            unknown_token_id=unknown_token_id,
            word_counts=word_counts,
        )

        # Remap table: full_id → trainer_id or unknown. int32 since trainer
        # vocab fits comfortably in int32 even at vocab_size=1M.
        remap = np.full(full_vocab_size, unknown_token_id, dtype=np.int32)
        for w, trainer_id in vocab.token_to_id.items():
            remap[full_vocab[w]] = trainer_id

        # Per-trainer-id discard probability. Unknown ID never dropped.
        discard_prob = np.zeros(unknown_token_id + 1, dtype=np.float64)
        if subsample_t > 0:
            total_known = sum(word_counts.values())
            if total_known > 0:
                sqrt_t = subsample_t**0.5
                for w, count in word_counts.items():
                    if count <= 0:
                        continue
                    freq_w = count / total_known
                    p = 1.0 - sqrt_t / (freq_w**0.5)
                    if p > 0:
                        discard_prob[vocab[w]] = p

        # Upper bound on n_pairs: total tokens minus the windowing edge per row.
        # Subsampling only reduces this; the 10% margin absorbs the variance.
        n_pairs_upper_bound = max(
            1, int((n_tokens - 2 * context_size * self.meta.n_rows) * 1.10)
        )

        # Build into a sibling `.tmp` dir; `os.replace` it to the canonical
        # path on success so a partial directory never appears at
        # `derived_path` even if this worker is killed mid-build.
        tmp_path = derived_path.with_suffix(derived_path.suffix + ".tmp")
        if tmp_path.exists():
            shutil.rmtree(tmp_path)
        tmp_path.mkdir(parents=True, exist_ok=True)
        x_path = tmp_path / DERIVED_X_FILE
        y_path = tmp_path / DERIVED_Y_FILE

        X = np.lib.format.open_memmap(
            x_path,
            mode="w+",
            dtype=np.int32,
            shape=(n_pairs_upper_bound, 2 * context_size),
        )
        Y = np.lib.format.open_memmap(
            y_path,
            mode="w+",
            dtype=np.int32,
            shape=(n_pairs_upper_bound,),
        )

        ds = datasets.load_from_disk(str(self.data_dir))
        rng = random.Random(subsample_seed)
        pos = 0
        logged_every = max(100_000, self.meta.n_rows // 50)
        for row_idx, row in enumerate(ds):
            full_ids = np.asarray(row["tokens"], dtype=np.int64)  # type: ignore[index,call-overload]
            ids = remap[full_ids]
            if subsample_t > 0:
                # Vectorised subsample: keep when rng draw >= discard_prob[id].
                # Per-row RNG seeded as (subsample_seed XOR row_idx) so worker
                # scheduling can't change the output if we ever parallelise.
                kept_mask = (
                    np.fromiter(
                        (rng.random() for _ in range(ids.size)),
                        dtype=np.float64,
                        count=ids.size,
                    )
                    >= discard_prob[ids]
                )
                ids = ids[kept_mask]
            n = int(ids.size) - 2 * context_size
            if n <= 0:
                continue
            if pos + n > n_pairs_upper_bound:
                n = n_pairs_upper_bound - pos
                if n <= 0:
                    logger.warning(f"pair estimate too low; stopping at {pos:,} pairs")
                    break
            s = ids.astype(np.int32)
            centers = np.arange(context_size, context_size + n)
            for j in range(context_size):
                X[pos : pos + n, j] = s[centers - context_size + j]
                X[pos : pos + n, context_size + j] = s[centers + 1 + j]
            Y[pos : pos + n] = s[centers]
            pos += n
            if (row_idx + 1) % logged_every == 0:
                logger.info(
                    f"  derive: row {row_idx + 1:,}/{self.meta.n_rows:,}, "
                    f"{pos:,} pairs written"
                )

        X.flush()
        Y.flush()
        del X, Y  # release mmap handles before the rename
        (tmp_path / DERIVED_VOCAB_FILE).write_bytes(vocab.serialize())
        derived_meta = {
            "n_pairs": pos,
            "vocab_size": vocab_size,
            "context_size": context_size,
            "subsample_t": subsample_t,
            "subsample_seed": subsample_seed,
            "forced_words": sorted(forced_words),
            "content_hash": self.meta.content_hash,
            "version": DERIVED_ARTIFACT_VERSION,
            "derived_completed_at": datetime.now().isoformat(timespec="seconds"),
        }
        (tmp_path / DERIVED_META_FILE).write_text(json.dumps(derived_meta, indent=2))

        # Atomic-ish swap: rmtree any stale partial at the canonical path,
        # then rename `<args>.tmp` → `<args>`. Both steps run inside the
        # caller's file lock so no peer is reading mid-swap.
        if derived_path.exists():
            shutil.rmtree(derived_path)
        os.replace(tmp_path, derived_path)

        slack = (n_pairs_upper_bound - pos) / max(n_pairs_upper_bound, 1) * 100
        logger.info(
            f"  derive done; {pos:,} pairs ({slack:.1f}% trailing slack); "
            f"cached at {derived_path}"
        )
        return (
            vocab,
            np.load(derived_path / DERIVED_X_FILE, mmap_mode="r")[:pos],
            np.load(derived_path / DERIVED_Y_FILE, mmap_mode="r")[:pos],
        )
