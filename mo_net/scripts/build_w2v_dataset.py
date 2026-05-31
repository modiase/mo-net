"""Build a hyperparameter-free word2vec prep artifact from a corpus URL.

Tokenises the corpus once against a full vocab (no top-K filter); produces
``<output-dir>/{data,full_vocab.msgpack,freq.npy,meta.json}``. Training
runs consume it via ``--prepared-dataset`` and derive their own
``(vocab, X, Y)`` lazily — see :mod:`mo_net.samples.word2vec.prepared`.

Parallelism is HF-native via ``Dataset.map(num_proc=N)``; defaults to
``mp.cpu_count()``.
"""

from __future__ import annotations

import hashlib
import inspect
import multiprocessing as mp
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import click
import msgpack  # type: ignore[import-untyped]
import numpy as np
from loguru import logger
from typing import cast

from mo_net.log import LogLevel, setup_logging
from mo_net.resources import get_resource
from mo_net.samples.word2vec.prepared import (
    ARTIFACT_DATA_DIR,
    ARTIFACT_FREQ_FILE,
    ARTIFACT_META_FILE,
    ARTIFACT_VOCAB_FILE,
    PREP_REV,
    PrepMeta,
    compute_content_hash,
)
from mo_net.samples.word2vec.vocab import get_stop_words, tokenize_line


def _tokeniser_hash() -> str:
    """Hash of the tokeniser source — changes here invalidate any prep
    artifact built against the old tokeniser."""
    src = inspect.getsource(tokenize_line)
    return hashlib.sha256(src.encode()).hexdigest()[:16]


def _stopwords_hash(stopwords: frozenset[str]) -> str:
    return hashlib.sha256(",".join(sorted(stopwords)).encode()).hexdigest()[:16]


def _tokenise_batch(
    batch: dict, *, text_column: str, stopwords: frozenset[str]
) -> dict:
    return {
        "tokens": [
            [t for t in tokenize_line(text) if t not in stopwords]
            for text in batch[text_column]
        ]
    }


def _to_ids_batch(batch: dict, *, full_vocab: dict[str, int]) -> dict:
    return {"tokens": [[full_vocab[t] for t in tokens] for tokens in batch["tokens"]]}


@click.command(help=__doc__)
@click.option(
    "--corpus-url",
    required=True,
    type=str,
    help="Resource URL of the corpus (s3://, https://, file://, hf://).",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory to write the prep artifact into. Refuses to clobber.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Cap rows read from the corpus (None = full corpus).",
)
@click.option(
    "--workers",
    type=int,
    default=None,
    help="Parallel workers for HF Dataset.map (defaults to mp.cpu_count()).",
)
@click.option(
    "--text-column",
    type=str,
    default="text",
    help="Column in the source dataset holding the raw text rows.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite an existing artifact directory.",
)
def main(
    corpus_url: str,
    output_dir: Path,
    limit: int | None,
    workers: int | None,
    text_column: str,
    force: bool,
) -> None:
    setup_logging(LogLevel.INFO)
    workers = workers if workers is not None else mp.cpu_count()
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        logger.error(f"{output_dir} is non-empty; pass --force to overwrite.")
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading corpus: {corpus_url}")
    ds = get_resource(corpus_url)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    if text_column not in ds.column_names:
        logger.error(
            f"text column {text_column!r} not in dataset columns {ds.column_names}"
        )
        sys.exit(1)

    stopwords = frozenset(get_stop_words())

    # Stage 1: tokenise (parallel)
    logger.info(f"stage 1/3: tokenising with {workers} workers")
    tokenised = ds.map(
        _tokenise_batch,
        batched=True,
        num_proc=workers,
        remove_columns=ds.column_names,
        fn_kwargs={"text_column": text_column, "stopwords": stopwords},
        desc="tokenise",
    )

    # Stage 2: vocab + freq (sequential reduction over the tokenised column)
    logger.info("stage 2/3: building full vocab + freq table")
    counter: Counter[str] = Counter()
    n_rows = 0
    n_tokens = 0
    log_every = max(100_000, len(tokenised) // 50)
    for row in tokenised:
        toks: list[str] = row["tokens"]  # type: ignore[index,assignment]
        counter.update(toks)
        n_rows += 1
        n_tokens += len(toks)
        if n_rows % log_every == 0:
            logger.info(
                f"  vocab pass: {n_rows:,} rows, {len(counter):,} unique tokens"
            )
    logger.info(
        f"  vocab pass done: {n_rows:,} rows, {len(counter):,} unique tokens, "
        f"{n_tokens:,} total tokens"
    )

    # Freeze full vocab: descending frequency, sequential IDs.
    sorted_words = [w for w, _ in counter.most_common()]
    full_vocab: dict[str, int] = {w: i for i, w in enumerate(sorted_words)}
    freq = np.array([counter[w] for w in sorted_words], dtype=np.int64)
    del counter

    # Stage 3: to_ids (parallel) + save
    logger.info(f"stage 3/3: converting tokens → full_vocab ids with {workers} workers")
    id_dataset = tokenised.map(
        _to_ids_batch,
        batched=True,
        num_proc=workers,
        remove_columns=["tokens"],
        fn_kwargs={"full_vocab": full_vocab},
        desc="to_ids",
    )

    data_dir = output_dir / ARTIFACT_DATA_DIR
    logger.info(f"saving HF dataset to {data_dir}")
    id_dataset.save_to_disk(str(data_dir))

    vocab_bytes = cast(bytes, msgpack.packb(full_vocab))
    freq_bytes = freq.tobytes()
    content_hash = compute_content_hash(vocab_bytes, freq_bytes)

    (output_dir / ARTIFACT_VOCAB_FILE).write_bytes(vocab_bytes)
    with open(output_dir / ARTIFACT_FREQ_FILE, "wb") as f:
        np.save(f, freq)

    meta = PrepMeta(
        corpus_url=corpus_url,
        limit=limit,
        n_rows=n_rows,
        n_tokens=n_tokens,
        full_vocab_size=len(full_vocab),
        tokeniser_hash=_tokeniser_hash(),
        stopwords_hash=_stopwords_hash(stopwords),
        prep_rev=PREP_REV,
        prep_completed_at=datetime.now().isoformat(timespec="seconds"),
        content_hash=content_hash,
    )
    (output_dir / ARTIFACT_META_FILE).write_text(meta.to_json())

    logger.info(
        f"Prep complete: {n_rows:,} rows, {n_tokens:,} tokens, "
        f"{len(full_vocab):,}-word full vocab, "
        f"content_hash={content_hash}"
    )
    logger.info(f"Artifact at: {output_dir}")


if __name__ == "__main__":
    main()
