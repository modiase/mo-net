"""Multiprocess corpus statistics for the English-sentences dataset.

Reads the raw corpus, tokenises each sentence with the same rule the
trainer uses (``Vocab.clean_token``), and folds counts into per-process
``Counter``s for 1-, 2-, 3-, 4-, and 5-grams in a single pass. The same
pass also collects 5-gram example windows that contain any of the
target keywords (``king``, ``queen``, ``woman``, ``man``).

Outputs are written to ``--output-dir`` as JSON so the marimo notebook
can load them with no extra deps:

  - totals.json              -- sentence count, total tokens, unique counts per n
  - 1grams.json … 5grams.json -- top-K (gram, count) pairs
  - keyword_samples.json     -- sample windows around the target words

Run on herakles inside the dev shell::

    nix develop -c python -m mo_net.scripts.corpus_stats \\
        --output-dir /data/mo-net/corpus_stats
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

from loguru import logger

from mo_net.samples.word2vec.vocab import (
    ENGLISH_SENTENCES_URL,
    get_stop_words,
    tokenize_line,
)
from mo_net.resources import get_resource

TARGET_WORDS: frozenset[str] = frozenset({"king", "queen", "woman", "man"})
MAX_N: int = 5
DEFAULT_TOP_K: int = 1000
DEFAULT_SAMPLES_PER_KEYWORD: int = 100
DEFAULT_CHUNK_SIZE: int = 5000

_STOP_WORDS: frozenset[str] = frozenset(get_stop_words())


def tokenize(line: str, include_stopwords: bool = False) -> list[str]:
    """Match the trainer's pipeline; pass ``include_stopwords=True`` for the
    raw distribution."""
    tokens = tokenize_line(line)
    if include_stopwords:
        return tokens
    return [t for t in tokens if t not in _STOP_WORDS]


def _process_chunk(
    args: tuple[Sequence[str], int, bool],
) -> tuple[list[Counter[tuple[str, ...]]], dict[str, list[str]]]:
    lines, samples_per_keyword, include_stopwords = args
    counters: list[Counter[tuple[str, ...]]] = [Counter() for _ in range(MAX_N)]
    samples: dict[str, list[str]] = {w: [] for w in TARGET_WORDS}

    for line in lines:
        tokens = tokenize(line, include_stopwords=include_stopwords)
        ntok = len(tokens)
        for n in range(1, MAX_N + 1):
            limit = ntok - n + 1
            if limit <= 0:
                continue
            counter = counters[n - 1]
            for i in range(limit):
                counter[tuple(tokens[i : i + n])] += 1

        # One window per keyword *occurrence* — centred on the keyword with
        # ``half`` tokens on each side. Skip edges where the centred window
        # doesn't fit; otherwise overlapping windows would catch the same
        # occurrence multiple times.
        half = MAX_N // 2
        for i, tok in enumerate(tokens):
            if tok in TARGET_WORDS and len(samples[tok]) < samples_per_keyword:
                start = i - half
                end = i + half + 1
                if start < 0 or end > ntok:
                    continue
                samples[tok].append(" ".join(tokens[start:end]))

    return counters, samples


def _chunks(seq: Sequence[str], size: int) -> Iterator[Sequence[str]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _save_top_k(counter: Counter[tuple[str, ...]], top_k: int, path: Path) -> None:
    top = [
        {"gram": " ".join(gram), "count": count}
        for gram, count in counter.most_common(top_k)
    ]
    path.write_text(json.dumps(top, ensure_ascii=False, indent=2))


def _gather(
    corpus_lines: Sequence[str],
    chunk_size: int,
    samples_per_keyword: int,
    workers: int,
    include_stopwords: bool,
) -> tuple[list[Counter[tuple[str, ...]]], dict[str, list[str]]]:
    counters: list[Counter[tuple[str, ...]]] = [Counter() for _ in range(MAX_N)]
    samples: dict[str, list[str]] = {w: [] for w in TARGET_WORDS}

    inputs: Iterable[tuple[Sequence[str], int, bool]] = (
        (chunk, samples_per_keyword, include_stopwords)
        for chunk in _chunks(corpus_lines, chunk_size)
    )
    with mp.Pool(processes=workers) as pool:
        for chunk_counters, chunk_samples in pool.imap_unordered(
            _process_chunk, inputs, chunksize=1
        ):
            for n in range(MAX_N):
                counters[n].update(chunk_counters[n])
            for w in TARGET_WORDS:
                remaining = samples_per_keyword - len(samples[w])
                if remaining > 0:
                    samples[w].extend(chunk_samples[w][:remaining])

    return counters, samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write the JSON output files to.",
    )
    parser.add_argument(
        "--corpus-url",
        type=str,
        default=ENGLISH_SENTENCES_URL,
        help="Resource URL of the corpus (defaults to the trainer's corpus).",
    )
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K, help="Top-K n-grams to save."
    )
    parser.add_argument(
        "--samples-per-keyword",
        type=int,
        default=DEFAULT_SAMPLES_PER_KEYWORD,
        help="Number of 5-gram windows to keep per target keyword.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Sentences per worker task.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=mp.cpu_count(),
        help="Number of worker processes (defaults to cpu_count).",
    )
    parser.add_argument(
        "--include-stopwords",
        action="store_true",
        help=(
            "Skip the stopword filter (raw corpus view). Default matches the "
            "trainer's pipeline, which filters stopwords before windowing."
        ),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = get_resource(args.corpus_url)
    logger.info(f"loading corpus from {corpus_path}")
    corpus_lines = corpus_path.read_text().splitlines()
    logger.info(f"{len(corpus_lines):,} sentences loaded")

    logger.info(
        f"stopword filter: {'OFF (raw)' if args.include_stopwords else 'ON (trainer view)'}"
    )
    start = time.time()
    counters, samples = _gather(
        corpus_lines,
        chunk_size=args.chunk_size,
        samples_per_keyword=args.samples_per_keyword,
        workers=args.workers,
        include_stopwords=args.include_stopwords,
    )
    logger.info(f"aggregation took {time.time() - start:.1f}s")

    for n in range(MAX_N):
        out = args.output_dir / f"{n + 1}grams.json"
        _save_top_k(counters[n], args.top_k, out)
        logger.info(
            f"wrote {out} ({len(counters[n]):,} unique, "
            f"{sum(counters[n].values()):,} total)"
        )

    totals = {
        "total_sentences": len(corpus_lines),
        "total_tokens": sum(counters[0].values()),
        **{f"unique_{n + 1}grams": len(counters[n]) for n in range(MAX_N)},
        **{f"total_{n + 1}grams": sum(counters[n].values()) for n in range(MAX_N)},
    }
    (args.output_dir / "totals.json").write_text(json.dumps(totals, indent=2))

    (args.output_dir / "keyword_samples.json").write_text(
        json.dumps(samples, ensure_ascii=False, indent=2)
    )

    logger.info(f"done. outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
