"""Tests for the build_w2v_dataset CLI."""

from __future__ import annotations

import json
from pathlib import Path

import msgpack  # type: ignore[import-untyped]
import numpy as np
import pytest
from click.testing import CliRunner

from mo_net.samples.word2vec.prepared import (
    ARTIFACT_DATA_DIR,
    ARTIFACT_FREQ_FILE,
    ARTIFACT_META_FILE,
    ARTIFACT_VOCAB_FILE,
    PREP_ARTIFACT_VERSION,
    PrepMeta,
)
from mo_net.scripts.build_w2v_dataset import main


@pytest.fixture(autouse=True)
def _resource_cache_in_tmp(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MO_NET_RESOURCE_CACHE", str(tmp_path / "cache"))
    from mo_net.settings import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _write_corpus(path: Path) -> str:
    path.write_text(
        "alpha bravo charlie delta echo foxtrot golf hotel\n"
        "india juliet kilo lima mike november oscar papa\n"
        "quebec romeo sierra tango uniform victor whisky xray\n"
        "alpha alpha bravo charlie delta echo\n"
        "hotel hotel india juliet kilo\n"
    )
    return f"file://{path}"


def _run(args: list[str]) -> tuple[int, str]:
    runner = CliRunner()
    result = runner.invoke(main, args, catch_exceptions=False)
    return result.exit_code, result.output


def test_prep_produces_expected_layout(tmp_path: Path):
    corpus = tmp_path / "corpus.txt"
    url = _write_corpus(corpus)
    out = tmp_path / "prep"

    code, _ = _run(["--corpus-url", url, "--output-dir", str(out), "--workers", "1"])
    assert code == 0

    assert (out / ARTIFACT_DATA_DIR).is_dir()
    assert (out / ARTIFACT_VOCAB_FILE).is_file()
    assert (out / ARTIFACT_FREQ_FILE).is_file()
    assert (out / ARTIFACT_META_FILE).is_file()

    meta = PrepMeta.from_json((out / ARTIFACT_META_FILE).read_text())
    assert meta.corpus_url == url
    assert meta.n_rows == 5
    assert meta.full_vocab_size >= 20  # plenty of distinct words
    assert meta.version == PREP_ARTIFACT_VERSION
    assert len(meta.content_hash) == 32


def test_prep_content_hash_stable_across_runs(tmp_path: Path):
    corpus = tmp_path / "corpus.txt"
    url = _write_corpus(corpus)

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    _run(["--corpus-url", url, "--output-dir", str(out_a), "--workers", "1"])
    _run(["--corpus-url", url, "--output-dir", str(out_b), "--workers", "1"])

    meta_a = PrepMeta.from_json((out_a / ARTIFACT_META_FILE).read_text())
    meta_b = PrepMeta.from_json((out_b / ARTIFACT_META_FILE).read_text())
    assert meta_a.content_hash == meta_b.content_hash
    assert meta_a.full_vocab_size == meta_b.full_vocab_size
    # The frequency table is deterministic too.
    freq_a = np.load(out_a / ARTIFACT_FREQ_FILE)
    freq_b = np.load(out_b / ARTIFACT_FREQ_FILE)
    assert np.array_equal(freq_a, freq_b)


def test_prep_refuses_to_clobber_nonempty_output(tmp_path: Path):
    corpus = tmp_path / "corpus.txt"
    url = _write_corpus(corpus)
    out = tmp_path / "prep"
    out.mkdir()
    (out / "existing").write_text("ignored")

    code, output = _run(
        ["--corpus-url", url, "--output-dir", str(out), "--workers", "1"]
    )
    assert code != 0
    assert "non-empty" in output or "non-empty" in output.lower()


def test_prep_full_vocab_has_every_token(tmp_path: Path):
    corpus = tmp_path / "corpus.txt"
    url = _write_corpus(corpus)
    out = tmp_path / "prep"

    _run(["--corpus-url", url, "--output-dir", str(out), "--workers", "1"])

    full_vocab = msgpack.unpackb((out / ARTIFACT_VOCAB_FILE).read_bytes())
    # Every non-stopword token from the corpus should be in the full vocab.
    expected = {
        "alpha",
        "bravo",
        "charlie",
        "delta",
        "echo",
        "foxtrot",
        "golf",
        "hotel",
        "india",
        "juliet",
        "kilo",
        "lima",
    }
    for w in expected:
        assert w in full_vocab, f"missing token: {w}"


def test_prep_freq_matches_corpus(tmp_path: Path):
    corpus = tmp_path / "corpus.txt"
    url = _write_corpus(corpus)
    out = tmp_path / "prep"

    _run(["--corpus-url", url, "--output-dir", str(out), "--workers", "1"])

    full_vocab = msgpack.unpackb((out / ARTIFACT_VOCAB_FILE).read_bytes())
    freq = np.load(out / ARTIFACT_FREQ_FILE)
    # 'alpha' appears 3 times across all rows (line 1 once, line 4 twice).
    assert int(freq[full_vocab["alpha"]]) == 3
    # 'hotel' appears 3 times (line 1 once, line 5 twice).
    assert int(freq[full_vocab["hotel"]]) == 3
    # 'foxtrot' appears once.
    assert int(freq[full_vocab["foxtrot"]]) == 1


def test_prep_limit_caps_rows(tmp_path: Path):
    corpus = tmp_path / "corpus.txt"
    url = _write_corpus(corpus)
    out = tmp_path / "prep"

    code, _ = _run(
        [
            "--corpus-url",
            url,
            "--output-dir",
            str(out),
            "--workers",
            "1",
            "--limit",
            "2",
        ]
    )
    assert code == 0
    meta = json.loads((out / ARTIFACT_META_FILE).read_text())
    assert meta["n_rows"] == 2
