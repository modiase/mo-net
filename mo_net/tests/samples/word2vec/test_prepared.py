"""Tests for PreparedDataset.derive — cache hit/miss + correctness."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from mo_net.samples.word2vec.prepared import (
    DERIVED_DIR,
    DERIVED_META_FILE,
    DERIVED_VOCAB_FILE,
    DERIVED_X_FILE,
    DERIVED_Y_FILE,
    PreparedDataset,
    args_hash,
)
from mo_net.scripts.build_w2v_dataset import main as build_main


@pytest.fixture(autouse=True)
def _resource_cache_in_tmp(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MO_NET_RESOURCE_CACHE", str(tmp_path / "cache"))
    from mo_net.settings import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def prep_artifact(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus.txt"
    # Diverse enough to survive stopword filter and produce a real vocab.
    lines = [
        "alpha bravo charlie delta echo foxtrot golf hotel",
        "india juliet kilo lima mike november oscar papa",
        "quebec romeo sierra tango uniform victor whisky xray",
        "alpha alpha bravo charlie delta echo",
        "hotel hotel india juliet kilo",
    ] * 4  # bulk up so windowing has room to bite
    corpus.write_text("\n".join(lines) + "\n")
    out = tmp_path / "prep"
    runner = CliRunner()
    result = runner.invoke(
        build_main,
        [
            "--corpus-url",
            f"file://{corpus}",
            "--output-dir",
            str(out),
            "--workers",
            "1",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    return out


def test_derive_returns_vocab_and_arrays(prep_artifact: Path):
    artifact = PreparedDataset(prep_artifact)
    vocab, X, Y = artifact.derive(
        vocab_size=50,
        context_size=2,
        subsample_t=0,
    )
    assert X.shape[0] == Y.shape[0]
    assert X.shape[1] == 4
    assert X.shape[0] > 0
    assert len(vocab) >= 8


def test_derive_cache_hit_skips_rebuild(prep_artifact: Path):
    artifact = PreparedDataset(prep_artifact)
    _, X1, Y1 = artifact.derive(vocab_size=50, context_size=2, subsample_t=0)
    derived_subdir = next((prep_artifact / DERIVED_DIR).iterdir())
    mtime_before = (derived_subdir / DERIVED_X_FILE).stat().st_mtime

    _, X2, Y2 = artifact.derive(vocab_size=50, context_size=2, subsample_t=0)
    mtime_after = (derived_subdir / DERIVED_X_FILE).stat().st_mtime
    assert mtime_before == mtime_after, "X.npy should not have been rewritten"
    assert np.array_equal(np.asarray(X1), np.asarray(X2))
    assert np.array_equal(np.asarray(Y1), np.asarray(Y2))


def test_derive_different_args_produces_separate_subdirs(prep_artifact: Path):
    artifact = PreparedDataset(prep_artifact)
    artifact.derive(vocab_size=50, context_size=2, subsample_t=0)
    artifact.derive(vocab_size=50, context_size=3, subsample_t=0)
    subdirs = sorted((prep_artifact / DERIVED_DIR).iterdir())
    assert len(subdirs) == 2


def test_derive_subdir_name_matches_args_hash(prep_artifact: Path):
    artifact = PreparedDataset(prep_artifact)
    artifact.derive(
        vocab_size=50,
        context_size=2,
        subsample_t=0,
        subsample_seed=42,
        forced_words=(),
    )
    expected_key = args_hash(
        content_hash=artifact.meta.content_hash,
        vocab_size=50,
        context_size=2,
        subsample_t=0,
        subsample_seed=42,
        forced_words=(),
    )
    assert (prep_artifact / DERIVED_DIR / expected_key).is_dir()


def test_derived_meta_records_actual_n_pairs(prep_artifact: Path):
    artifact = PreparedDataset(prep_artifact)
    _, X, Y = artifact.derive(vocab_size=50, context_size=2, subsample_t=0)
    derived = next((prep_artifact / DERIVED_DIR).iterdir())
    import json

    meta = json.loads((derived / DERIVED_META_FILE).read_text())
    assert meta["n_pairs"] == X.shape[0]
    assert meta["context_size"] == 2
    assert meta["vocab_size"] == 50
    assert meta["content_hash"] == artifact.meta.content_hash


def test_derive_subsampling_reduces_pair_count(prep_artifact: Path):
    artifact = PreparedDataset(prep_artifact)
    _, X_no_sub, _ = artifact.derive(vocab_size=50, context_size=2, subsample_t=0)
    # Aggressive subsampling drops most common tokens.
    _, X_sub, _ = artifact.derive(vocab_size=50, context_size=2, subsample_t=1e-3)
    assert X_sub.shape[0] < X_no_sub.shape[0]


def test_derive_loads_cached_vocab(prep_artifact: Path):
    artifact = PreparedDataset(prep_artifact)
    vocab1, _, _ = artifact.derive(vocab_size=50, context_size=2, subsample_t=0)
    vocab2, _, _ = artifact.derive(vocab_size=50, context_size=2, subsample_t=0)
    assert set(vocab1.vocab) == set(vocab2.vocab)
    assert vocab1.unknown_token_id == vocab2.unknown_token_id
