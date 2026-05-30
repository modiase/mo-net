"""Tests for `health.provider`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from sqlalchemy import text

from mo_net.functions import sparse_cross_entropy
from mo_net.protos import NormalisationType
from mo_net.samples.word2vec import CBOWModel
from mo_net.samples.word2vec.analogy import all_windows
from mo_net.samples.word2vec.health import provider as health_provider
from mo_net.samples.word2vec.strategy.softmax import SoftmaxConfig
from mo_net.samples.word2vec.vocab import Vocab
from mo_net.train import TrainingParameters
from mo_net.train.backends.log import InMemorySqliteBackend
from mo_net.train.run import TrainingRun
from mo_net.train.trainer.trainer import (
    BasicTrainer,
    TrainingSuccessful,
    get_optimiser,
)

ALWAYS_LOGGED = {
    "embedding_health/anisotropy",
    "embedding_health/uniformity",
    "embedding_health/top1_pc_variance",
    "embedding_health/top3_pc_variance",
    "embedding_health/n_highlight_matched",
}


@pytest.fixture
def setup():
    """Build (model, vocab, trainer, backend, params) — caller registers provider."""
    sentences = [
        ["the", "quick", "brown", "fox"],
        ["the", "lazy", "dog"],
        ["the", "cat", "sat"],
        ["quick", "brown", "fox", "jumps"],
    ] * 20
    vocab, tokenized = Vocab.from_sentences(sentences, max_size=1000)
    model = CBOWModel.create(
        vocab=vocab,
        embedding_dim=16,
        context_size=1,
        softmax_config=SoftmaxConfig.negative_sampling(k=5),
        key=jax.random.PRNGKey(42),
    )
    windows = list(all_windows(tokenized, window_size=3))
    X = jnp.array([[w[0], w[2]] for w in windows])
    Y = jnp.array([w[1] for w in windows])
    split = int(0.8 * len(X))
    X_train, Y_train, X_val, Y_val = X[:split], Y[:split], X[split:], Y[split:]
    params = TrainingParameters(
        batch_size=8,
        num_epochs=2,
        quiet=True,
        train_set_size=len(X_train),
        no_monitoring=True,
        dropout_keep_probs=(),
        history_max_len=100,
        learning_rate_limits=(0.001, 0.001),
        log_level="INFO",
        max_restarts=0,
        monotonic=False,
        normalisation_type=NormalisationType.NONE,
        regulariser_lambda=0.0,
        seed=42,
        trace_logging=False,
        warmup_epochs=0,
        workers=0,
    )
    backend = InMemorySqliteBackend()
    run = TrainingRun(seed=42, name="t", backend=backend)
    optim = get_optimiser("adam", model, params)
    trainer = BasicTrainer(
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_val,
        Y_val=Y_val,
        model=model,
        optimiser=optim,
        run=run,
        training_parameters=params,
        loss_fn=sparse_cross_entropy,
        key=jax.random.PRNGKey(42),
        disable_shutdown=True,
    )
    return model, vocab, trainer, backend, params


def _count_by_name(backend: InMemorySqliteBackend, name: str) -> int:
    assert backend._session is not None
    return (
        backend._session.execute(
            text("SELECT COUNT(*) FROM metrics WHERE name = :n"), {"n": name}
        ).scalar()
        or 0
    )


def _names_in_metrics(backend: InMemorySqliteBackend) -> set[str]:
    assert backend._session is not None
    rows = backend._session.execute(
        text("SELECT DISTINCT name FROM metrics")
    ).fetchall()
    return {r[0] for r in rows}


class TestHealthProvider:
    def test_fires_only_at_epoch_end_by_default(self, setup) -> None:
        model, vocab, trainer, backend, params = setup
        trainer.subscribe_metric_provider(health_provider(model=model, vocab=vocab))
        result = trainer.train()
        assert isinstance(result, TrainingSuccessful)

        # Exactly one health snapshot per epoch.
        assert (
            _count_by_name(backend, "embedding_health/anisotropy") == params.num_epochs
        )

        result.model_checkpoint_path.unlink(missing_ok=True)

    def test_always_logged_metric_names_present(self, setup) -> None:
        model, vocab, trainer, backend, _ = setup
        trainer.subscribe_metric_provider(health_provider(model=model, vocab=vocab))
        result = trainer.train()
        assert isinstance(result, TrainingSuccessful)

        names = _names_in_metrics(backend)
        assert ALWAYS_LOGGED <= names

        result.model_checkpoint_path.unlink(missing_ok=True)

    def test_conditional_metrics_absent_when_no_highlights_match(self, setup) -> None:
        model, vocab, trainer, backend, _ = setup
        # No vocab token matches these — within/between fields should be None.
        trainer.subscribe_metric_provider(
            health_provider(
                model=model,
                vocab=vocab,
                highlight_words=("zzzz_unknown", "nope_neither"),
            )
        )
        result = trainer.train()
        assert isinstance(result, TrainingSuccessful)

        names = _names_in_metrics(backend)
        assert "embedding_health/within_highlight_cosine" not in names
        assert "embedding_health/between_cosine" not in names
        assert "embedding_health/within_between_ratio" not in names
        # Unconditional ones still appear.
        assert "embedding_health/anisotropy" in names

        result.model_checkpoint_path.unlink(missing_ok=True)

    def test_failure_skips_silently(self, setup, monkeypatch) -> None:
        model, vocab, trainer, backend, _ = setup
        import mo_net.samples.word2vec.health as hl

        def _boom(*args, **kwargs):
            raise RuntimeError("simulated SVD failure")

        monkeypatch.setattr(hl, "compute_health_metrics", _boom)
        trainer.subscribe_metric_provider(health_provider(model=model, vocab=vocab))
        result = trainer.train()
        assert isinstance(result, TrainingSuccessful)

        # No health rows landed, but training completed and defaults are present.
        names = _names_in_metrics(backend)
        for name in ALWAYS_LOGGED:
            assert name not in names
        assert "val_loss" in names

        result.model_checkpoint_path.unlink(missing_ok=True)

    def test_every_n_batches_override(self, setup) -> None:
        model, vocab, trainer, backend, params = setup
        trainer.subscribe_metric_provider(
            health_provider(model=model, vocab=vocab, every_n_batches=2)
        )
        result = trainer.train()
        assert isinstance(result, TrainingSuccessful)

        # Fires every 2 batches, not at epoch ends. total_batches // 2 snapshots.
        expected = params.total_batches // 2
        assert _count_by_name(backend, "embedding_health/anisotropy") == expected

        result.model_checkpoint_path.unlink(missing_ok=True)
