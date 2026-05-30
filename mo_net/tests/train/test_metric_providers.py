"""Tests for the metric-provider chain on `BasicTrainer`."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from sqlalchemy import text

from mo_net.functions import sparse_cross_entropy
from mo_net.protos import NormalisationType
from mo_net.samples.word2vec import CBOWModel
from mo_net.samples.word2vec.analogy import all_windows
from mo_net.samples.word2vec.strategy.softmax import SoftmaxConfig
from mo_net.samples.word2vec.vocab import Vocab
from mo_net.train import TrainingParameters
from mo_net.train.backends.log import InMemorySqliteBackend
from mo_net.train.run import TrainingRun
from mo_net.train.trainer.trainer import (
    BasicTrainer,
    MetricContext,
    TrainingSuccessful,
    get_optimiser,
)


@pytest.fixture
def trained_trainer_factory():
    """Returns (build_and_train, backend) — caller registers providers between."""

    def _build():
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
            num_epochs=1,
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
        return trainer, backend, params

    yield _build


def _names_in_metrics(backend: InMemorySqliteBackend) -> set[str]:
    assert backend._session is not None
    rows = backend._session.execute(
        text("SELECT DISTINCT name FROM metrics")
    ).fetchall()
    return {r[0] for r in rows}


class TestDefaultProviders:
    """The trainer ships with batch_loss/val_loss/learning_rate providers."""

    def test_default_providers_log_existing_metrics(
        self, trained_trainer_factory, tmp_path: Path
    ) -> None:
        del tmp_path  # only for fixture isolation
        trainer, backend, _ = trained_trainer_factory()
        result = trainer.train()
        assert isinstance(result, TrainingSuccessful)

        names = _names_in_metrics(backend)
        assert {"batch_loss", "val_loss", "learning_rate"} <= names

        result.model_checkpoint_path.unlink(missing_ok=True)


class TestSubscribedProvider:
    """Providers registered via `subscribe_metric_provider` contribute rows."""

    def test_subscribed_provider_contributes_metrics(
        self, trained_trainer_factory
    ) -> None:
        trainer, backend, _ = trained_trainer_factory()
        trainer.subscribe_metric_provider(lambda ctx: {"custom_metric": 42.0})
        result = trainer.train()
        assert isinstance(result, TrainingSuccessful)

        names = _names_in_metrics(backend)
        assert "custom_metric" in names

        assert backend._session is not None
        values = backend._session.execute(
            text("SELECT DISTINCT value FROM metrics WHERE name = 'custom_metric'")
        ).fetchall()
        assert values == [(42.0,)]

        result.model_checkpoint_path.unlink(missing_ok=True)


class TestProviderExceptionIsolation:
    """A provider that raises must not crash the training loop."""

    def test_provider_exception_does_not_crash_training(
        self, trained_trainer_factory
    ) -> None:
        trainer, backend, _ = trained_trainer_factory()
        trainer.subscribe_metric_provider(
            lambda ctx: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        result = trainer.train()
        assert isinstance(result, TrainingSuccessful)

        # Defaults still landed.
        names = _names_in_metrics(backend)
        assert {"batch_loss", "val_loss", "learning_rate"} <= names

        result.model_checkpoint_path.unlink(missing_ok=True)


class TestMetricContext:
    """`MetricContext` carries the right fields per batch."""

    def test_metric_context_fields(self, trained_trainer_factory) -> None:
        trainer, _, params = trained_trainer_factory()
        captured: list[MetricContext] = []

        def _capture(ctx: MetricContext) -> Mapping[str, float] | None:
            captured.append(ctx)
            return None  # don't contribute metrics

        trainer.subscribe_metric_provider(_capture)
        result = trainer.train()
        assert isinstance(result, TrainingSuccessful)
        assert len(captured) == params.total_batches

        # iteration is 1-based.
        assert captured[0].iteration == 1
        assert captured[-1].iteration == params.total_batches

        # is_epoch_end fires exactly once per epoch.
        epoch_ends = [ctx for ctx in captured if ctx.is_epoch_end]
        assert len(epoch_ends) == params.num_epochs

        # batch_loss + val_loss are real floats (not jnp arrays).
        assert isinstance(captured[0].batch_loss, float)
        assert isinstance(captured[0].val_loss, float)

        result.model_checkpoint_path.unlink(missing_ok=True)


class TestNoneSkipsContribution:
    """Providers returning None contribute no rows, but coexist with others."""

    def test_none_provider_does_not_pollute(self, trained_trainer_factory) -> None:
        trainer, backend, _ = trained_trainer_factory()
        trainer.subscribe_metric_provider(lambda ctx: None)
        trainer.subscribe_metric_provider(lambda ctx: {"present": 1.0})
        result = trainer.train()
        assert isinstance(result, TrainingSuccessful)

        names = _names_in_metrics(backend)
        assert "present" in names

        result.model_checkpoint_path.unlink(missing_ok=True)
