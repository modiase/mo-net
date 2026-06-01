"""`train` command for the word2vec CLI.

End-to-end training driver: vocab + pairs construction (with cache),
optional warm-start from a ``.mar`` archive, model construction, trainer
setup, and final archive write. The full surface is the click decorators
on the function; everything else is the body of ``train()``.
"""

from __future__ import annotations

import contextlib
import time
from pathlib import Path
from typing import Literal, assert_never, cast

import jax
import numpy as np
from loguru import logger

from mo_net import print_device_info
from mo_net.archive import Manifest, ModelMetadata
from mo_net.functions import sparse_cross_entropy
from mo_net.log import LogLevel, setup_logging
from mo_net.protos import NormalisationType
from mo_net.regulariser.weight_decay import EmbeddingWeightDecayRegulariser
from mo_net.samples.word2vec.archive import (
    load_word2vec_archive,
    save_word2vec_archive,
)
from mo_net.samples.word2vec.cli.group import cli, training_options
from mo_net.samples.word2vec.models import CBOWModel, SkipGramModel
from mo_net.samples.word2vec.strategy.softmax import SoftmaxConfig
from mo_net.samples.word2vec.prepared import PreparedDataset
from mo_net.settings import get_settings
from mo_net.train import TrainingParameters
from mo_net.train.backends.log import NullBackend, parse_connection_string
from mo_net.train.run import TrainingRun
from mo_net.train.trainer.trainer import (
    BasicTrainer,
    TrainingFailed,
    TrainingSuccessful,
    get_optimiser,
)


@cli.command("train", help="Train a Word2Vec model")
@training_options
def train(
    *,
    # keep-sorted start
    batch_size: int,
    checkpoint_strategy: Literal["min-val", "last", "both"],
    context_size: int,
    embedding_dim: int,
    eval_batch_size: int,
    health_frequency: int | None,
    history_max_len: int,
    include_words: tuple[str, ...],
    lambda_: float,
    learning_rate: float,
    log_level: LogLevel,
    logging_backend_connection_string: str | None,
    model_output_path: Path | None,
    model_path: Path | None,
    model_type: Literal["cbow", "skipgram"],
    monitor: bool,
    negative_samples: int,
    num_epochs: int,
    prepared_dataset: Path,
    run_name: str | None,
    softmax_strategy: Literal["full", "negative-sampling", "hierarchical"],
    subsample_seed: int,
    subsample_t: float,
    vocab_size: int,
    warmup_epochs: int,
    # keep-sorted end
):
    """Train a Word2Vec model on a pre-built prepared dataset."""
    setup_logging(log_level)

    print_device_info()

    seed = time.time_ns() // 1000
    logger.info(f"Using seed: {seed}")
    key = jax.random.PRNGKey(seed)

    match softmax_strategy:
        case "full":
            softmax_config = SoftmaxConfig.full_softmax()
        case "negative-sampling":
            softmax_config = SoftmaxConfig.negative_sampling(k=negative_samples)
        case "hierarchical":
            softmax_config = SoftmaxConfig.hierarchical_softmax()
        case _:
            raise ValueError(f"Unknown softmax strategy: {softmax_strategy}")

    include_words = tuple(word.lower() for word in include_words)
    parent_run_name: str | None = None
    resume_state = None
    resumed_model: CBOWModel | SkipGramModel | None = None
    if model_path is not None:
        # Resume from a prior archive: load weights, capture the training_state
        # so we can position the optimiser and propagate lineage into the new
        # manifest. ``prefer="last"`` because resumes continue from the most
        # recent state, not the historical best. Caller is responsible for
        # passing a --prepared-dataset whose derived vocab matches the
        # archive's vocab; a mismatch will surface as an embedding-shape
        # error at training time.
        model_, _archive_vocab, manifest = load_word2vec_archive(
            model_path, training=True, key=key, prefer="last"
        )
        resumed_model = model_
        if manifest.training_state is None:
            logger.warning(
                f"{model_path} has no training_state; starting from epoch 0."
            )
        else:
            resume_state = manifest.training_state
            parent_run_name = resume_state.run_name
            logger.info(
                f"Resuming from {parent_run_name!r} at epoch "
                f"{resume_state.completed_epoch} (iteration "
                f"{resume_state.current_iteration})."
            )

    # Construct backend + run + resolve parent BEFORE data prep so the
    # dashboard surfaces the row (and Loki logs route to it) from minute one.
    backend = (
        parse_connection_string(logging_backend_connection_string)
        if logging_backend_connection_string
        else NullBackend()
    )
    logger.info(f"Training-log backend: {backend.connection_string}")
    resolved_name = run_name or f"{model_type}_run_{seed}"
    run = TrainingRun(seed=seed, name=resolved_name, backend=backend)

    parent_run_id: int | None = None
    resume_lineage_id: str | None = None
    if resume_state is not None:
        backend.create()
        if parent_run_name is not None:
            parent_run_id = backend.lookup_run_id_by_name(parent_run_name)
            if parent_run_id is None:
                logger.warning(
                    f"Parent run {parent_run_name!r} not found in backend; "
                    f"lineage_id will be carried but parent_run_id is NULL."
                )
        resume_lineage_id = resume_state.lineage_id

    import os as _os

    # total_batches is unknown until data prep yields train_set_size; back-fill
    # via run.update_totals() once that's known.
    run.start_run(
        total_batches=0,
        total_epochs=num_epochs,
        lineage_id=resume_lineage_id,
        parent_run_id=parent_run_id,
        build_rev=_os.environ.get("MO_NET_BUILD_REV"),
    )
    logger.info(
        f"Registered run {run.name!r} (run_id={run.id}); "
        "total_batches will be back-filled after data prep."
    )

    artifact = PreparedDataset(prepared_dataset)
    logger.info(
        f"Prepared dataset: {prepared_dataset} "
        f"(corpus={artifact.meta.corpus_url}, n_rows={artifact.meta.n_rows:,}, "
        f"n_tokens={artifact.meta.n_tokens:,}, "
        f"full_vocab_size={artifact.meta.full_vocab_size:,})"
    )
    vocab, X_train, Y_train = artifact.derive(
        vocab_size=vocab_size,
        context_size=context_size,
        subsample_t=subsample_t,
        subsample_seed=subsample_seed,
        forced_words=include_words,
    )
    if model_type == "skipgram":
        Y_train, X_train = X_train, Y_train

    if resumed_model is not None:
        model: CBOWModel | SkipGramModel = resumed_model
    else:
        neg_sampling_dist = vocab.get_negative_sampling_distribution()
        match model_type:
            case "cbow":
                model = CBOWModel.create(
                    context_size=context_size,
                    embedding_dim=embedding_dim,
                    key=key,
                    softmax_config=softmax_config,
                    negative_sampling_dist=neg_sampling_dist,
                    tracing_enabled=False,
                    vocab=vocab,
                )
            case "skipgram":
                model = SkipGramModel.create(
                    embedding_dim=embedding_dim,
                    key=key,
                    softmax_config=softmax_config,
                    negative_sampling_dist=neg_sampling_dist,
                    tracing_enabled=False,
                    vocab=vocab,
                    negative_samples=negative_samples,
                )
            case never:
                assert_never(never)

    train_size = int(0.8 * len(X_train))
    X_train_split = X_train[:train_size]
    Y_train_split = Y_train[:train_size]
    X_val = X_train[train_size:]
    Y_val = Y_train[train_size:]

    if model_type == "skipgram":
        # Each (center, [c1..c_{2*ctx}]) becomes 2*ctx separate (center, c_i) rows
        # so the standard sparse_cross_entropy + backward path see matching shapes.
        ctx_full = Y_train_split.shape[1]
        X_train_split = np.repeat(np.asarray(X_train_split), ctx_full).reshape(-1, 1)
        Y_train_split = np.asarray(Y_train_split).flatten()
        X_val = np.repeat(np.asarray(X_val), ctx_full).reshape(-1, 1)
        Y_val = np.asarray(Y_val).flatten()

    logger.info(f"Vocabulary size: {len(vocab)}")
    logger.info(f"Embedding dimension: {embedding_dim}")
    logger.info(f"Context size: {context_size}")
    logger.info(f"Softmax strategy: {softmax_strategy}")
    logger.info(f"Training samples (post expansion): {len(X_train_split)}")

    training_parameters = TrainingParameters(
        batch_size=batch_size,
        checkpoint_strategy=checkpoint_strategy,
        dropout_keep_probs=(),
        eval_batch_size=eval_batch_size,
        history_max_len=history_max_len,
        # Cosine schedule decays from learning_rate to learning_rate/100 over
        # num_epochs; matching the (decayed, peak) ordering the trainer wires
        # WarmupScheduler -> CosineScheduler with.
        learning_rate_limits=(learning_rate / 100, learning_rate),
        log_level=log_level,
        max_restarts=0,
        monotonic=False,
        no_monitoring=not monitor,
        normalisation_type=NormalisationType.NONE,
        num_epochs=num_epochs,
        quiet=False,
        regulariser_lambda=lambda_,
        seed=seed,
        trace_logging=False,
        train_set_size=len(X_train_split),
        warmup_epochs=warmup_epochs,
        workers=0,
    )

    run.update_totals(total_batches=training_parameters.total_batches)
    logger.info(
        f"Back-filled run {run.name!r} totals "
        f"(total_batches={training_parameters.total_batches})."
    )

    optimiser = get_optimiser("adam", model, training_parameters)

    EmbeddingWeightDecayRegulariser.attach(
        lambda_=lambda_,
        batch_size=batch_size,
        optimiser=optimiser,
        model=cast(CBOWModel | SkipGramModel, model),
    )

    trainer = BasicTrainer(
        X_train=X_train_split,
        Y_train=Y_train_split,
        X_val=X_val,
        Y_val=Y_val,
        model=model,
        optimiser=optimiser,
        run=run,
        training_parameters=training_parameters,
        loss_fn=sparse_cross_entropy,
        key=jax.random.PRNGKey(seed),
        start_epoch=resume_state.completed_epoch if resume_state else None,
        lineage_id=resume_lineage_id,
        parent_run_id=parent_run_id,
    )

    if health_frequency is not None:
        from mo_net.samples.word2vec import health

        trainer.subscribe_metric_provider(
            health.provider(
                model=cast(CBOWModel | SkipGramModel, model),
                vocab=vocab,
                every_n_batches=health_frequency or None,
            )
        )
        logger.info(
            "Embedding-health logging enabled "
            f"({'epoch-end' if not health_frequency else f'every {health_frequency} batches'})."
        )

    logger.info(
        f"Starting {model_type} training with {len(X_train_split)} training samples"
    )
    result = trainer.train()

    output_manifest = Manifest(
        metadata=ModelMetadata(type=model_type, softmax_strategy=softmax_strategy),
        training_state=trainer.current_state(parent_run_name=parent_run_name),
    )

    match result:
        case TrainingSuccessful():
            if model_output_path is None:
                output_dir = get_settings().output_dir
                output_dir.mkdir(parents=True, exist_ok=True)
                model_output_path = output_dir / f"{run.name}.mar"

            best_for_archive: CBOWModel | SkipGramModel | None = None
            if (
                checkpoint_strategy == "both"
                and result.model_checkpoint_path
                and result.model_checkpoint_path.exists()
            ):
                # For 'both', the in-memory `model` is the last state; the
                # min-val state lives on disk at `model_checkpoint_path`.
                best_for_archive = _load_pickle_into_model(
                    result.model_checkpoint_path, model_type, key
                )

            save_word2vec_archive(
                model_output_path,
                model=model,
                vocab=vocab,
                manifest=output_manifest,
                best_model=best_for_archive,
            )
            logger.info(f"Training completed. Archive saved to: {model_output_path}")
            with contextlib.suppress(Exception):
                result.model_checkpoint_path.unlink(missing_ok=True)  # type: ignore[attr-defined]
        case TrainingFailed():
            logger.warning(f"Training stopped early: {result.message}")
            if result.model_checkpoint_path and result.model_checkpoint_path.exists():
                if model_output_path is None:
                    output_dir = get_settings().output_dir
                    output_dir.mkdir(parents=True, exist_ok=True)
                    model_output_path = output_dir / f"{run.name}.mar"

                # Load the on-disk best checkpoint (the trainer stops mid-run,
                # so the in-memory model is at a regressed state we don't want
                # to ship).
                model = _load_pickle_into_model(
                    result.model_checkpoint_path, model_type, key
                )
                save_word2vec_archive(
                    model_output_path,
                    model=model,
                    vocab=vocab,
                    manifest=output_manifest,
                )
                logger.info(
                    f"Saved best checkpoint (epoch {result.model_checkpoint_save_epoch}) "
                    f"to: {model_output_path}"
                )
                with contextlib.suppress(Exception):
                    result.model_checkpoint_path.unlink(missing_ok=True)
        case never_2:
            assert_never(never_2)


def _load_pickle_into_model(
    pkl_path: Path,
    model_type: Literal["cbow", "skipgram"],
    key: jax.Array,
) -> CBOWModel | SkipGramModel:
    """Inflate a raw model pickle (intermediate trainer checkpoint).

    Used at archive-write time to read on-disk weights the trainer dumped
    during the loop, before wrapping them in a ``.mar``.
    """
    match model_type:
        case "skipgram":
            return SkipGramModel.load(pkl_path, training=False, key=key)
        case "cbow":
            return CBOWModel.load(pkl_path, training=False)
        case never:
            assert_never(never)
