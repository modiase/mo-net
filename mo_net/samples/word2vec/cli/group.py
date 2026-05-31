"""Click group + shared option decorator for the word2vec CLI.

The ``cli`` group is the root of the ``mo_net.samples.word2vec`` command
tree. The ``training_options`` decorator centralises the ~20 ``@click.option``
declarations shared by training commands so the per-command function only
restates the parameter list (not the click metadata).
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Callable, ParamSpec, TypeVar

import click

from mo_net.log import LogLevel

P = ParamSpec("P")
R = TypeVar("R")


@click.group()
def cli() -> None:
    """Word2Vec model CLI"""


def training_options(f: Callable[P, R]) -> Callable[P, R]:
    @click.option(
        "--batch-size",
        type=int,
        help="Batch size",
        default=10000,
    )
    @click.option(
        "--checkpoint-strategy",
        type=click.Choice(["min-val", "last", "both"]),
        help=(
            "Which model snapshot to persist: 'min-val' = lowest val_loss "
            "seen (selecting on val leaks slightly, default), 'last' = "
            "end-of-run weights, 'both' = main path is latest + sibling "
            ".best.pkl."
        ),
        default="min-val",
    )
    @click.option(
        "--context-size",
        type=int,
        help="Context size",
        default=4,
    )
    @click.option(
        "--corpus-url",
        type=str,
        help=(
            "Resource URL of the training corpus. Any scheme that "
            "mo_net.resources supports works: s3://, https://, file://, "
            "and hf:// for Hugging Face datasets, e.g. "
            "'hf://HuggingFaceFW/fineweb?config=sample-10BT"
            "&split=train&text_field=text'. Default: the project's S3 "
            "English-sentences corpus."
        ),
        default=None,
    )
    @click.option(
        "--embedding-dim",
        type=int,
        help="Embedding dimension",
        default=32,
    )
    @click.option(
        "--health-frequency",
        type=int,
        default=None,
        help=(
            "Cadence in batches for logging embedding-health metrics "
            "(anisotropy / uniformity / PC variance / within-between cosine). "
            "0 = once per epoch; N>0 = every N batches; omitted = disabled."
        ),
    )
    @click.option(
        "--history-max-len",
        type=int,
        default=10,
        help="Number of epochs to track before checking for rising validation loss",
    )
    @click.option(
        "--include",
        "include_words",
        type=str,
        multiple=True,
        help="Words to force include in vocabulary (can be used multiple times)",
        default=(),
    )
    @click.option(
        "--lambda",
        "lambda_",
        type=float,
        help="Weight decay regulariser lambda",
        default=1e-5,
    )
    @click.option(
        "--learning-rate",
        type=float,
        help="Learning rate",
        default=1e-4,
    )
    @click.option(
        "--log-level",
        type=LogLevel,
        help="Log level",
        default=LogLevel.INFO,
    )
    @click.option(
        "--logging-backend-connection-string",
        type=str,
        help=(
            "Connection string for the training-log backend. Examples: "
            "sqlite:///path/to/train.db, postgresql://user:pw@host/db, "
            "mysql://user:pw@host/db, null://. When omitted, training logs are "
            "discarded (NullBackend)."
        ),
        default=None,
    )
    @click.option(
        "--model-output-path",
        type=Path,
        help="Path to save the trained model",
        default=None,
    )
    @click.option(
        "--model-path",
        type=Path,
        help="Path to the trained model",
        default=None,
    )
    @click.option(
        "--model-type",
        type=click.Choice(["cbow", "skipgram"]),
        help="Model type",
        default="cbow",
    )
    @click.option(
        "--monitor/--no-monitor",
        "monitor",
        default=False,
        help="Enable training monitor (early stopping on rising validation loss)",
    )
    @click.option(
        "--negative-samples",
        type=int,
        help="Number of negative samples (only used with negative-sampling strategy)",
        default=5,
    )
    @click.option(
        "--num-epochs",
        type=int,
        help="Number of epochs",
        default=100,
    )
    @click.option(
        "--run-name",
        type=str,
        help=(
            "Explicit run name recorded in the training-log backend. "
            "Defaults to <model_type>_run_<seed>."
        ),
        default=None,
    )
    @click.option(
        "--sentence-limit",
        type=int,
        help="Cap the number of sentences read from the English corpus. "
        "Default: full corpus (~1.5M sentences).",
        default=None,
    )
    @click.option(
        "--softmax-strategy",
        type=click.Choice(["full", "negative-sampling", "hierarchical"]),
        help="Softmax computation strategy",
        default="negative-sampling",
    )
    @click.option(
        "--subsample-t",
        type=float,
        help=(
            "Mikolov frequent-word subsampling threshold: each occurrence "
            "of word w is dropped with probability 1 - sqrt(t/f(w)). "
            "Set to 0 to disable. Default: 1e-5."
        ),
        default=1e-5,
    )
    @click.option(
        "--vocab-size",
        type=int,
        help="Maximum vocabulary size",
        default=1000,
    )
    @click.option(
        "--warmup-epochs",
        type=int,
        help="Warmup epochs",
        default=1,
    )
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper
