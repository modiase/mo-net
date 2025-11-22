from __future__ import annotations

import functools
import time
from pathlib import Path
from typing import Callable, ParamSpec, TypeVar

import click
import jax
import jax.numpy as jnp
from loguru import logger

from mo_net import print_device_info
from mo_net.functions import sparse_cross_entropy
from mo_net.log import LogLevel, setup_logging
from mo_net.model.layer.embedding import Embedding
from mo_net.model.layer.linear import Linear
from mo_net.model.layer.output import SparseCategoricalSoftmaxOutputLayer
from mo_net.model.layer.reshape import Flatten
from mo_net.model.model import Model
from mo_net.model.module.base import Hidden, Output
from mo_net.model.module.rnn import RNN
from mo_net.protos import NormalisationType
from mo_net.train import TrainingParameters
from mo_net.train.backends.log import SqliteBackend
from mo_net.train.run import TrainingRun
from mo_net.train.trainer.trainer import (
    BasicTrainer,
    TrainingFailed,
    TrainingSuccessful,
    get_optimiser,
)

P = ParamSpec("P")
R = TypeVar("R")


class RNNLanguageModel(Model):
    """RNN Language Model: Embedding → RNN → Linear → Softmax."""

    @classmethod
    def get_name(cls) -> str:
        return "rnn_language_model"

    @classmethod
    def get_description(cls) -> str:
        return "RNN Language Model with Learnt Embeddings"

    @classmethod
    def create(
        cls,
        *,
        embedding_dim: int,
        hidden_dim: int,
        key: jax.Array,
        sequence_length: int,
        tracing_enabled: bool = False,
        vocab_size: int,
    ) -> RNNLanguageModel:
        key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)

        return cls(
            input_dimensions=(sequence_length,),
            hidden=(
                Hidden(
                    layers=(
                        Embedding(
                            input_dimensions=(sequence_length,),
                            output_dimensions=(sequence_length, embedding_dim),
                            vocab_size=vocab_size,
                            parameters_init_fn=Embedding.Parameters.xavier,
                            key=subkey1,
                            store_output_activations=tracing_enabled,
                        ),
                    )
                ),
                RNN(
                    input_dimensions=(sequence_length, embedding_dim),
                    hidden_dim=hidden_dim,
                    return_sequences=True,
                    key=subkey2,
                    store_output_activations=tracing_enabled,
                ),
                Hidden(
                    layers=(Flatten(input_dimensions=(sequence_length, hidden_dim)),)
                ),
            ),
            output=Output(
                layers=(
                    Linear(
                        input_dimensions=(sequence_length * hidden_dim,),
                        output_dimensions=(sequence_length * vocab_size,),
                        parameters_init_fn=functools.partial(
                            Linear.Parameters.xavier, key=subkey3
                        ),
                        store_output_activations=tracing_enabled,
                    ),
                ),
                output_layer=SparseCategoricalSoftmaxOutputLayer(
                    input_dimensions=(sequence_length * vocab_size,)
                ),
            ),
        )


def training_options(f: Callable[P, R]) -> Callable[P, R]:
    @click.option("--batch-size", type=int, help="Batch size", default=32)
    @click.option("--num-epochs", type=int, help="Number of epochs", default=50)
    @click.option("--learning-rate", type=float, help="Learning rate", default=1e-3)
    @click.option("--warmup-epochs", type=int, help="Warmup epochs", default=5)
    @click.option(
        "--model-output-path",
        type=Path,
        help="Path to save the trained model",
        default=None,
    )
    @click.option("--log-level", type=LogLevel, help="Log level", default=LogLevel.INFO)
    @click.option(
        "--lambda",
        "lambda_",
        type=float,
        help="Weight decay regulariser lambda",
        default=1e-4,
    )
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper


@click.group()
def cli():
    """RNN Language Model CLI for text generation"""
    pass


@cli.command("demo", help="Run a demo with synthetic data")
@click.option(
    "--vocab-size", type=int, help="Size of vocabulary", default=100, required=False
)
@click.option(
    "--embedding-dim", type=int, help="Embedding dimension", default=32, required=False
)
@click.option(
    "--hidden-dim", type=int, help="RNN hidden dimension", default=64, required=False
)
@click.option(
    "--sequence-length", type=int, help="Sequence length", default=10, required=False
)
@training_options
def demo(
    vocab_size: int,
    embedding_dim: int,
    hidden_dim: int,
    sequence_length: int,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    lambda_: float,
    warmup_epochs: int,
    model_output_path: Path | None,
    log_level: LogLevel,
):
    """Run a demo with synthetic data to verify the model works."""
    setup_logging(log_level)
    print_device_info()

    logger.info("=== RNN Language Model Demo ===")
    logger.info(f"Vocabulary size: {vocab_size}")
    logger.info(f"Embedding dimension: {embedding_dim}")
    logger.info(f"Hidden dimension: {hidden_dim}")
    logger.info(f"Sequence length: {sequence_length}")

    # Generate synthetic training data
    # Input: random sequences of word indices
    # Target: next word at each position (shifted by 1)
    seed = time.time_ns() // 1000
    logger.info(f"Seed: {seed}")
    key = jax.random.PRNGKey(seed)

    # Create synthetic data: 1000 training examples
    num_samples = 1000
    key, subkey = jax.random.split(key)
    X_train = jax.random.randint(
        subkey, (num_samples, sequence_length), 0, vocab_size
    ).astype(jnp.int32)

    # Targets: next token at each position
    # For simplicity, we'll predict the next token in the sequence
    # Y[i, t] = X[i, t+1] (for t < seq_len-1), and Y[i, -1] = random token
    key, subkey = jax.random.split(key)
    Y_train = jnp.concatenate(
        [
            X_train[:, 1:],
            jax.random.randint(subkey, (num_samples, 1), 0, vocab_size),
        ],
        axis=1,
    ).astype(jnp.int32)

    logger.info(f"Training data shape: X={X_train.shape}, Y={Y_train.shape}")

    # Create model
    key, subkey = jax.random.split(key)
    model = RNNLanguageModel.create(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        sequence_length=sequence_length,
        tracing_enabled=False,
        key=subkey,
    )

    logger.info(f"Model created: {model.print()}")
    logger.info(f"Model parameter count: {model.parameter_count}")

    training_parameters = TrainingParameters(
        batch_size=batch_size,
        dropout_keep_probs=(),
        history_max_len=100,
        learning_rate_limits=(learning_rate, learning_rate),
        log_level=log_level,
        max_restarts=0,
        monotonic=False,
        no_monitoring=True,
        normalisation_type=NormalisationType.NONE,
        num_epochs=num_epochs,
        quiet=False,
        regulariser_lambda=lambda_,
        seed=seed,
        trace_logging=False,
        train_set_size=len(X_train),
        warmup_epochs=warmup_epochs,
        workers=0,
    )

    train_size = int(0.8 * len(X_train))
    run = TrainingRun(
        seed=seed, name=f"rnn_language_demo_{seed}", backend=SqliteBackend()
    )

    trainer = BasicTrainer(
        X_train=X_train[:train_size],
        X_val=X_train[train_size:],
        Y_train=Y_train[:train_size].flatten(),
        Y_val=Y_train[train_size:].flatten(),
        key=jax.random.PRNGKey(seed),
        transform_fn=None,
        loss_fn=sparse_cross_entropy,
        model=model,
        optimiser=get_optimiser("adam", model, training_parameters),
        run=run,
        training_parameters=training_parameters,
    )

    logger.info(f"Starting RNN Language Model demo training with {train_size} samples")
    result = trainer.train()

    match result:
        case TrainingSuccessful():
            if model_output_path is None:
                from mo_net.data import DATA_DIR

                model_output_path = DATA_DIR / "output" / f"{run.name}.pkl"
            result.model_checkpoint_path.rename(model_output_path)
            logger.info(f"Training completed. Model saved to: {model_output_path}")

            logger.info("\n=== Sample Generation ===")
            logger.info(f"Input sequence: {X_train[0:1][0].tolist()}")
            predictions = RNNLanguageModel.load(
                model_output_path, training=False
            ).predict(X_train[0:1])
            logger.info(f"Predicted tokens: {predictions.tolist()}")

        case TrainingFailed():
            logger.error(f"Training failed: {result.message}")


@cli.command("info", help="Display information about the RNN language model")
def info():
    """Display information about the RNN language model architecture."""
    logger.info("=== RNN Language Model Architecture ===")
    logger.info("\nComponents:")
    logger.info("1. Embedding Layer: Converts word indices to dense vectors")
    logger.info("2. RNN Layer: Processes sequences with recurrent connections")
    logger.info("3. Linear Layer: Projects RNN outputs to vocabulary size")
    logger.info("4. Softmax: Predicts next word probabilities")
    logger.info("\nFeatures:")
    logger.info("- Learnt word embeddings")
    logger.info("- Vanilla RNN with tanh activation")
    logger.info("- Sequence-to-sequence modeling")
    logger.info("- Next-word prediction at each timestep")
    logger.info("\nUsage:")
    logger.info("  python -m mo_net.samples.rnn_language demo --help")


if __name__ == "__main__":
    cli()
