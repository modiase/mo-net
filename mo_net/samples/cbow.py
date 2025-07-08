from __future__ import annotations

import functools
import random
import re
import time
from collections import Counter, defaultdict
from collections.abc import Collection, Iterator, Sequence
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Callable, ParamSpec, TypeVar, assert_never, cast

import click
import jax.numpy as jnp
import jax.random as random
import msgpack  # type: ignore[import-untyped]
from loguru import logger
from more_itertools import windowed

from mo_net.data import DATA_DIR
from mo_net.device import DeviceType, print_device_info, set_default_device
from mo_net.functions import sparse_cross_entropy
from mo_net.log import LogLevel, setup_logging
from mo_net.model.layer.average import Average
from mo_net.model.layer.embedding import Embedding
from mo_net.model.layer.linear import Linear
from mo_net.model.layer.output import SparseCategoricalCrossentropyOutputLayer
from mo_net.model.model import Model
from mo_net.model.module.base import Hidden, Output
from mo_net.optimizer.base import Base as BaseOptimizer
from mo_net.protos import NormalisationType, TrainingStepHandler, d
from mo_net.resources import get_resource
from mo_net.train import TrainingParameters
from mo_net.train.backends.log import SqliteBackend
from mo_net.train.run import TrainingRun
from mo_net.train.trainer.trainer import (
    BasicTrainer,
    TrainingFailed,
    TrainingSuccessful,
    get_optimizer,
)

P = ParamSpec("P")
R = TypeVar("R")


class EmbeddingWeightDecayRegulariser(TrainingStepHandler):
    def __init__(self, *, lambda_: float, batch_size: int, layer: Embedding):
        self._lambda = lambda_
        self._layer = layer
        self._batch_size = batch_size

    def after_compute_update(self, learning_rate: float) -> None:
        del learning_rate  # unused
        dP = self._layer.cache.get("dP", self._layer.empty_gradient())  # type: ignore[attr-defined]
        if dP is None:
            return
        self._layer.cache["dP"] = d(
            dP
            + Embedding.Parameters(
                embeddings=self._lambda * self._layer.parameters.embeddings,
            )
        )

    def compute_regularisation_loss(self) -> float:
        return (
            0.5
            * self._lambda
            * jnp.sum(self._layer.parameters.embeddings**2)
            / self._batch_size
        )

    def __call__(self) -> float:
        return self.compute_regularisation_loss()

    @staticmethod
    def attach(
        *,
        lambda_: float,
        batch_size: int,
        optimizer: BaseOptimizer,
        model: CBOWModel,
    ) -> None:
        optimizer.register_after_compute_update_handler(
            EmbeddingWeightDecayRegulariser(
                lambda_=lambda_, batch_size=batch_size, layer=model.embedding_layer
            ).after_compute_update
        )


@dataclass(slots=True, frozen=True)
class Vocab:
    vocab: tuple[str, ...]
    token_to_id: dict[str, int]
    id_to_token: dict[int, str]
    unknown_token_id: int

    def serialize(self) -> bytes:
        return msgpack.packb(
            {
                "vocab": list(self.vocab),
                "token_to_id": self.token_to_id,
                "unknown_token_id": self.unknown_token_id,
            }
        )

    @classmethod
    def deserialize(cls, path: Path) -> Vocab:
        with open(path, "rb") as f:
            data = msgpack.unpackb(f.read())
            return cls(
                vocab=tuple(data["vocab"]),
                token_to_id=data["token_to_id"],
                id_to_token=defaultdict(
                    lambda: "<unknown>",
                    {i: token for token, i in data["token_to_id"].items()},
                ),
                unknown_token_id=data.get("unknown_token_id", len(data["vocab"])),
            )

    @classmethod
    def from_vocab(cls, vocab: Collection[str]) -> Vocab:
        vocab_tuple = tuple(vocab)
        return cls(
            vocab=vocab_tuple,
            token_to_id={token: i for i, token in enumerate(vocab_tuple)},
            id_to_token=defaultdict(
                lambda: "<unknown>",
                {i: token for i, token in enumerate(vocab_tuple)},
            ),
            unknown_token_id=len(vocab_tuple),
        )

    @classmethod
    def from_sentences(cls, sentences: Collection[str], max_size: int) -> Vocab:
        most_common_tokens = [
            token
            for token, _ in Counter(
                token
                for sentence in sentences
                if sentence
                for token in sentence.split()
            ).most_common(max_size)
        ]

        vocab_tuple = tuple(most_common_tokens)
        unknown_token_id = len(vocab_tuple)
        return cls(
            vocab=vocab_tuple,
            token_to_id={token: i for i, token in enumerate(vocab_tuple)},
            id_to_token=defaultdict(
                lambda: "<unknown>",
                {i: token for i, token in enumerate(vocab_tuple)},
            ),
            unknown_token_id=unknown_token_id,
        )

    def __len__(self) -> int:
        return len(self.vocab) + 1

    def __getitem__(self, token: str) -> int:
        """Get token ID, returning unknown_token_id if token not in vocabulary"""
        return self.token_to_id.get(token, self.unknown_token_id)


def clean_token(token: str) -> str:
    """
    Remove non-printable characters and punctuation
    """
    return re.sub(r"[^\w\s]|[^\x20-\x7E]", "", token).lower().strip()


def all_windows(
    sentences: Collection[Sequence[int]], window_size: int
) -> Iterator[Sequence[int]]:
    return cast(
        Iterator[Sequence[int]],
        (
            window
            for sentence in sentences
            for window in windowed(sentence, window_size)
            if all(item is not None for item in window)
        ),
    )


def get_training_set(
    tokenized_sentences: Collection[Sequence[int]], context_size: int, vocab_size: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    context, target = zip(
        *[
            (
                tuple(
                    chain(
                        sentence[i - context_size : i],
                        sentence[i + 1 : i + context_size + 1],
                    )
                ),
                sentence[i],
            )
            for sentence in tokenized_sentences
            for i in range(context_size, len(sentence) - context_size)
        ],
        strict=True,
    )
    return jnp.array(context), jnp.array(list(target))


class CBOWModel(Model):
    @classmethod
    def get_name(cls) -> str:
        return "cbow"

    @classmethod
    def get_description(cls) -> str:
        return "Continuous Bag of Words Model"

    @classmethod
    def create(
        cls,
        *,
        vocab_size: int,
        embedding_dim: int,
        context_size: int,
        tracing_enabled: bool = False,
    ) -> CBOWModel:
        return cls(
            input_dimensions=(context_size * 2,),
            hidden=(
                Hidden(
                    layers=(
                        Embedding(
                            input_dimensions=(context_size * 2,),
                            output_dimensions=(context_size * 2, embedding_dim),
                            vocab_size=vocab_size,
                            parameters=Embedding.Parameters.xavier(
                                vocab_size, embedding_dim
                            ),
                            store_output_activations=tracing_enabled,
                        ),
                        Average(
                            input_dimensions=(context_size * 2, embedding_dim),
                            axis=0,
                        ),
                    )
                ),
            ),
            output=Output(
                layers=(
                    Linear(
                        input_dimensions=(embedding_dim,),
                        output_dimensions=(vocab_size,),
                        parameters=Linear.Parameters.xavier(
                            (embedding_dim,), (vocab_size,)
                        ),
                        store_output_activations=tracing_enabled,
                    ),
                ),
                output_layer=SparseCategoricalCrossentropyOutputLayer(
                    input_dimensions=(vocab_size,)
                ),
            ),
        )

    @property
    def embedding_layer(self) -> Embedding:
        return cast(Embedding, self.hidden_modules[0].layers[0])

    @property
    def embeddings(self) -> jnp.ndarray:
        return self.embedding_layer.parameters.embeddings


class PredictModel(CBOWModel):
    @classmethod
    def get_name(cls) -> str:
        return "predict"

    @classmethod
    def get_description(cls) -> str:
        return "Autoregressive Prediction Model based on CBOW"

    @classmethod
    def from_cbow(cls, *, cbow_model: CBOWModel, context_width: int) -> PredictModel:
        """Create a PredictModel from a trained CBOW model for autoregressive generation"""
        embedding_layer = cbow_model.embedding_layer
        vocab_size = embedding_layer.vocab_size
        embedding_dim = embedding_layer.output_dimensions[1]

        new_embedding_layer = Embedding(
            input_dimensions=(context_width,),
            output_dimensions=(context_width, embedding_dim),
            vocab_size=vocab_size,
            parameters=embedding_layer.parameters,
            store_output_activations=False,
        )

        return cls(
            input_dimensions=(context_width,),
            hidden=(
                Hidden(
                    layers=(
                        new_embedding_layer,
                        Average(
                            input_dimensions=(context_width, embedding_dim),
                            axis=0,
                        ),
                    )
                ),
            ),
            output=cbow_model.output,
        )


def training_options(f: Callable[P, R]) -> Callable[P, R]:
    @click.option(
        "--embedding-dim",
        type=int,
        help="Embedding dimension",
        default=128,
    )
    @click.option(
        "--context-size",
        type=int,
        help="Context size",
        default=4,
    )
    @click.option(
        "--batch-size",
        type=int,
        help="Batch size",
        default=10000,
    )
    @click.option(
        "--model-path",
        type=Path,
        help="Path to the trained model",
        default=None,
    )
    @click.option(
        "--num-epochs",
        type=int,
        help="Number of epochs",
        default=100,
    )
    @click.option(
        "--learning-rate",
        type=float,
        help="Learning rate",
        default=1e-4,
    )
    @click.option(
        "--warmup-epochs",
        type=int,
        help="Warmup epochs",
        default=5,
    )
    @click.option(
        "--model-output-path",
        type=Path,
        help="Path to save the trained model",
        default=None,
    )
    @click.option(
        "--log-level",
        type=LogLevel,
        help="Log level",
        default=LogLevel.INFO,
    )
    @click.option(
        "--lambda",
        "lambda_",
        type=float,
        help="Weight decay regulariser lambda",
        default=1e-5,
    )
    @click.option(
        "--vocab-size",
        type=int,
        help="Maximum vocabulary size",
        default=1000,
    )
    @click.option(
        "--device",
        type=click.Choice(["cpu", "gpu", "mps", "auto"]),
        help="Device to use for training (auto will select the best available)",
        default="auto",
    )
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper


@click.group()
def cli():
    """CBOW (Continuous Bag of Words) model CLI"""
    pass


@cli.command("train", help="Train a CBOW model")
@training_options
def train(
    embedding_dim: int,
    context_size: int,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    lambda_: float,
    warmup_epochs: int,
    model_path: Path | None,
    model_output_path: Path | None,
    log_level: LogLevel,
    vocab_size: int,
    device: DeviceType,
):
    """Train a CBOW model on Shakespeare text"""
    setup_logging(log_level)

    set_default_device(device)
    print_device_info()

    sentences = (
        get_resource("s3://mo-net-resources/english-sentences.txt")
        .read_text()
        .split("\n")
    )[:100000]
    vocab = Vocab.from_sentences(sentences, max_size=vocab_size)

    tokenized_sentences = [
        [vocab[token] for token in sentence.split()]
        for sentence in sentences
        if sentence
    ]

    X_train, Y_train = get_training_set(tokenized_sentences, context_size, len(vocab))

    logger.info(f"Vocabulary size: {len(vocab)}")
    logger.info(f"Embedding dimension: {embedding_dim}")
    logger.info(f"Context size: {context_size}")
    logger.info(f"Training samples: {len(X_train)}")

    if model_path is None:
        model = CBOWModel.create(
            vocab_size=len(vocab),
            embedding_dim=embedding_dim,
            context_size=context_size,
            tracing_enabled=False,
        )
    else:
        model = CBOWModel.load(model_path, training=True)

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
        trace_logging=False,
        train_set_size=len(X_train),
        warmup_epochs=warmup_epochs,
        workers=0,
    )

    train_size = int(0.8 * len(X_train))
    X_train_split = X_train[:train_size]
    Y_train_split = Y_train[:train_size]
    X_val = X_train[train_size:]
    Y_val = Y_train[train_size:]

    seed = time.time_ns() // 1000
    logger.info(f"Using seed: {seed}")

    run = TrainingRun(seed=seed, name=f"cbow_run_{seed}", backend=SqliteBackend())
    optimizer = get_optimizer("adam", model, training_parameters)
    EmbeddingWeightDecayRegulariser.attach(
        lambda_=lambda_,
        batch_size=batch_size,
        optimizer=optimizer,
        model=model,
    )

    trainer = BasicTrainer(
        X_train=X_train_split,
        Y_train=Y_train_split,
        X_val=X_val,
        Y_val=Y_val,
        model=model,
        optimizer=optimizer,
        run=run,
        training_parameters=training_parameters,
        loss_fn=sparse_cross_entropy,
        key=random.PRNGKey(seed),
    )

    logger.info(f"Starting CBOW training with {len(X_train_split)} training samples")
    result = trainer.train()

    match result:
        case TrainingSuccessful():
            if model_output_path is None:
                model_output_path = DATA_DIR / "output" / f"{run.name}.pkl"
            result.model_checkpoint_path.rename(model_output_path)
            logger.info(f"Training completed. Model saved to: {model_output_path}")
            vocab_path = model_output_path.with_suffix(".vocab")
            vocab_path.write_bytes(vocab.serialize())
            logger.info(f"Vocabulary saved to: {vocab_path}")
        case TrainingFailed():
            logger.error(f"Training failed: {result.message}")
        case never:
            assert_never(never)


@cli.command("infer", help="Generate text autoregressively from a prompt")
@click.argument("prompt", type=str, required=True)
@click.option(
    "--model-path", type=Path, required=True, help="Path to the trained model"
)
@click.option(
    "--context-width",
    type=int,
    default=4,
    help="Number of context tokens to use for prediction",
)
@click.option(
    "--predict-tokens", type=int, default=4, help="Number of tokens to predict"
)
@click.option(
    "--temperature",
    type=float,
    default=1.0,
    help="Temperature for sampling (higher = more random)",
)
@click.option(
    "--top-p", type=float, default=0.9, help="Top-p (nucleus) sampling parameter"
)
def infer(
    prompt: str,
    model_path: Path,
    context_width: int,
    predict_tokens: int,
    temperature: float,
    top_p: float,
):
    """Generate text autoregressively from a prompt using CBOW model"""
    if not model_path.exists():
        raise click.ClickException(f"Model file not found: {model_path}")

    vocab_path = model_path.with_suffix(".vocab")
    if not vocab_path.exists():
        raise click.ClickException(f"Vocabulary file not found: {vocab_path}")
    vocab = Vocab.deserialize(vocab_path)

    cbow_model = CBOWModel.load(model_path, training=False)
    predict_model = PredictModel.from_cbow(
        cbow_model=cbow_model, context_width=context_width
    )

    tokens = [clean_token(token) for token in prompt.split() if clean_token(token)]
    if not tokens:
        raise click.ClickException("No valid tokens found in prompt")

    token_ids = [vocab[token] for token in tokens]
    context = (
        [vocab.unknown_token_id] * (context_width - len(token_ids)) + token_ids
        if len(token_ids) < context_width
        else token_ids[-context_width:]
    )
    predicted_tokens = []

    for _ in range(predict_tokens):
        logits = (
            predict_model.forward_prop(jnp.array(context)) / temperature
        ).squeeze()
        logits[vocab.unknown_token_id] = float("-inf")
        probs = jnp.exp(logits - jnp.max(logits)) / jnp.sum(
            jnp.exp(logits - jnp.max(logits))
        )
        sorted_indices = jnp.argsort(probs)[::-1]
        cumulative_probs = jnp.cumsum(probs[sorted_indices])
        cutoff_idx = (
            jnp.where(cumulative_probs >= top_p)[0][0]
            if len(jnp.where(cumulative_probs >= top_p)[0]) > 0
            else len(sorted_indices) - 1
        )
        valid_indices = sorted_indices[: cutoff_idx + 1]
        valid_probs = probs[valid_indices] / jnp.sum(probs[valid_indices])
        predicted_token_id = random.choice(
            random.PRNGKey(42), valid_indices, shape=(), p=valid_probs
        )
        predicted_tokens.append(vocab.id_to_token[predicted_token_id])
        context = context[1:] + [predicted_token_id]

    click.echo(click.style(f"> {prompt}", fg="green"))
    click.echo(f"{' '.join(predicted_tokens)}")


@cli.command("sample", help="Show word similarities for random words")
@click.option(
    "--model-path",
    type=Path,
    required=True,
    help="Path to the trained model",
)
@click.option(
    "--num-words",
    type=int,
    default=10,
    help="Number of random words to check",
)
@click.option(
    "--num-similarities",
    type=int,
    default=5,
    help="Number of similar words to show per word",
)
def sample(model_path: Path, num_words: int, num_similarities: int):
    """Show word similarities for random words from the corpus"""
    if not model_path.exists():
        raise click.ClickException(f"Model file not found: {model_path}")

    vocab_path = model_path.with_suffix(".vocab")
    if not vocab_path.exists():
        raise click.ClickException(f"Vocabulary file not found: {vocab_path}")

    vocab = Vocab.deserialize(vocab_path)
    model = CBOWModel.load(model_path, training=False)

    # Use random choice to select random words
    word_indices = random.choice(
        random.PRNGKey(42),
        len(vocab.vocab),
        shape=(min(num_words, len(vocab)),),
        replace=False,
    )
    random_words = [list(vocab.vocab)[int(i)] for i in word_indices]

    click.echo(f"Showing similarities for {len(random_words)} random words:")
    click.echo()

    for word in random_words:
        word_id = vocab[word]
        word_embedding = model.embeddings[word_id]

        click.echo(f"'{word}' (ID: {word_id}):")

        similarities = []
        for other_word in vocab.vocab:
            if other_word != word:
                other_id = vocab[other_word]
                other_embedding = model.embeddings[other_id]
                similarity = jnp.dot(word_embedding, other_embedding) / (
                    jnp.linalg.norm(word_embedding) * jnp.linalg.norm(other_embedding)
                )
                similarities.append((other_word, similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)
        for similar_word, similarity in similarities[:num_similarities]:
            click.echo(f"    {similar_word}: {similarity:.4f}")
        click.echo()


if __name__ == "__main__":
    cli()
