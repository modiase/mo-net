from __future__ import annotations

import contextlib
import functools
import json
import pickle
import time
import zipfile
from collections.abc import Collection, Iterator, Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import IO, Callable, Final, Literal, ParamSpec, TypeVar, assert_never, cast

import click
import jax
import jax.numpy as jnp
from loguru import logger
from more_itertools import windowed

from mo_net import print_device_info
from mo_net.data import DATA_DIR
from mo_net.functions import LossFn, sparse_cross_entropy
from mo_net.log import LogLevel, setup_logging
from mo_net.model.layer.average import Average
from mo_net.model.layer.base import Hidden as HiddenLayer
from mo_net.model.layer.embedding import Embedding
from mo_net.model.layer.linear import Linear
from mo_net.model.layer.output import OutputLayer, SparseCategoricalSoftmaxOutputLayer
from mo_net.model.model import Model
from mo_net.model.module.base import Hidden, Output
from mo_net.protos import (
    Activations,
    D,
    Dimensions,
    NormalisationType,
    SupportsDeserialize,
)
from mo_net.regulariser.weight_decay import EmbeddingWeightDecayRegulariser
from mo_net.samples.word2vec.vocab import (
    TokenizedSentence,
    Vocab,
    get_english_sentences,
    get_training_set,
)
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


MODEL_ZIP_INTERNAL_PATH: Final = "model.pkl"
VOCAB_ZIP_INTERNAL_PATH: Final = "vocab.msgpack"
METADATA_ZIP_INTERNAL_PATH: Final = "metadata.json"


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


class CBOWModel(Model):
    @dataclass(frozen=True, kw_only=True)
    class Serialized:
        input_dimensions: tuple[int, ...]
        hidden_modules: tuple[SupportsDeserialize, ...]
        output_module: SupportsDeserialize

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
        context_size: int,
        embedding_dim: int,
        key: jax.Array,
        tracing_enabled: bool = False,
        vocab_size: int,
    ) -> CBOWModel:
        key1, key2 = jax.random.split(key, 2)
        return cls(
            input_dimensions=(context_size * 2,),
            hidden=(
                Hidden(
                    layers=(
                        Embedding(
                            input_dimensions=(context_size * 2,),
                            output_dimensions=(context_size * 2, embedding_dim),
                            vocab_size=vocab_size,
                            store_output_activations=tracing_enabled,
                            key=key1,
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
                        parameters_init_fn=lambda dim_in,
                        dim_out: Linear.Parameters.xavier(dim_in, dim_out, key=key2),
                        store_output_activations=tracing_enabled,
                    ),
                ),
                output_layer=SparseCategoricalSoftmaxOutputLayer(
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

    def dump(self, out: IO[bytes] | Path) -> None:
        with (
            open(out, "wb")
            if isinstance(out, Path)
            else contextlib.nullcontext(out) as io
        ):
            pickle.dump(
                self.Serialized(
                    input_dimensions=tuple(self.input_layer.input_dimensions),
                    hidden_modules=tuple(
                        module.serialize() for module in self.hidden_modules
                    ),
                    output_module=self.output_module.serialize(),
                ),
                io,
            )

    @classmethod
    def load(
        cls,
        source: IO[bytes] | Path,
        training: bool = False,
        freeze_parameters: bool = False,
    ) -> CBOWModel:
        if isinstance(source, Path):
            with open(source, "rb") as f:
                serialized = pickle.load(f)
        else:
            serialized = pickle.load(source)
        if not isinstance(serialized, cls.Serialized):
            raise ValueError(f"Invalid serialized model: {serialized}")
        return cls(
            input_dimensions=serialized.input_dimensions,
            hidden=tuple(
                module.deserialize(
                    training=training, freeze_parameters=freeze_parameters
                )
                for module in serialized.hidden_modules
            ),
            output=serialized.output_module.deserialize(
                training=training, freeze_parameters=freeze_parameters
            ),
        )


class SkipGramModel(Model):
    @dataclass(frozen=True, kw_only=True)
    class Serialized:
        input_dimensions: tuple[int, ...]
        hidden_modules: tuple[SupportsDeserialize, ...]
        output_module: SupportsDeserialize
        negative_samples: int

    def __init__(
        self,
        *,
        hidden: Sequence[Hidden | HiddenLayer],  # noqa: F821
        input_dimensions: Dimensions,
        key: jax.Array,
        negative_samples: int = 5,
        output: Output | OutputLayer | None = None,
    ):
        super().__init__(
            input_dimensions=input_dimensions, hidden=hidden, output=output
        )
        self._key = key
        self._negative_samples = negative_samples

    @classmethod
    def get_name(cls) -> str:
        return "skipgram"

    @classmethod
    def get_description(cls) -> str:
        return "Skip-Gram Model"

    @classmethod
    def create(
        cls,
        *,
        embedding_dim: int,
        key: jax.Array,
        negative_samples: int,
        tracing_enabled: bool = False,
        vocab_size: int,
    ) -> SkipGramModel:
        key1, key2 = jax.random.split(key, 2)
        return cls(
            input_dimensions=(1,),
            hidden=(
                Hidden(
                    layers=(
                        Embedding(
                            input_dimensions=(1,),
                            output_dimensions=(
                                1,
                                embedding_dim,
                            ),
                            vocab_size=vocab_size,
                            store_output_activations=tracing_enabled,
                            key=key1,
                        ),
                        Average(
                            input_dimensions=(1, embedding_dim),
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
                        parameters_init_fn=lambda dim_in,
                        dim_out: Linear.Parameters.xavier(dim_in, dim_out, key=key2),
                        store_output_activations=tracing_enabled,
                    ),
                ),
                output_layer=SparseCategoricalSoftmaxOutputLayer(
                    input_dimensions=(vocab_size,)
                ),
            ),
            key=key,
            negative_samples=negative_samples,
        )

    def compute_loss(
        self, X: jnp.ndarray, Y_true: jnp.ndarray, loss_fn: LossFn
    ) -> float:
        Y_pred = self.forward_prop(X)
        return loss_fn(Y_pred, Y_true.flatten()) + sum(
            contributor() for contributor in self.loss_contributors
        )

    def backward_prop(self, Y_true: jnp.ndarray) -> D[Activations]:
        batch_size, context_size = Y_true.shape
        self._key, subkey = jax.random.split(self._key)

        dZ = cast(
            SparseCategoricalSoftmaxOutputLayer, self.output.output_layer
        ).backward_prop_with_negative(
            Y_true=Y_true.flatten(),
            Y_negative=jax.random.choice(
                subkey,
                self.embedding_layer.vocab_size,
                shape=(batch_size * context_size * self._negative_samples,),
            ),
        )

        for layer in reversed(self.output_module.layers):
            dZ = layer.backward_prop(dZ=dZ)

        for module in reversed(self.hidden_modules):
            dZ = module.backward_prop(dZ=dZ)
        return self.input_layer.backward_prop(dZ)

    @property
    def embedding_layer(self) -> Embedding:
        return cast(Embedding, self.hidden_modules[0].layers[0])

    @property
    def embeddings(self) -> jnp.ndarray:
        return self.embedding_layer.parameters.embeddings

    def dump(self, out: IO[bytes] | Path) -> None:
        with (
            open(out, "wb")
            if isinstance(out, Path)
            else contextlib.nullcontext(out) as io
        ):
            pickle.dump(
                self.Serialized(
                    input_dimensions=tuple(self.input_layer.input_dimensions),
                    hidden_modules=tuple(
                        module.serialize() for module in self.hidden_modules
                    ),
                    output_module=self.output_module.serialize(),
                    negative_samples=self._negative_samples,
                ),
                io,
            )

    @classmethod
    def load(
        cls,
        source: IO[bytes] | Path,
        training: bool = False,
        freeze_parameters: bool = False,
        key: jax.Array | None = None,
    ) -> SkipGramModel:
        if key is None:
            key = jax.random.PRNGKey(0)

        if isinstance(source, Path):
            with open(source, "rb") as f:
                serialized = pickle.load(f)
        else:
            serialized = pickle.load(source)
        if not isinstance(serialized, cls.Serialized):
            raise ValueError(f"Invalid serialized model: {serialized}")

        return cls(
            input_dimensions=serialized.input_dimensions,
            hidden=tuple(
                module.deserialize(
                    training=training, freeze_parameters=freeze_parameters
                )
                for module in serialized.hidden_modules
            ),
            output=serialized.output_module.deserialize(
                training=training, freeze_parameters=freeze_parameters
            ),
            key=key,
            negative_samples=serialized.negative_samples,
        )


class PredictModel(CBOWModel):
    @classmethod
    def get_name(cls) -> str:
        return "predict"

    @classmethod
    def get_description(cls) -> str:
        return "Autoregressive Prediction Model based on CBOW"

    @classmethod
    def from_cbow(
        cls, *, cbow_model: CBOWModel, context_width: int, key: jax.Array
    ) -> PredictModel:
        """Create a PredictModel from a trained CBOW model for autoregressive generation"""
        embedding_layer = cbow_model.embedding_layer
        vocab_size = embedding_layer.vocab_size
        embedding_dim = embedding_layer.output_dimensions[1]

        new_embedding_layer = Embedding(
            input_dimensions=(context_width,),
            output_dimensions=(context_width, embedding_dim),
            key=key,
            parameters=embedding_layer.parameters,
            store_output_activations=False,
            vocab_size=vocab_size,
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
        "--batch-size",
        type=int,
        help="Batch size",
        default=10000,
    )
    @click.option(
        "--context-size",
        type=int,
        help="Context size",
        default=4,
    )
    @click.option(
        "--embedding-dim",
        type=int,
        help="Embedding dimension",
        default=32,
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
        "--negative-samples",
        type=int,
        help="Number of negative samples",
        default=5,
    )
    @click.option(
        "--num-epochs",
        type=int,
        help="Number of epochs",
        default=100,
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
        default=5,
    )
    @click.option(
        "--include",
        "include_words",
        type=str,
        multiple=True,
        help="Words to force include in vocabulary (can be used multiple times)",
        default=(),
    )
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper


@click.group()
def cli():
    """Word2Vec model CLI"""


@cli.command("train", help="Train a Word2Vec model")
@training_options
def train(
    *,
    batch_size: int,
    context_size: int,
    embedding_dim: int,
    include_words: tuple[str, ...],
    lambda_: float,
    learning_rate: float,
    log_level: LogLevel,
    model_output_path: Path | None,
    model_path: Path | None,
    model_type: Literal["cbow", "skipgram"],
    negative_samples: int,
    num_epochs: int,
    vocab_size: int,
    warmup_epochs: int,
):
    """Train a Word2Vec model on English text"""
    setup_logging(log_level)

    print_device_info()

    seed = time.time_ns() // 1000
    logger.info(f"Using seed: {seed}")
    key = jax.random.PRNGKey(seed)

    if model_path is not None:
        with zipfile.ZipFile(model_path, "r") as zf:
            with zf.open(MODEL_ZIP_INTERNAL_PATH) as mf:
                with zf.open(METADATA_ZIP_INTERNAL_PATH) as md:
                    metadata = json.loads(md.read().decode("utf-8"))
                    loaded_model_type = metadata.get("type", "cbow")

                if loaded_model_type == "skipgram":
                    model: CBOWModel | SkipGramModel = SkipGramModel.load(
                        mf, training=True, key=key
                    )
                else:
                    model = CBOWModel.load(mf, training=True)
                vocab = Vocab.from_bytes(zf.read(VOCAB_ZIP_INTERNAL_PATH))
        sentences = get_english_sentences()
        tokenized_sentences: Collection[TokenizedSentence] = [
            [vocab[token] for token in sentence] for sentence in sentences if sentence
        ]
    else:
        vocab, tokenized_sentences = Vocab.english_sentences(
            max_vocab_size=vocab_size, forced_words=include_words
        )
        match model_type:
            case "cbow":
                model = CBOWModel.create(
                    context_size=context_size,
                    embedding_dim=embedding_dim,
                    key=key,
                    tracing_enabled=False,
                    vocab_size=len(vocab),
                )
            case "skipgram":
                model = SkipGramModel.create(
                    embedding_dim=embedding_dim,
                    key=key,
                    tracing_enabled=False,
                    vocab_size=len(vocab),
                    negative_samples=negative_samples,
                )
            case never:
                assert_never(never)

    X_train, Y_train = get_training_set(tokenized_sentences, context_size)
    if model_type == "skipgram":
        Y_train, X_train = X_train, Y_train

    logger.info(f"Vocabulary size: {len(vocab)}")
    logger.info(f"Embedding dimension: {embedding_dim}")
    logger.info(f"Context size: {context_size}")
    logger.info(f"Training samples: {len(X_train)}")

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
    X_train_split = X_train[:train_size]
    Y_train_split = Y_train[:train_size]
    X_val = X_train[train_size:]
    Y_val = Y_train[train_size:]

    if model_type == "skipgram":
        X_train_split = X_train_split.reshape(-1, 1)
        X_val = X_val.reshape(-1, 1)

    run = TrainingRun(
        seed=seed, name=f"{model_type}_run_{seed}", backend=SqliteBackend()
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
    )

    logger.info(
        f"Starting {model_type} training with {len(X_train_split)} training samples"
    )
    result = trainer.train()

    match result:
        case TrainingSuccessful():
            if model_output_path is None:
                model_output_path = DATA_DIR / "output" / f"{run.name}.zip"

            zip_buffer = BytesIO()
            with zipfile.ZipFile(
                zip_buffer, "w", compression=zipfile.ZIP_DEFLATED
            ) as zf:
                model_buffer = BytesIO()
                model.dump(model_buffer)
                zf.writestr(MODEL_ZIP_INTERNAL_PATH, model_buffer.getvalue())
                zf.writestr(VOCAB_ZIP_INTERNAL_PATH, vocab.serialize())
                zf.writestr(
                    METADATA_ZIP_INTERNAL_PATH,
                    json.dumps({"type": model_type}).encode("utf-8"),
                )

            model_output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(model_output_path, "wb") as f:
                f.write(zip_buffer.getvalue())

            logger.info(f"Training completed. Model zip saved to: {model_output_path}")
            with contextlib.suppress(Exception):
                result.model_checkpoint_path.unlink(missing_ok=True)  # type: ignore[attr-defined]
        case TrainingFailed():
            logger.error(f"Training failed: {result.message}")
        case never_2:
            assert_never(never_2)


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

    seed = time.time_ns() // 1000
    logger.info(f"Using seed: {seed}")

    try:
        with zipfile.ZipFile(model_path, "r") as zf:
            vocab_bytes = zf.read(VOCAB_ZIP_INTERNAL_PATH)
            with zf.open(METADATA_ZIP_INTERNAL_PATH) as md:
                metadata = json.loads(md.read().decode("utf-8"))
                model_type = metadata.get("type", "cbow")

            with zf.open(MODEL_ZIP_INTERNAL_PATH) as mf:
                match model_type:
                    case "skipgram":
                        model: CBOWModel | SkipGramModel = SkipGramModel.load(
                            mf, training=False, key=jax.random.PRNGKey(seed)
                        )
                    case "cbow":
                        model = CBOWModel.load(mf, training=False)
                    case never:
                        assert_never(never)
    except KeyError as e:
        raise click.ClickException(
            f"Missing file in zip: {e.args[0]}. Expected {MODEL_ZIP_INTERNAL_PATH}, {VOCAB_ZIP_INTERNAL_PATH} and optionally {METADATA_ZIP_INTERNAL_PATH}"
        )
    except zipfile.BadZipFile as e:
        logger.exception(f"Invalid zip file: {model_path}", e)
        raise click.ClickException(f"Invalid zip file: {model_path}")

    vocab = Vocab.from_bytes(vocab_bytes)

    word_indices = jax.random.choice(
        jax.random.PRNGKey(seed),
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
