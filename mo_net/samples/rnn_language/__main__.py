from __future__ import annotations

import functools
import pickle
import time
import zipfile
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
from mo_net.samples.word2vec.vocab import Vocab
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
        pretrained_embeddings: jnp.ndarray | None = None,
        freeze_embeddings: bool = False,
    ) -> RNNLanguageModel:
        key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)

        if pretrained_embeddings is not None:
            embedding_params = Embedding.Parameters(embeddings=pretrained_embeddings)
            embedding_layer = Embedding(
                input_dimensions=(sequence_length,),
                output_dimensions=(sequence_length, embedding_dim),
                vocab_size=vocab_size,
                parameters=embedding_params,
                freeze_parameters=freeze_embeddings,
                key=subkey1,
                store_output_activations=tracing_enabled,
            )
        else:
            embedding_layer = Embedding(
                input_dimensions=(sequence_length,),
                output_dimensions=(sequence_length, embedding_dim),
                vocab_size=vocab_size,
                parameters_init_fn=Embedding.Parameters.xavier,
                key=subkey1,
                store_output_activations=tracing_enabled,
            )

        return cls(
            input_dimensions=(sequence_length,),
            hidden=(
                Hidden(layers=(embedding_layer,)),
                RNN(
                    input_dimensions=(embedding_dim,),
                    hidden_dimensions=(hidden_dim,),
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

    seed = time.time_ns() // 1000
    logger.info(f"Seed: {seed}")
    key = jax.random.PRNGKey(seed)

    num_samples = 1000
    key, subkey = jax.random.split(key)
    X_train = jax.random.randint(
        subkey, (num_samples, sequence_length), 0, vocab_size
    ).astype(jnp.int32)

    # Y[i, t] = X[i, t+1], with last position filled by random token
    key, subkey = jax.random.split(key)
    Y_train = jnp.concatenate(
        [
            X_train[:, 1:],
            jax.random.randint(subkey, (num_samples, 1), 0, vocab_size),
        ],
        axis=1,
    ).astype(jnp.int32)

    logger.info(f"Training data shape: X={X_train.shape}, Y={Y_train.shape}")

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


def load_word2vec_model(model_path: Path) -> tuple[jnp.ndarray, Vocab]:
    """Load word2vec embeddings and vocabulary from a trained model."""
    import mo_net.samples.word2vec.__main__  # noqa: F401 - needed for pickle

    class Word2VecUnpickler(pickle.Unpickler):
        """Resolves class names when word2vec was trained as __main__."""

        def find_class(self, module: str, name: str):
            if module == "__main__" or "rnn_language" in module:
                module = "mo_net.samples.word2vec.__main__"
            return super().find_class(module, name)

    with zipfile.ZipFile(model_path, "r") as zf:
        vocab = Vocab.from_bytes(zf.read("vocab.msgpack"))
        with zf.open("model.pkl") as mf:
            serialized = Word2VecUnpickler(mf).load()
            embedding_layer = serialized.hidden_modules[0].layers[0]
            embeddings = embedding_layer.parameters.embeddings

    return embeddings, vocab


def create_language_model_training_data(
    vocab: Vocab,
    sequence_length: int,
    num_sentences: int = 10000,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Create sliding window training data from English sentences."""
    from mo_net.samples.word2vec.vocab import get_english_sentences

    sentences = get_english_sentences(limit=num_sentences)

    tokenized = [
        [vocab[word] for word in sentence]
        for sentence in sentences
        if len(sentence) > sequence_length
    ]

    X_list = []
    Y_list = []
    for tokens in tokenized:
        for i in range(len(tokens) - sequence_length):
            X_list.append(tokens[i : i + sequence_length])
            Y_list.append(tokens[i + 1 : i + sequence_length + 1])

    if not X_list:
        raise ValueError("No training data created. Try smaller sequence_length.")

    return jnp.array(X_list, dtype=jnp.int32), jnp.array(Y_list, dtype=jnp.int32)


@cli.command("train-pretrained", help="Train with pre-trained word2vec embeddings")
@click.option(
    "--word2vec-model",
    type=Path,
    required=True,
    help="Path to trained word2vec model (zip file)",
)
@click.option(
    "--sequence-length",
    type=int,
    default=5,
    help="Sequence length for language modeling",
)
@click.option(
    "--hidden-dim",
    type=int,
    default=64,
    help="RNN hidden dimension",
)
@click.option(
    "--freeze-embeddings/--finetune-embeddings",
    default=True,
    help="Whether to freeze word2vec embeddings during training",
)
@click.option(
    "--num-sentences",
    type=int,
    default=10000,
    help="Number of sentences to use for training",
)
@training_options
def train_pretrained(
    word2vec_model: Path,
    sequence_length: int,
    hidden_dim: int,
    freeze_embeddings: bool,
    num_sentences: int,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    lambda_: float,
    warmup_epochs: int,
    model_output_path: Path | None,
    log_level: LogLevel,
):
    """Train RNN language model with pre-trained word2vec embeddings."""
    setup_logging(log_level)
    print_device_info()

    logger.info("=== RNN Language Model with Pre-trained Word2Vec ===")
    logger.info(f"Loading word2vec model from {word2vec_model}")
    embeddings, vocab = load_word2vec_model(word2vec_model)

    vocab_size = len(vocab)
    embedding_dim = embeddings.shape[1]

    logger.info(f"Vocabulary size: {vocab_size}")
    logger.info(f"Embedding dimension: {embedding_dim}")
    logger.info(f"Hidden dimension: {hidden_dim}")
    logger.info(f"Sequence length: {sequence_length}")
    logger.info(f"Freeze embeddings: {freeze_embeddings}")
    logger.info(f"Creating training data from {num_sentences} sentences...")
    X_train, Y_train = create_language_model_training_data(
        vocab, sequence_length, num_sentences
    )
    logger.info(f"Training data: X={X_train.shape}, Y={Y_train.shape}")

    seed = time.time_ns() // 1000
    logger.info(f"Seed: {seed}")
    key = jax.random.PRNGKey(seed)

    model = RNNLanguageModel.create(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        sequence_length=sequence_length,
        pretrained_embeddings=embeddings,
        freeze_embeddings=freeze_embeddings,
        tracing_enabled=False,
        key=key,
    )

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
        seed=seed, name=f"rnn_language_w2v_{seed}", backend=SqliteBackend()
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

    logger.info(f"Starting training with {train_size} samples")
    result = trainer.train()

    match result:
        case TrainingSuccessful():
            from mo_net.data import DATA_DIR

            if model_output_path is None:
                model_output_path = DATA_DIR / "output" / f"{run.name}.pkl"

            result.model_checkpoint_path.rename(model_output_path)
            logger.info(f"Training completed. Model saved to: {model_output_path}")

            logger.info("\n=== Sample Text Generation ===")
            sample_x = X_train[:1]
            input_words = [vocab.id_to_token[int(idx)] for idx in sample_x[0]]
            logger.info(f"Input: {' '.join(input_words)}")

            logits = model.forward_prop(sample_x)
            pred_reshaped = logits.reshape(sequence_length, vocab_size)
            top_predictions = jnp.argmax(pred_reshaped, axis=1)
            predicted_words = [vocab.id_to_token[int(idx)] for idx in top_predictions]
            logger.info(f"Predicted next words: {' '.join(predicted_words)}")

        case TrainingFailed():
            logger.error(f"Training failed: {result.message}")


@cli.command("generate", help="Generate text using a trained model")
@click.option(
    "--model-path",
    type=Path,
    required=True,
    help="Path to trained RNN language model",
)
@click.option(
    "--word2vec-model",
    type=Path,
    required=True,
    help="Path to word2vec model (for vocabulary)",
)
@click.option(
    "--prompt",
    type=str,
    required=True,
    help="Starting words for generation (space-separated)",
)
@click.option(
    "--num-words",
    type=int,
    default=10,
    help="Number of words to generate",
)
@click.option(
    "--temperature",
    type=float,
    default=1.0,
    help="Sampling temperature (higher = more random)",
)
def generate(
    model_path: Path,
    word2vec_model: Path,
    prompt: str,
    num_words: int,
    temperature: float,
):
    """Generate text using a trained RNN language model."""
    setup_logging(LogLevel.INFO)

    _, vocab = load_word2vec_model(word2vec_model)
    vocab_size = len(vocab)
    model = RNNLanguageModel.load(model_path, training=False)
    sequence_length = model.input_layer.input_dimensions[0]

    prompt_words = prompt.lower().split()
    prompt_tokens = [vocab[word] for word in prompt_words]

    if len(prompt_tokens) < sequence_length:
        prompt_tokens = [vocab.unknown_token_id] * (
            sequence_length - len(prompt_tokens)
        ) + prompt_tokens
    elif len(prompt_tokens) > sequence_length:
        prompt_tokens = prompt_tokens[-sequence_length:]

    generated_words = list(prompt_words)
    current_tokens = prompt_tokens.copy()

    key = jax.random.PRNGKey(int(time.time()))

    for _ in range(num_words):
        X = jnp.array([current_tokens], dtype=jnp.int32)
        logits = model.forward_prop(X)
        pred_reshaped = logits.reshape(sequence_length, vocab_size)
        last_logits = pred_reshaped[-1]

        if temperature != 1.0:
            last_logits = last_logits / temperature

        key, subkey = jax.random.split(key)
        probs = jax.nn.softmax(last_logits)
        next_token = int(jax.random.choice(subkey, vocab_size, p=probs))

        next_word = vocab.id_to_token[next_token]
        generated_words.append(next_word)
        current_tokens = current_tokens[1:] + [next_token]

    click.echo(" ".join(generated_words))


if __name__ == "__main__":
    cli()
