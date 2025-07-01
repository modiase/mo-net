from __future__ import annotations

import re
from collections.abc import Collection, Iterator, Sequence

import numpy as np
from loguru import logger
from more_itertools import windowed

from mo_net.data import DATA_DIR
from mo_net.model.layer.embedding import Embedding
from mo_net.model.layer.linear import Linear
from mo_net.model.layer.output import SoftmaxOutputLayer
from mo_net.model.layer.reshape import Reshape
from mo_net.model.model import Model
from mo_net.model.module.base import Hidden, Output
from mo_net.protos import NormalisationType
from mo_net.resources import get_resource
from mo_net.train import TrainingParameters
from mo_net.train.backends.log import CsvBackend
from mo_net.train.run import TrainingRun
from mo_net.train.trainer.trainer import BasicTrainer, get_optimizer


def clean_token(token: str) -> str:
    """
    Remove non-printable characters and punctuation
    """
    return re.sub(r"[^\w\s]|[^\x20-\x7E]", "", token).lower().strip()


def all_windows(
    sentences: Collection[Sequence[int]], window_size: int
) -> Iterator[Sequence[int]]:
    return (
        window
        for sentence in sentences
        for window in windowed(sentence, window_size)
        if window is not None
    )


def get_training_set(
    sentences: Collection[Sequence[int]], context_size: int, vocab_size: int
) -> tuple[np.ndarray, np.ndarray]:
    contexts = []
    targets = []

    for sentence in sentences:
        for i in range(context_size, len(sentence) - context_size):
            # Get context words (before and after the target word)
            context = (
                sentence[i - context_size : i] + sentence[i + 1 : i + context_size + 1]
            )
            target = sentence[i]

            contexts.append(context)
            targets.append(target)

    return np.array(contexts), np.eye(vocab_size)[targets]


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
        embedding_layer = Embedding(
            input_dimensions=(context_size * 2,),
            output_dimensions=(context_size * 2, embedding_dim),
            vocab_size=vocab_size,
            parameters=Embedding.Parameters.xavier(vocab_size, embedding_dim),
            store_output_activations=tracing_enabled,
        )

        # Reshape to flatten the context dimension
        reshape_layer = Reshape(
            input_dimensions=(context_size * 2, embedding_dim),
            output_dimensions=(context_size * 2 * embedding_dim,),
        )

        # Linear layer to average across context words
        average_layer = Linear(
            input_dimensions=(context_size * 2 * embedding_dim,),
            output_dimensions=(embedding_dim,),
            parameters=Linear.Parameters.of(
                W=np.ones((context_size * 2 * embedding_dim, embedding_dim))
                / (context_size * 2),
                B=np.zeros(embedding_dim),
            ),
            store_output_activations=tracing_enabled,
            freeze_parameters=True,
        )

        hidden_module = Hidden(layers=(embedding_layer, reshape_layer, average_layer))

        output_module = Output(
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
            output_layer=SoftmaxOutputLayer(input_dimensions=(vocab_size,)),
        )

        return cls(
            input_dimensions=(context_size * 2,),
            hidden=(hidden_module,),
            output=output_module,
        )


def train_cbow():
    shakespeare = get_resource("s3://mo-net-resources/shakespeare.txt").read_text()
    sentences = shakespeare.split("\n")[:1000]
    vocab = sorted(set(token for sentence in sentences for token in sentence.split()))
    token_to_id = {token: i for i, token in enumerate(vocab)}
    token_ids = [
        [token_to_id[token] for token in sentence.split()]
        for sentence in sentences
        if sentence
    ]

    vocab_size = len(vocab)
    embedding_dim = 128
    context_size = 4

    X_train, Y_train = get_training_set(token_ids, context_size, vocab_size)

    logger.info(f"Vocabulary size: {vocab_size}")
    logger.info(f"Embedding dimension: {embedding_dim}")
    logger.info(f"Context size: {context_size}")
    logger.info(f"X_train.shape: {X_train.shape}")
    logger.info(f"Y_train.shape: {Y_train.shape}")

    logger.info(f"X_train: {X_train[:5]}")
    logger.info(f"Y_train: {Y_train[:5]}")

    model = CBOWModel.create(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        context_size=context_size,
        tracing_enabled=False,
    )

    training_parameters = TrainingParameters(
        batch_size=10,
        dropout_keep_probs=(),
        history_max_len=100,
        learning_rate_limits=(1e-4, 1e-2),
        log_level="INFO",
        max_restarts=0,
        monotonic=False,
        no_monitoring=True,
        no_transform=True,
        normalisation_type=NormalisationType.NONE,
        num_epochs=50,
        quiet=False,
        regulariser_lambda=0.0,
        trace_logging=False,
        train_set_size=len(X_train),
        warmup_epochs=5,
        workers=0,
    )

    train_size = int(0.8 * len(X_train))
    X_train_split = X_train[:train_size]
    Y_train_split = Y_train[:train_size]
    X_val = X_train[train_size:]
    Y_val = Y_train[train_size:]

    run = TrainingRun(seed=42, backend=CsvBackend(path=DATA_DIR / "cbow.csv"))
    optimizer = get_optimizer("adam", model, training_parameters)

    trainer = BasicTrainer(
        X_train=X_train_split,
        Y_train=Y_train_split,
        X_val=X_val,
        Y_val=Y_val,
        model=model,
        optimizer=optimizer,
        run=run,
        training_parameters=training_parameters,
    )

    logger.info(f"Starting CBOW training with {len(X_train_split)} training samples")
    logger.info(f"Model architecture: {model.print()}")

    result = trainer.train()

    if hasattr(result, "model_checkpoint_path"):
        logger.info(
            f"Training completed. Model saved to: {result.model_checkpoint_path}"
        )

        embeddings = model.hidden_modules[0].layers[0].parameters.embeddings
        logger.info(f"Learned embeddings shape: {embeddings.shape}")

        id_to_token = {i: token for token, i in token_to_id.items()}

        logger.info("Sample word similarities:")
        for word in ["the", "and", "to", "of", "a", "queen", "king"]:
            if word in token_to_id:
                word_id = token_to_id[word]
                word_embedding = embeddings[word_id]

                similarities = []
                for other_word in [
                    "the",
                    "and",
                    "to",
                    "of",
                    "a",
                    "in",
                    "is",
                    "it",
                    "you",
                    "that",
                    "queen",
                    "king",
                ]:
                    if other_word in token_to_id and other_word != word:
                        other_id = token_to_id[other_word]
                        other_embedding = embeddings[other_id]
                        similarity = np.dot(word_embedding, other_embedding) / (
                            np.linalg.norm(word_embedding)
                            * np.linalg.norm(other_embedding)
                        )
                        similarities.append((other_word, similarity))

                similarities.sort(key=lambda x: x[1], reverse=True)
                logger.info(f"'{word}' similar to: {similarities[:3]}")
    else:
        logger.error(f"Training failed: {result}")


if __name__ == "__main__":
    train_cbow()
