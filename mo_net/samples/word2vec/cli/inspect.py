"""Read-only inspection commands: ``calc`` / ``sample`` / ``eval``.

All three load a trained ``.mar`` archive and probe it without writing
anything. The shared helper :func:`load_model_and_vocab` translates an
archive load error into a click-friendly exception.
"""

from __future__ import annotations

import time
from pathlib import Path

import click
import jax
from loguru import logger

from mo_net.samples.word2vec.analogy import (
    compute_similarity,
    find_most_similar_words,
    parse_and_calculate_expression,
)
from mo_net.samples.word2vec.archive import load_word2vec_archive
from mo_net.samples.word2vec.cli.group import cli
from mo_net.samples.word2vec.models import CBOWModel, SkipGramModel
from mo_net.samples.word2vec.vocab import Vocab


def load_model_and_vocab(model_path: Path) -> tuple[CBOWModel | SkipGramModel, Vocab]:
    if not model_path.exists():
        raise click.ClickException(f"Model file not found: {model_path}")

    seed = time.time_ns() // 1000
    logger.info(f"Using seed: {seed}")

    try:
        model, vocab, _ = load_word2vec_archive(
            model_path,
            training=False,
            key=jax.random.PRNGKey(seed),
        )
    except Exception as exc:
        raise click.ClickException(
            f"Failed to load archive {model_path}: {exc}"
        ) from exc
    return model, vocab


@cli.command("calc", help="Calculate word arithmetic expressions")
@click.option(
    "--model-path",
    type=Path,
    required=True,
    help="Path to the trained model",
)
@click.option(
    "--expr",
    type=str,
    required=True,
    help="Expression to calculate (e.g., 'king - man + woman = queen')",
)
@click.option(
    "--num-results",
    type=int,
    default=5,
    help="Number of similar words to show",
)
def calculate(model_path: Path, expr: str, num_results: int):
    model, vocab = load_model_and_vocab(model_path)

    try:
        result_vector, target_word = parse_and_calculate_expression(expr, model, vocab)
        similarities = find_most_similar_words(result_vector, model, vocab)

        click.echo(f"Expression: {expr}")
        click.echo(f"Target word: {target_word}")
        click.echo()
        click.echo("Most similar words to result:")
        for word, (rank, similarity) in list(similarities.items())[:num_results]:
            click.echo(f"{rank=}, {word=}, {similarity=:.4f}")

        click.echo()
        click.echo(f"Similarity to target word: {similarities[target_word][1]:.4f}")
        click.echo(
            f"Rank of target word: {similarities[target_word][0]}/{len(vocab.vocab)}"
        )

        click.echo()
        click.echo("Similarity to 5 other random words:")
        _, subkey = jax.random.split(jax.random.PRNGKey(time.time_ns() // 1000))

        for _ in range(5):
            subkey, new_key = jax.random.split(subkey)
            word_idx = jax.random.randint(new_key, (), 0, len(vocab.vocab))
            word = list(vocab.vocab)[int(word_idx)]
            rank = similarities[word][0]
            similarity = similarities[word][1]
            click.echo(f"{rank=}, {word=}, {similarity=:.4f}")

    except ValueError as e:
        raise click.ClickException(f"Invalid expression: {e}") from e


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
    model, vocab = load_model_and_vocab(model_path)

    random_words = [
        list(vocab.vocab)[int(i)]
        for i in jax.random.choice(
            jax.random.PRNGKey(time.time_ns() // 1000),
            len(vocab.vocab),
            shape=(min(num_words, len(vocab)),),
            replace=False,
        )
    ]

    click.echo(f"Showing similarities for {len(random_words)} random words:")
    click.echo()

    for word in random_words:
        word_embedding = model.embeddings[vocab[word]]
        click.echo(f"'{word}' (ID: {vocab[word]}):")

        similarities = [
            (
                other_word,
                compute_similarity(word_embedding, model.embeddings[vocab[other_word]]),
            )
            for other_word in vocab.vocab
            if other_word != word
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)
        for similar_word, similarity in similarities[:num_similarities]:
            click.echo(f"    {similar_word}: {similarity:.4f}")
        click.echo()


@cli.command("eval", help="Evaluate word embeddings on analogy tasks")
@click.option(
    "--model-path",
    type=Path,
    required=True,
    help="Path to the trained model",
)
@click.option(
    "--top-k",
    type=int,
    default=5,
    help="Consider top-k predictions as correct",
)
def evaluate(model_path: Path, top_k: int):
    """Evaluate trained word2vec model on word analogy tasks."""
    from mo_net.samples.word2vec.eval import (
        evaluate_analogies,
        evaluate_analogy,
        evaluate_model,
        get_default_analogies,
        print_analogy_results,
    )

    model, vocab = load_model_and_vocab(model_path)

    click.echo("Evaluating word2vec model...")
    click.echo(f"Vocabulary size: {len(vocab.vocab)}")
    click.echo()

    analogies = get_default_analogies()
    results = [
        result
        for example in analogies
        if (result := evaluate_analogy(example, model, vocab, top_k))
    ]

    print_analogy_results(results)

    metrics = evaluate_analogies(analogies, model, vocab, top_k)
    click.echo("\n=== Evaluation Metrics ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            click.echo(f"{key}: {value:.2%}")
        else:
            click.echo(f"{key}: {value}")

    click.echo()
    evaluate_model(model, vocab, analogies)
