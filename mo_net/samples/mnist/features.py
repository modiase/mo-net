#!/usr/bin/env python3
"""
Script to train an input vector that minimizes loss for one-hot encodings of all classes.
The model is loaded as frozen, and only the input vector is trained.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from loguru import logger

from mo_net.functions import sparse_cross_entropy
from mo_net.model.model import Model


def create_one_hot_targets(num_classes: int) -> jnp.ndarray:
    """Create one-hot encoded targets for all classes."""
    targets = jnp.zeros((num_classes, num_classes))
    for i in range(num_classes):
        targets = targets.at[i, i].set(1.0)
    return targets


def visualize_inputs_with_outputs(
    input_data: jnp.ndarray,
    model_outputs: jnp.ndarray,
    input_dimensions: tuple[int, ...],
    save_path: Path | None = None,
) -> None:
    """
    Visualize reconstructed inputs and their corresponding model outputs.

    Args:
        input_data: The input data to visualize (batch_size, *input_dimensions)
        model_outputs: The model's output probabilities (batch_size, num_classes)
        input_dimensions: Original input dimensions (e.g., (channels, height, width) for images)
        save_path: Optional path to save the visualization
    """
    batch_size = input_data.shape[0]
    num_classes = model_outputs.shape[1]

    _, axes = plt.subplots(batch_size, 2, figsize=(12, 3 * batch_size))
    if batch_size == 1:
        axes = axes.reshape(1, -1)

    for i in range(batch_size):
        sample_input = input_data[i]
        sample_output = model_outputs[i]

        if len(input_dimensions) == 3:
            channels, height, width = input_dimensions
            if channels == 1:
                img = sample_input.reshape(height, width)
                axes[i, 0].imshow(1 - img, cmap="gray_r")
            elif channels == 3:
                img = sample_input.reshape(channels, height, width).transpose(1, 2, 0)
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                axes[i, 0].imshow(img)
            else:
                img = sample_input.reshape(channels, height, width)[0]
                axes[i, 0].imshow(1 - img, cmap="gray_r")
        elif len(input_dimensions) == 1:
            size = input_dimensions[0]
            sqrt_size = int(jnp.sqrt(size))
            if sqrt_size * sqrt_size == size:
                img = sample_input.reshape(sqrt_size, sqrt_size)
                axes[i, 0].imshow(img, cmap="gray_r")
            else:
                axes[i, 0].plot(sample_input)
        else:
            axes[i, 0].plot(sample_input.flatten())

        axes[i, 0].set_title(f"Class {i} - Reconstructed Input")
        axes[i, 0].axis("off")

        bars = axes[i, 1].bar(
            range(num_classes), sample_output, color="skyblue", alpha=0.7
        )

        target_class = i
        bars[target_class].set_color("red")
        bars[target_class].set_alpha(0.8)

        axes[i, 1].set_title(f"Model Output Probabilities (Target: Class {i})")
        axes[i, 1].set_xlabel("Class")
        axes[i, 1].set_ylabel("Probability")
        axes[i, 1].set_xticks(range(num_classes))
        axes[i, 1].set_ylim(0, 1)

        for bar, prob in zip(bars, sample_output, strict=True):
            height = bar.get_height()
            axes[i, 1].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{prob:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    plt.suptitle(
        f"Reconstructed Inputs and Model Outputs (shape: {input_dimensions})", y=1.02
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved visualization to: {save_path}")

    plt.show()


def train_input_vector(
    *,
    key: jax.Array,
    learning_rate: float,
    model_path: Path,
    num_iterations: int,
    patience: int = 200,
    regularization_strength: float = 0.01,
    sparsity_strength: float = 1e-6,
    verbose: bool = True,
) -> tuple[jnp.ndarray, Sequence[float], jnp.ndarray]:
    """
    Train an input vector to minimize loss for one-hot encodings of all classes.

    Args:
        model_path: Path to the frozen model file
        num_iterations: Number of training iterations
        learning_rate: Learning rate for training
        regularization_strength: Strength of L2 regularization on input values
        verbose: Whether to print training progress

    Returns:
        Tuple of (trained_input_vector, loss_history, final_outputs)
    """

    with open(model_path, "rb") as f:
        frozen_model = Model.load(f, training=True, freeze_parameters=True)

    input_dimensions = frozen_model.input_layer.input_dimensions
    num_classes = frozen_model.output_module.output_layer.output_dimensions[0]

    logger.info(f"Model input dimensions: {input_dimensions}")
    logger.info(
        f"Model output dimensions: {frozen_model.output_module.output_layer.output_dimensions}"
    )
    logger.info(f"Number of classes: {num_classes}")

    key = jax.random.split(key)[0]
    input_vector = jax.random.uniform(
        key, (num_classes, *input_dimensions), jnp.float32, 0, 0.1
    )
    logger.info(f"Created trainable input vector with shape: {input_vector.shape}")

    targets = create_one_hot_targets(num_classes)
    logger.debug(f"Targets shape: {targets.shape}")

    loss_history = []
    best_loss = float("inf")
    patience_counter = 0

    logger.info(f"Starting training for {num_iterations} iterations...")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Regularization strength: {regularization_strength}")
    logger.info(f"Training all {num_classes} classes simultaneously")
    logger.info("-" * 50)

    for iteration in range(num_iterations):
        frozen_model.forward_prop(input_vector)

        classification_loss = frozen_model.compute_loss(
            input_vector, targets, loss_fn=sparse_cross_entropy
        )

        regularization_loss = (
            regularization_strength * jnp.sum(input_vector**2) / num_classes
        ).item()

        sparsity_loss = (
            sparsity_strength * jnp.sum(jnp.abs(input_vector)) / num_classes
        ).item()

        total_loss = classification_loss + regularization_loss + sparsity_loss
        loss_history.append(total_loss)

        input_gradient = frozen_model.backward_prop(Y_true=targets)

        reg_gradient = 2 * regularization_strength * input_vector / num_classes
        sparsity_gradient = sparsity_strength * jnp.sign(input_vector) / num_classes
        total_gradient = input_gradient + reg_gradient + sparsity_gradient

        grad_norm = jnp.linalg.norm(total_gradient)
        if grad_norm > 1.0:
            total_gradient = total_gradient / grad_norm

        input_vector = input_vector - learning_rate * total_gradient

        input_vector = jnp.clip(input_vector, 0, 1)

        if total_loss < best_loss:
            best_loss = total_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(
                f"Early stopping at iteration {iteration} (no improvement for {patience} iterations)"
            )
            break

        if verbose and (iteration % 100 == 0 or iteration == num_iterations - 1):
            input_norm = jnp.linalg.norm(input_vector)
            max_input = jnp.max(input_vector)
            min_input = jnp.min(input_vector)
            logger.info(
                f"Iteration {iteration:4d}: Loss = {total_loss:.6f} (Class: {classification_loss:.6f}, Reg: {regularization_loss:.6f}, Sparsity: {sparsity_loss:.6f}, Input Norm: {input_norm:.6f}, Range: [{min_input:.3f}, {max_input:.3f}])"
            )

    logger.info("-" * 50)
    logger.info(f"Training completed. Final loss: {loss_history[-1]:.6f}")

    final_outputs = frozen_model.forward_prop(input_vector)

    return input_vector, loss_history, final_outputs


def main():
    parser = argparse.ArgumentParser(
        description="Train an input vector to minimize loss for one-hot encodings of all classes"
    )
    parser.add_argument("model_path", type=Path, help="Path to the frozen model file")
    parser.add_argument(
        "--iterations",
        type=int,
        default=5000,
        help="Number of training iterations (default: 5000)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for training (default: 0.001)",
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=0.01,
        help="L2 regularization strength for input values (default: 0.01)",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=1e-6,
        help="L1 sparsity regularization strength (default: 1e-6)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=200,
        help="Early stopping patience (default: 200)",
    )
    parser.add_argument(
        "--output", type=Path, help="Path to save the trained input vector (optional)"
    )
    parser.add_argument(
        "--visualization",
        type=Path,
        help="Path to save the input visualization (optional)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    logger.remove()
    log_level = "ERROR" if args.quiet else args.log_level
    logger.add(sys.stderr, level=log_level)

    if not args.model_path.exists():
        logger.error(f"Model file not found: {args.model_path}")
        sys.exit(1)

    try:
        input_vector, _, final_outputs = train_input_vector(
            learning_rate=args.learning_rate,
            model_path=args.model_path,
            num_iterations=args.iterations,
            patience=args.patience,
            regularization_strength=args.regularization,
            sparsity_strength=args.sparsity,
            verbose=not args.quiet,
        )

        if args.output:
            logger.info(f"Saving trained input vector to: {args.output}")
            jnp.savez(
                args.output,
                values=input_vector,
                input_dimensions=input_vector.shape[1:],
            )

        logger.info("Final input vector:")
        logger.info(f"  Shape: {input_vector.shape}")
        logger.info(f"  Norm: {jnp.linalg.norm(input_vector):.6f}")

        logger.info(f"Final model outputs shape: {final_outputs.shape}")

        num_classes = input_vector.shape[0]
        for i in range(num_classes):
            target_prob = final_outputs[i, i]
            logger.info(f"Class {i} confidence: {target_prob:.6f}")

        logger.info("Visualizing reconstructed inputs and outputs...")
        visualize_inputs_with_outputs(
            input_vector,
            final_outputs,
            input_vector.shape[1:],
            save_path=args.visualization,
        )

    except Exception as e:
        logger.exception(f"Error during training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
