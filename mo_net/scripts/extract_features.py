#!/usr/bin/env python3
"""
Script to train an input layer that minimizes loss for a target that is a one-hot encoding of class 0.
The model is loaded as frozen, and only the input layer is trained.
"""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from mo_net.model.layer.base import ParametrisedHidden
from mo_net.model.model import Model
from mo_net.optimizer.adam import AdaM
from mo_net.optimizer.scheduler import ConstantScheduler
from mo_net.protos import Activations, D, d


class SimpleTrainableInput(ParametrisedHidden):
    """
    A simple trainable input layer that directly learns the input values.
    This is optimized for the task of generating inputs that minimize loss for a specific target.
    """

    @dataclass
    class Parameters:
        values: np.ndarray  # The actual input values to learn

        def __add__(self, other):
            if isinstance(other, self.__class__):
                return self.__class__(values=self.values + other.values)
            else:
                return self.__class__(values=self.values + other)

        def __radd__(self, other):
            return self.__add__(other)

        def __neg__(self):
            return self.__class__(values=-self.values)

        def __sub__(self, other):
            return self.__add__(-other)

        def __rsub__(self, other):
            return self.__sub__(other)

        def __mul__(self, other):
            if isinstance(other, (float, int)):
                return self.__class__(values=other * self.values)
            elif isinstance(other, self.__class__):
                return self.__class__(values=self.values * other.values)
            else:
                return NotImplemented

        def __rmul__(self, other):
            return self.__mul__(other)

        def __truediv__(self, other):
            if isinstance(other, self.__class__):
                return self.__class__(values=self.values / (other.values + 1e-8))
            elif isinstance(other, (float, int)):
                return self.__mul__(1 / other)
            else:
                return NotImplemented

        def __pow__(self, scalar):
            return self.__class__(values=self.values**scalar)

    def __init__(self, input_dimensions: tuple[int, ...], batch_size: int = 1):
        super().__init__(
            layer_id="simple_trainable_input",
            input_dimensions=(batch_size,),
            output_dimensions=input_dimensions,
        )

        # Initialize with small random values
        self._parameters = self.Parameters(
            values=np.random.randn(batch_size, *input_dimensions) * 0.01
        )

        self._cache = {"dP": None}

    def _forward_prop(self, input_activations: Activations) -> Activations:
        # Simply return the learned values, ignoring the input
        return Activations(self._parameters.values)

    def _backward_prop(self, dZ: D[Activations]) -> D[Activations]:
        # Gradient flows directly to the learned values
        self._cache["dP"] = d(self.Parameters(values=dZ))
        # Return zero gradient for the input (since we don't care about it)
        return d(Activations(np.zeros_like(dZ)))

    def update_parameters(self) -> None:
        if self._cache["dP"] is None:
            raise ValueError("Gradient not set during backward pass.")
        self._parameters = self._parameters + self._cache["dP"]
        self._cache["dP"] = None

    def empty_gradient(self) -> D[Parameters]:
        return d(self.Parameters(values=np.zeros_like(self._parameters.values)))

    @property
    def parameters(self):
        return self._parameters

    @property
    def cache(self):
        return self._cache

    @property
    def parameter_count(self) -> int:
        return self._parameters.values.size

    @property
    def parameter_nbytes(self) -> int:
        return self._parameters.values.nbytes

    def write_serialized_parameters(self, buffer) -> None:
        del buffer  # Unused

    def read_serialized_parameters(self, data) -> None:
        del data  # Unused


def create_one_hot_targets(num_classes: int, batch_size: int = 10) -> np.ndarray:
    """Create one-hot encoded targets for all classes."""
    targets = np.zeros((batch_size, num_classes))
    for i in range(batch_size):
        targets[i, i] = 1.0
    return targets


def visualize_inputs_with_outputs(
    input_data: np.ndarray,
    model_outputs: np.ndarray,
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

    # Create figure with subplots: one row per class, two columns (image + probabilities)
    fig, axes = plt.subplots(batch_size, 2, figsize=(12, 3 * batch_size))
    if batch_size == 1:
        axes = axes.reshape(1, -1)

    for i in range(batch_size):
        # Get the input and output for this sample
        sample_input = input_data[i]
        sample_output = model_outputs[i]

        # Left subplot: Show the reconstructed input
        if len(input_dimensions) == 3:  # Image data (channels, height, width)
            channels, height, width = input_dimensions
            if channels == 1:  # Grayscale
                img = sample_input.reshape(height, width)
                axes[i, 0].imshow(img, cmap="gray_r")
            elif channels == 3:  # RGB
                img = sample_input.reshape(channels, height, width).transpose(1, 2, 0)
                # Normalize to [0, 1] range for display
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                axes[i, 0].imshow(img)
            else:
                # For other channel counts, show first channel
                img = sample_input.reshape(channels, height, width)[0]
                axes[i, 0].imshow(img, cmap="gray_r")
        elif len(input_dimensions) == 1:  # 1D data (like flattened MNIST)
            # Try to reshape as a square image if possible
            size = input_dimensions[0]
            sqrt_size = int(np.sqrt(size))
            if sqrt_size * sqrt_size == size:
                # It's a perfect square, display as image
                img = sample_input.reshape(sqrt_size, sqrt_size)
                axes[i, 0].imshow(img, cmap="gray_r")
            else:
                # Not a perfect square, plot as 1D
                axes[i, 0].plot(sample_input)
        else:
            # For other dimensions, flatten and show as 1D
            axes[i, 0].plot(sample_input.flatten())

        axes[i, 0].set_title(f"Class {i} - Reconstructed Input")
        axes[i, 0].axis("off")

        # Right subplot: Show the model's output probabilities
        bars = axes[i, 1].bar(
            range(num_classes), sample_output, color="skyblue", alpha=0.7
        )

        # Highlight the target class (should be the highest probability)
        target_class = i
        bars[target_class].set_color("red")
        bars[target_class].set_alpha(0.8)

        axes[i, 1].set_title(f"Model Output Probabilities (Target: Class {i})")
        axes[i, 1].set_xlabel("Class")
        axes[i, 1].set_ylabel("Probability")
        axes[i, 1].set_xticks(range(num_classes))
        axes[i, 1].set_ylim(0, 1)

        # Add probability values on top of bars
        for j, (bar, prob) in enumerate(zip(bars, sample_output)):
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


def train_input_layer(
    model_path: Path,
    num_iterations: int,
    learning_rate: float,
    batch_size: int,
    regularization_strength: float = 0.01,
    verbose: bool = True,
) -> tuple[SimpleTrainableInput, Sequence[float], np.ndarray]:
    """
    Train an input layer to minimize loss for one-hot encodings of all classes.

    Args:
        model_path: Path to the frozen model file
        num_iterations: Number of training iterations
        learning_rate: Learning rate for the optimizer
        batch_size: Batch size for training (should match number of classes)
        regularization_strength: Strength of L2 regularization on input values
        verbose: Whether to print training progress

    Returns:
        Tuple of (trained_input_layer, loss_history, final_outputs)
    """

    # Load the frozen model
    logger.info(f"Loading frozen model from: {model_path}")
    with open(model_path, "rb") as f:
        frozen_model = Model.load(f, training=True, freeze_parameters=True)

    # Get model input dimensions
    input_dimensions = frozen_model.input_layer.input_dimensions
    num_classes = frozen_model.output_module.output_layer.output_dimensions[0]

    logger.info(f"Model input dimensions: {input_dimensions}")
    logger.info(
        f"Model output dimensions: {frozen_model.output_module.output_layer.output_dimensions}"
    )
    logger.info(f"Number of classes: {num_classes}")

    # Create simple trainable input layer with better initialization
    input_layer = SimpleTrainableInput(input_dimensions, batch_size)

    # Initialize with small random values that look more like real digits
    # For MNIST-like data, we want values around 0-1 range
    input_layer._parameters = input_layer.Parameters(
        values=np.random.uniform(0, 0.1, input_layer.parameters.values.shape)
    )

    logger.info(
        f"Created trainable input layer with {input_layer.parameter_count} parameters"
    )

    # Create targets (one-hot encoding for all classes)
    targets = create_one_hot_targets(num_classes, batch_size)
    logger.debug(f"Targets shape: {targets.shape}")

    # Create a combined model with trainable input and frozen rest
    combined_model = Model(
        input_dimensions=(batch_size,),
        hidden=[input_layer] + list(frozen_model.hidden_modules),
        output=frozen_model.output_module,
    )

    # Create optimizer with learning rate scheduling
    scheduler = ConstantScheduler(learning_rate=learning_rate)
    optimizer = AdaM(
        model=combined_model,
        config=AdaM.Config(scheduler=scheduler),
    )

    # Training loop
    loss_history = []
    best_loss = float("inf")
    patience_counter = 0
    patience = 200  # Early stopping patience

    logger.info(f"Starting training for {num_iterations} iterations...")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Regularization strength: {regularization_strength}")
    logger.info(f"Training all {num_classes} classes simultaneously")
    logger.info("-" * 50)

    for iteration in range(num_iterations):
        # Forward pass: generate input and get model output
        dummy_input = np.ones((batch_size,))  # Dummy input for the trainable layer
        model_output = combined_model.forward_prop(dummy_input)

        # Compute classification loss
        classification_loss = combined_model.compute_loss(dummy_input, targets)

        # Compute regularization loss (L2 penalty on input values)
        input_values = input_layer.parameters.values
        regularization_loss = (
            regularization_strength * np.sum(input_values**2) / batch_size
        )

        # Add sparsity regularization to encourage more realistic digit-like patterns
        sparsity_loss = 0.001 * np.sum(np.abs(input_values)) / batch_size

        # Total loss
        total_loss = classification_loss + regularization_loss + sparsity_loss
        loss_history.append(total_loss)

        # Backward pass - gradients will flow back to the trainable input layer
        combined_model.backward_prop(Y_true=targets)

        # Add regularization gradients to the input layer's gradient cache
        if input_layer.cache["dP"] is not None:
            # Add gradient of regularization term: 2 * regularization_strength * input_values / batch_size
            reg_gradient = 2 * regularization_strength * input_values / batch_size
            # Add gradient of sparsity term: 0.001 * sign(input_values) / batch_size
            sparsity_gradient = 0.001 * np.sign(input_values) / batch_size
            current_gradient = input_layer.cache["dP"].values
            total_gradient = current_gradient + reg_gradient + sparsity_gradient

            # Gradient clipping to prevent exploding gradients
            grad_norm = np.linalg.norm(total_gradient)
            if grad_norm > 1.0:
                total_gradient = total_gradient / grad_norm

            input_layer.cache["dP"] = d(input_layer.Parameters(values=total_gradient))

        # Update parameters (only the input layer will be updated since others are frozen)
        optimizer.compute_update()
        combined_model.update_parameters()

        # Clip input values to reasonable range (0-1 for MNIST-like data)
        input_values = np.clip(input_layer.parameters.values, 0, 1)
        input_layer._parameters = input_layer.Parameters(values=input_values)

        # Early stopping
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

        # Print progress
        if verbose and (iteration % 100 == 0 or iteration == num_iterations - 1):
            input_norm = np.linalg.norm(input_values)
            max_input = np.max(input_values)
            min_input = np.min(input_values)
            logger.info(
                f"Iteration {iteration:4d}: Loss = {total_loss:.6f} (Class: {classification_loss:.6f}, Reg: {regularization_loss:.6f}, Sparsity: {sparsity_loss:.6f}, Input Norm: {input_norm:.6f}, Range: [{min_input:.3f}, {max_input:.3f}])"
            )

    logger.info("-" * 50)
    logger.info(f"Training completed. Final loss: {loss_history[-1]:.6f}")

    # Get final outputs
    dummy_input = np.ones((batch_size,))
    final_outputs = combined_model.forward_prop(dummy_input)

    return input_layer, loss_history, final_outputs


def main():
    parser = argparse.ArgumentParser(
        description="Train an input layer to minimize loss for a one-hot encoding of class 0"
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
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for training (default: 10)",
    )
    parser.add_argument(
        "--output", type=Path, help="Path to save the trained input layer (optional)"
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

    # Configure logging
    logger.remove()
    log_level = "DEBUG" if args.quiet else args.log_level
    logger.add(sys.stderr, level=log_level)

    # Check if model file exists
    if not args.model_path.exists():
        logger.error(f"Model file not found: {args.model_path}")
        sys.exit(1)

    try:
        # Train the input layer
        input_layer, loss_history, final_outputs = train_input_layer(
            model_path=args.model_path,
            num_iterations=args.iterations,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            regularization_strength=args.regularization,
            verbose=not args.quiet,
        )

        # Save the trained input layer if requested
        if args.output:
            logger.info(f"Saving trained input layer to: {args.output}")
            np.savez(
                args.output,
                values=input_layer.parameters.values,
                input_dimensions=input_layer.input_dimensions,
                output_dimensions=input_layer.output_dimensions,
            )

        # Print final results
        logger.info("Final input layer parameters:")
        logger.info(f"  Values shape: {input_layer.parameters.values.shape}")
        logger.info(
            f"  Values norm: {np.linalg.norm(input_layer.parameters.values):.6f}"
        )

        # Generate final input and show model output
        dummy_input = np.ones((args.batch_size,))
        final_input = input_layer.forward_prop(dummy_input)
        logger.info(f"Final generated input shape: {final_input.shape}")
        logger.info(f"Input norm: {np.linalg.norm(final_input):.6f}")

        # Get the model's output probabilities for the final inputs
        logger.info(f"Final model outputs shape: {final_outputs.shape}")

        # Print confidence scores for each class
        for i in range(args.batch_size):
            target_prob = final_outputs[i, i]
            logger.info(f"Class {i} confidence: {target_prob:.6f}")

        # Visualize the reconstructed inputs and outputs
        logger.info("Visualizing reconstructed inputs and outputs...")
        visualize_inputs_with_outputs(
            final_input,
            final_outputs,
            input_layer.output_dimensions,
            save_path=args.visualization,
        )

    except Exception as e:
        logger.exception(f"Error during training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
