from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np

from mnist_numpy.model.mlp import MultiLayerPerceptron


@click.command()
@click.argument("model_path", type=Path)
def main(model_path: Path):
    model = MultiLayerPerceptron.load(model_path.open("rb"))
    weights = np.concatenate(
        tuple(layer._parameters._W.flatten() for layer in model.non_input_layers)
    )
    biases = np.concatenate(
        tuple(layer._parameters._B for layer in model.non_input_layers)
    )

    weights_array = np.abs(np.array(list(weights)))
    biases_array = np.abs(np.array(list(biases)))

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    w_min, w_max = weights_array.min(), weights_array.max()
    w_bins = min(50, max(10, int(np.sqrt(len(weights_array)))))
    ax1.hist(weights_array, bins=w_bins, alpha=0.7, color="blue", log=True)
    ax1.set_title(f"Weights Distribution (absolute values, n={len(weights_array)})")
    ax1.set_xlabel(f"Absolute Value [min={w_min:.4f}, max={w_max:.4f}]")
    ax1.set_ylabel("Log Frequency")
    ax1.set_xscale("log")
    ax1.grid(alpha=0.3)

    b_min, b_max = biases_array.min(), biases_array.max()
    b_bins = min(50, max(10, int(np.sqrt(len(biases_array)))))
    ax2.hist(biases_array, bins=b_bins, alpha=0.7, color="green", log=True)
    ax2.set_title(f"Biases Distribution (absolute values, n={len(biases_array)})")
    ax2.set_xlabel(f"Absolute Value [min={b_min:.4f}, max={b_max:.4f}]")
    ax2.set_ylabel("Log Frequency")
    ax2.set_xscale("log")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
