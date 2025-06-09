import re
import signal
import sys
from pathlib import Path

import click
import h5py
import matplotlib.pyplot as plt
import numpy as np
from InquirerPy import inquirer
from loguru import logger

from mo_net.data import DATA_DIR


def print_group_statistics(group: h5py.Group, prefix: str = "") -> None:
    """Print statistics for a given HDF5 group recursively."""
    for key, item in group.items():
        path = f"{prefix}/{key}" if prefix else key

        if isinstance(item, h5py.Group):
            if all(attr in item.attrs for attr in ["mean", "std", "min", "max"]):
                logger.info(f"{path} statistics:")
                logger.info(f"  Mean: {item.attrs['mean']:.6f}")
                logger.info(f"  Std: {item.attrs['std']:.6f}")
                logger.info(f"  Min: {item.attrs['min']:.6f}")
                logger.info(f"  Max: {item.attrs['max']:.6f}")
            if "deciles" in item.attrs:
                logger.info(f"{path} deciles: {item.attrs['deciles']}")

            print_group_statistics(item, path)


def plot_histograms(file: h5py.File, iteration_key: str) -> None:
    """Plot histograms for the given iteration with related parameters grouped by rows."""
    plt.style.use("dark_background")

    iteration_group = file[iteration_key]

    layer_count = 0

    if "weights" in iteration_group:
        for layer_key in iteration_group["weights"]:
            if isinstance(iteration_group["weights"][layer_key], h5py.Group):
                layer_num = int(layer_key.split("_")[1])
                layer_count = max(layer_count, layer_num + 1)

    if "gradients" in iteration_group:
        for layer_key in iteration_group["gradients"]:
            if isinstance(iteration_group["gradients"][layer_key], h5py.Group):
                layer_num = int(layer_key.split("_")[1])
                layer_count = max(layer_count, layer_num + 1)

    if "updates" in iteration_group:
        if "weights" in iteration_group["updates"]:
            for layer_key in iteration_group["updates"]["weights"]:
                if isinstance(
                    iteration_group["updates"]["weights"][layer_key], h5py.Group
                ):
                    layer_num = int(layer_key.split("_")[1])
                    layer_count = max(layer_count, layer_num + 1)
        if "biases" in iteration_group["updates"]:
            for layer_key in iteration_group["updates"]["biases"]:
                if isinstance(
                    iteration_group["updates"]["biases"][layer_key], h5py.Group
                ):
                    layer_num = int(layer_key.split("_")[1])
                    layer_count = max(layer_count, layer_num + 1)

    has_weights = "weights" in iteration_group
    has_biases = "biases" in iteration_group
    has_raw_gradients = "raw_gradients" in iteration_group
    has_activations = "activations" in iteration_group
    has_updates = "updates" in iteration_group
    n_rows = layer_count
    n_cols = 7  # weights, biases, weight gradients, bias gradients, weight updates, bias updates, activations

    row_height = 2.7  # Height in inches per row
    fig = plt.figure(figsize=(16, row_height * n_rows + 0.8))
    fig.suptitle(f"Histograms for {iteration_key}", fontsize=16, y=0.99)

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
        }
    )

    gs = fig.add_gridspec(
        nrows=layer_count,
        ncols=n_cols,
        hspace=1.0,
        wspace=0.4,
        top=0.94,
        bottom=0.04,
        left=0.04,
        right=0.96,
    )

    for layer_idx in range(layer_count):
        layer_key = f"layer_{layer_idx}"

        # Plot weights (column 1)
        if has_weights and layer_key in iteration_group["weights"]:
            layer = iteration_group["weights"][layer_key]
            if "histogram_values" in layer and "histogram_bins" in layer:
                ax = fig.add_subplot(gs[layer_idx, 0])
                values = layer["histogram_values"][()]
                bins = layer["histogram_bins"][()]
                ax.bar(bins[:-1], values, width=np.diff(bins), alpha=0.7, color="red")
                ax.set_title(f"Weights L{layer_idx}", fontsize=8)
                ax.set_yscale("log")
                ax.set_xlabel("Value")
                ax.set_ylabel("Count (log)")
                ax.grid(alpha=0.3)

        # Plot biases (column 2)
        if has_biases and layer_key in iteration_group["biases"]:
            layer = iteration_group["biases"][layer_key]
            if "histogram_values" in layer and "histogram_bins" in layer:
                ax = fig.add_subplot(gs[layer_idx, 1])
                values = layer["histogram_values"][()]
                bins = layer["histogram_bins"][()]
                ax.bar(
                    bins[:-1], values, width=np.diff(bins), alpha=0.7, color="orange"
                )
                ax.set_title(f"Biases L{layer_idx}", fontsize=8)
                ax.set_yscale("log")
                ax.set_xlabel("Value")
                ax.set_ylabel("Count (log)")
                ax.grid(alpha=0.3)

        # Plot weight gradients (column 3)
        if has_raw_gradients and layer_key in iteration_group["raw_gradients"]:
            gradient_layer = iteration_group["raw_gradients"][layer_key]
            if (
                "weights" in gradient_layer
                and "histogram_values" in gradient_layer["weights"]
            ):
                ax = fig.add_subplot(gs[layer_idx, 2])
                values = gradient_layer["weights"]["histogram_values"][()]
                bins = gradient_layer["weights"]["histogram_bins"][()]
                ax.bar(
                    bins[:-1], values, width=np.diff(bins), alpha=0.7, color="yellow"
                )
                ax.set_title(f"Weight Gradients L{layer_idx}", fontsize=8)
                ax.set_yscale("log")
                ax.set_xlabel("Value")
                ax.set_ylabel("Count (log)")
                ax.grid(alpha=0.3)

        # Plot bias gradients (column 4)
        if has_raw_gradients and layer_key in iteration_group["raw_gradients"]:
            gradient_layer = iteration_group["raw_gradients"][layer_key]
            if (
                "biases" in gradient_layer
                and "histogram_values" in gradient_layer["biases"]
            ):
                ax = fig.add_subplot(gs[layer_idx, 3])
                values = gradient_layer["biases"]["histogram_values"][()]
                bins = gradient_layer["biases"]["histogram_bins"][()]
                ax.bar(bins[:-1], values, width=np.diff(bins), alpha=0.7, color="green")
                ax.set_title(f"Bias Gradients L{layer_idx}", fontsize=8)
                ax.set_yscale("log")
                ax.set_xlabel("Value")
                ax.set_ylabel("Count (log)")
                ax.grid(alpha=0.3)

        # Plot weight updates (column 5)
        if has_updates and "weights" in iteration_group["updates"]:
            weights_update_group = iteration_group["updates"]["weights"]
            if layer_key in weights_update_group:
                update_layer = weights_update_group[layer_key]
                if (
                    "histogram_values" in update_layer
                    and "histogram_bins" in update_layer
                ):
                    ax = fig.add_subplot(gs[layer_idx, 4])
                    values = update_layer["histogram_values"][()]
                    bins = update_layer["histogram_bins"][()]
                    ax.bar(
                        bins[:-1], values, width=np.diff(bins), alpha=0.7, color="blue"
                    )
                    ax.set_title(f"Weight Updates L{layer_idx}", fontsize=8)
                    ax.set_yscale("log")
                    ax.set_xlabel("Value")
                    ax.set_ylabel("Count (log)")
                    ax.grid(alpha=0.3)

        # Plot bias updates (column 6)
        if has_updates and "biases" in iteration_group["updates"]:
            biases_update_group = iteration_group["updates"]["biases"]
            if layer_key in biases_update_group:
                update_layer = biases_update_group[layer_key]
                if (
                    "histogram_values" in update_layer
                    and "histogram_bins" in update_layer
                ):
                    ax = fig.add_subplot(gs[layer_idx, 5])
                    values = update_layer["histogram_values"][()]
                    bins = update_layer["histogram_bins"][()]
                    ax.bar(
                        bins[:-1],
                        values,
                        width=np.diff(bins),
                        alpha=0.7,
                        color="violet",
                    )
                    ax.set_title(f"Bias Updates L{layer_idx}", fontsize=8)
                    ax.set_yscale("log")
                    ax.set_xlabel("Value")
                    ax.set_ylabel("Count (log)")
                    ax.grid(alpha=0.3)

        # Plot activations (column 7)
        if has_activations and layer_key in iteration_group["activations"]:
            activation_layer = iteration_group["activations"][layer_key]
            if (
                "histogram_values" in activation_layer
                and "histogram_bins" in activation_layer
            ):
                ax = fig.add_subplot(gs[layer_idx, 6])
                values = activation_layer["histogram_values"][()]
                bins = activation_layer["histogram_bins"][()]
                ax.bar(
                    bins[:-1], values, width=np.diff(bins), alpha=0.7, color="indigo"
                )
                ax.set_title(f"Activations L{layer_idx}", fontsize=8)
                ax.set_xlabel("Value")
                ax.set_yscale("log")
                ax.set_ylabel("Count")
                ax.grid(alpha=0.3)

    legend_entries = [
        plt.Rectangle((0, 0), 1, 1, color="red", alpha=0.7),
        plt.Rectangle((0, 0), 1, 1, color="orange", alpha=0.7),
        plt.Rectangle((0, 0), 1, 1, color="yellow", alpha=0.7),
        plt.Rectangle((0, 0), 1, 1, color="green", alpha=0.7),
        plt.Rectangle((0, 0), 1, 1, color="blue", alpha=0.7),
        plt.Rectangle((0, 0), 1, 1, color="violet", alpha=0.7),
        plt.Rectangle((0, 0), 1, 1, color="indigo", alpha=0.7),
    ]
    legend_labels = [
        "Weights",
        "Biases",
        "Weight Raw Gradients",
        "Bias Raw Gradients",
        "Weight Updates",
        "Bias Updates",
        "Activations",
    ]

    for ax in fig.get_axes():
        ax.legend(legend_entries, legend_labels, loc="upper right")
        break

    plt.subplots_adjust(hspace=0.8, wspace=0.4)

    plt.show()


def extract_iteration_number(key: str) -> int:
    """Extract the numeric iteration value from the iteration key."""
    match = re.match(r"iteration_(\d+)", key)
    if match:
        return int(match.group(1))
    raise ValueError(f"Invalid iteration key: {key}")


def sigint_handler(sig, frame):
    """Handle SIGINT (Ctrl+C) by asking for confirmation to exit."""
    confirm = inquirer.confirm(
        message="Do you want to exit?",
        default=False,
    ).execute()

    if confirm:
        logger.info("Exiting...")
        sys.exit(0)
    else:
        logger.info("Continuing...")
        return


@click.command()
@click.option("--trace_log_path", "-t", type=Path, help="Path to the trace log file")
@click.option("--iteration", "-i", type=int, help="Specific iteration to analyze")
@click.option(
    "--list-iterations", "-l", is_flag=True, help="List all available iterations"
)
def main(
    trace_log_path: Path | None = None,
    iteration: int | None = None,
    list_iterations: bool = False,
) -> None:
    signal.signal(signal.SIGINT, sigint_handler)

    if trace_log_path is None:
        run_dir = DATA_DIR / "run"
        hdf5_files = tuple(run_dir.glob("*.hdf5"))
        if not hdf5_files:
            logger.error(f"No .hdf5 files found in {run_dir}")
            sys.exit(1)
        trace_log_path = max(hdf5_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Using latest trace log file: {trace_log_path}")
    if not trace_log_path.exists():
        logger.error(f"File not found: {trace_log_path}")
        sys.exit(1)

    with h5py.File(trace_log_path, "r") as f:
        available_iterations = sorted(
            (k for k in f.keys() if k.startswith("iteration_")),
            key=extract_iteration_number,
        )

        if not available_iterations:
            logger.error("No iterations found in the trace log file.")
            sys.exit(1)

        if list_iterations:
            logger.info(f"Available iterations in {trace_log_path}:")
            for iter_key in available_iterations:
                timestamp = f[iter_key].attrs.get("timestamp", "N/A")
                logger.info(f"  {iter_key} (timestamp: {timestamp})")
            return

        while True:
            if iteration is None:
                selection = inquirer.fuzzy(
                    message="Available iterations:",
                    choices=[
                        {
                            "name": f"[{idx}]: {iteration}",
                            "value": idx,
                        }
                        for idx, iteration in enumerate(available_iterations)
                    ],
                ).execute()

                try:
                    index = int(selection)
                    if 0 <= index < len(available_iterations):
                        iteration_key = available_iterations[index]
                    else:
                        logger.error(f"Invalid selection: {selection}")
                        continue
                except ValueError:
                    logger.error(f"Invalid input: {selection}")
                    continue
            else:
                iteration_key = f"iteration_{iteration}"
                if iteration_key not in f:
                    logger.error(f"Iteration {iteration} not found in trace log.")
                    sys.exit(1)

            logger.info(f"Statistics for {iteration_key}:")
            print_group_statistics(f[iteration_key])

            plot_histograms(f, iteration_key)

            iteration = None


if __name__ == "__main__":
    main()
