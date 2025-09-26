from __future__ import annotations

import contextlib
import pickle
import tempfile
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import IO, assert_never

import click
import jax
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
from loguru import logger

from mo_net.data import DATA_DIR, SplitConfig, load_data
from mo_net.functions import get_activation_fn, mse_loss
from mo_net.log import LogLevel, setup_logging
from mo_net.model.layer.activation import Activation
from mo_net.model.layer.linear import Linear
from mo_net.model.layer.output import MseOutputLayer
from mo_net.model.model import Model
from mo_net.model.module.base import Output
from mo_net.model.module.dense import Dense
from mo_net.protos import NormalisationType, SupportsDeserialize
from mo_net.regulariser.l1_regulariser import L1Regulariser
from mo_net.resources import MNIST_TEST_URL, MNIST_TRAIN_URL
from mo_net.train import TrainingParameters
from mo_net.train.backends.log import SqliteBackend
from mo_net.train.run import TrainingRun
from mo_net.train.trainer.trainer import (
    BasicTrainer,
    TrainingFailed,
    TrainingSuccessful,
    get_optimiser,
)


def save_and_show_plot(fig, output_path: Path | None, show: bool) -> None:
    if output_path:
        html_output = output_path.with_suffix(".html")
    else:
        tmp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False)
        html_output = Path(tmp_file.name)

    fig.write_html(str(html_output))

    if show:
        webbrowser.open(f"file://{Path(html_output).absolute()}")


class MLPDecoderModel(Model):
    @dataclass(frozen=True, kw_only=True)
    class Serialized:
        input_dimensions: tuple[int, ...]
        hidden_modules: tuple[SupportsDeserialize, ...]
        output_module: SupportsDeserialize

    @classmethod
    def get_name(cls) -> str:
        return "mlp_decoder"

    @classmethod
    def get_description(cls) -> str:
        return "MLP Decoder for reconstruction"

    @classmethod
    def create(
        cls,
        *,
        input_size: int,
        hidden_sizes: tuple[int, ...],
        output_size: int,
        key: jnp.ndarray,
    ) -> MLPDecoderModel:
        hidden_modules = []
        current_size = input_size

        for hidden_size in hidden_sizes:
            key, subkey = jax.random.split(key)
            hidden_modules.append(
                Dense(
                    input_dimensions=(current_size,),
                    output_dimensions=(hidden_size,),
                    activation_fn=get_activation_fn("relu"),
                    key=subkey,
                    store_output_activations=False,
                )
            )
            current_size = hidden_size

        key, subkey = jax.random.split(key)

        return cls(
            input_dimensions=(input_size,),
            hidden=tuple(hidden_modules),
            output=Output(
                layers=(
                    Linear(
                        input_dimensions=(current_size,),
                        output_dimensions=(output_size,),
                        parameters_init_fn=lambda input_dims,
                        output_dims: Linear.Parameters.xavier(
                            input_dims, output_dims, key=subkey
                        ),
                        store_output_activations=False,
                    ),
                    Activation(
                        input_dimensions=(output_size,),
                        activation_fn=get_activation_fn("sigmoid"),
                    ),
                ),
                output_layer=MseOutputLayer(input_dimensions=(output_size,)),
            ),
        )

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
    ) -> "MLPDecoderModel":
        if isinstance(source, Path):
            with open(source, "rb") as f:
                serialized = pickle.load(f)
        else:
            serialized = pickle.load(source)
        if (
            not hasattr(serialized, "input_dimensions")
            or not hasattr(serialized, "hidden_modules")
            or not hasattr(serialized, "output_module")
        ):
            raise ValueError(f"Invalid serialized model: {type(serialized)}")
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


def train_decoder(
    encoder_model: Model,
    layer_index: str,
    X_train: jnp.ndarray,
    Y_train: jnp.ndarray,
    X_val: jnp.ndarray,
    output_path: Path,
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    l1_lambda: float = 0.0,
) -> MLPDecoderModel:
    logger.info(f"Extracting activations from layer {layer_index}...")
    train_activations = encoder_model.forward_prop_to(X_train, layer_index)
    val_activations = encoder_model.forward_prop_to(X_val, layer_index)

    logger.info(f"Activation shape: {train_activations.shape}")
    logger.info(f"Target shape: {Y_train.shape}")

    decoder = MLPDecoderModel.create(
        input_size=train_activations.shape[1],
        hidden_sizes=(128, 128),
        output_size=784,
        key=jax.random.PRNGKey(42),
    )

    training_parameters = TrainingParameters(
        batch_size=batch_size,
        dropout_keep_probs=(),
        history_max_len=10,
        learning_rate_limits=(learning_rate, learning_rate),
        log_level="INFO",
        max_restarts=0,
        monotonic=False,
        no_monitoring=True,
        normalisation_type=NormalisationType.NONE,
        num_epochs=num_epochs,
        quiet=False,
        regulariser_lambda=1e-4,
        seed=42,
        trace_logging=False,
        train_set_size=len(train_activations),
        warmup_epochs=0,
        workers=0,
    )

    optimiser = get_optimiser("adam", decoder, training_parameters)

    if l1_lambda > 0:
        L1Regulariser.attach(
            lambda_=l1_lambda,
            batch_size=batch_size,
            optimiser=optimiser,
            model=decoder,
            key=jax.random.PRNGKey(42),
        )

    trainer = BasicTrainer(
        X_train=train_activations,
        X_val=val_activations,
        Y_train=X_train,  # Original images as target
        Y_val=X_val,  # Original images as target
        key=jax.random.PRNGKey(42),
        transform_fn=None,
        loss_fn=mse_loss,
        model=decoder,
        optimiser=optimiser,
        run=TrainingRun(
            seed=42,
            name=f"decoder_layer_{layer_index}",
            backend=SqliteBackend(),
        ),
        training_parameters=training_parameters,
        output_path=output_path,
        monotonic=False,
    )

    logger.info("Training decoder...")
    result = trainer.train()

    match result:
        case TrainingSuccessful():
            logger.info(
                f"Decoder training completed. Model saved to: {result.model_checkpoint_path}"
            )
            decoder.dump(output_path)
            return decoder
        case TrainingFailed():
            raise RuntimeError(f"Decoder training failed: {result.message}")
        case never:
            assert_never(never)


@click.group()
def cli():
    pass


@cli.command("train", help="Train a decoder for a specific layer")
@click.argument("model_path", type=Path)
@click.argument("layer_number", type=str)
@click.option(
    "--output-path", "-o", type=Path, help="Output path for the decoder model"
)
@click.option("--num-epochs", type=int, default=10, help="Number of training epochs")
@click.option("--batch-size", type=int, default=32, help="Batch size")
@click.option("--learning-rate", type=float, default=1e-3, help="Learning rate")
@click.option("--l1-lambda", type=float, default=0.0, help="l1 regularisation strength")
@click.option("--log-level", type=LogLevel, default=LogLevel.INFO, help="Log level")
def train(
    model_path: Path,
    layer_number: str,
    output_path: Path | None,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    l1_lambda: float,
    log_level: LogLevel,
):
    setup_logging(log_level)

    logger.info(f"Loading model from {model_path}")
    encoder_model = Model.load(model_path, training=False, freeze_parameters=True)

    if len(encoder_model.hidden_modules) == 0:
        raise ValueError("Model has no hidden modules")

    logger.info("Loading MNIST data...")
    X_train, _, X_val, _ = load_data(
        MNIST_TRAIN_URL,
        split=SplitConfig.of(0.8, 0),
        one_hot=True,
    )

    X_train = X_train.reshape(-1, 28 * 28)
    X_val = X_val.reshape(-1, 28 * 28)

    if output_path is None:
        output_path = (
            DATA_DIR / "output" / f"decoder_{model_path.stem}_layer_{layer_number}.pkl"
        )
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    train_decoder(
        encoder_model=encoder_model,
        layer_index=layer_number,
        X_train=X_train,
        Y_train=X_train,  # Target should be the original images, not labels!
        X_val=X_val,
        output_path=output_path,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        l1_lambda=l1_lambda,
    )

    logger.info("Decoder training completed successfully!")
    logger.info(f"Decoder model saved to: {output_path}")


@cli.command("sample", help="Show input and reconstruction side by side")
@click.argument("encoder_model_path", type=Path)
@click.argument("decoder_model_path", type=Path)
@click.argument("layer_number", type=str)
@click.option("--output-path", "-o", type=Path, help="Output path for the sample plot")
@click.option(
    "--no-show", is_flag=True, default=False, help="Don't open plot in browser"
)
@click.option("--log-level", type=LogLevel, default=LogLevel.INFO, help="Log level")
def sample(
    encoder_model_path: Path,
    decoder_model_path: Path,
    layer_number: str,
    output_path: Path | None,
    no_show: bool,
    log_level: LogLevel,
):
    setup_logging(log_level)

    logger.info(f"Loading encoder model from {encoder_model_path}")
    encoder_model = Model.load(encoder_model_path, training=False)

    logger.info(f"Loading decoder model from {decoder_model_path}")
    decoder_model = MLPDecoderModel.load(decoder_model_path, training=False)

    X_test, _, _, _ = load_data(
        MNIST_TEST_URL,
        split=SplitConfig.of(0.1, 0),
        one_hot=True,
    )
    X_test = X_test.reshape(-1, 28 * 28)

    sample_images = X_test[
        np.random.choice(len(X_test), size=min(16, len(X_test)), replace=False)
    ]
    sample_activations = encoder_model.forward_prop_to(sample_images, layer_number)
    logger.info(f"Sample activations shape: {sample_activations.shape}")
    logger.info(
        f"Sample activations range: [{sample_activations.min():.4f}, {sample_activations.max():.4f}]"
    )

    reconstructions = decoder_model.forward_prop(sample_activations)
    logger.info(f"Reconstructions shape: {reconstructions.shape}")
    logger.info(
        f"Reconstructions range: [{reconstructions.min():.4f}, {reconstructions.max():.4f}]"
    )
    logger.info(
        f"Sample images range: [{sample_images.min():.4f}, {sample_images.max():.4f}]"
    )

    fig = sp.make_subplots(
        rows=len(sample_images),
        cols=2,
        subplot_titles=[
            f"Input {i}" if j == 0 else f"Reconstruction {i}"
            for i in range(len(sample_images))
            for j in range(2)
        ],
        specs=[
            [{"type": "heatmap"}, {"type": "heatmap"}]
            for _ in range(len(sample_images))
        ],
    )

    for i in range(len(sample_images)):
        fig.add_trace(
            go.Heatmap(
                z=sample_images[i].reshape(28, 28),
                colorscale="gray",
                showscale=False,
                name=f"Input {i}",
            ),
            row=i + 1,
            col=1,
        )
        fig.add_trace(
            go.Heatmap(
                z=reconstructions[i].flatten().reshape(28, 28),
                colorscale="gray",
                showscale=False,
                name=f"Reconstruction {i}",
            ),
            row=i + 1,
            col=2,
        )

    fig.update_layout(
        title="Input Images vs Reconstructions",
        height=200 * len(sample_images),
        width=800,
        showlegend=False,
    )

    for i in range(len(sample_images)):
        fig.update_xaxes(scaleanchor=f"y{i * 2 + 1}", scaleratio=1, row=i + 1, col=1)
        fig.update_yaxes(scaleanchor=f"x{i * 2 + 1}", scaleratio=1, row=i + 1, col=1)
        fig.update_xaxes(scaleanchor=f"y{i * 2 + 2}", scaleratio=1, row=i + 1, col=2)
        fig.update_yaxes(scaleanchor=f"x{i * 2 + 2}", scaleratio=1, row=i + 1, col=2)

    save_and_show_plot(fig, output_path, not no_show)


@cli.command("reconstruct", help="Reconstruct inputs from decoder activations")
@click.argument("encoder_model_path", type=Path)
@click.argument("decoder_model_path", type=Path)
@click.argument("layer_number", type=str)
@click.option(
    "--output-path", "-o", type=Path, help="Output path for the reconstruction plot"
)
@click.option(
    "--no-show", is_flag=True, default=False, help="Don't open plot in browser"
)
@click.option("--log-level", type=LogLevel, default=LogLevel.INFO, help="Log level")
def reconstruct(
    encoder_model_path: Path,
    decoder_model_path: Path,
    layer_number: str,
    output_path: Path | None,
    no_show: bool,
    log_level: LogLevel,
):
    setup_logging(log_level)

    logger.info(f"Loading encoder model from {encoder_model_path}")
    encoder_model = Model.load(encoder_model_path, training=False)

    logger.info(f"Loading decoder model from {decoder_model_path}")
    decoder_model = MLPDecoderModel.load(decoder_model_path, training=False)

    X_sample, _, _, _ = load_data(
        MNIST_TRAIN_URL,
        split=SplitConfig.of(0.1, 0),
        one_hot=True,
    )
    X_sample = X_sample.reshape(-1, 28 * 28)

    sample_activations = encoder_model.forward_prop_to(X_sample[:100], layer_number)
    activation_size = sample_activations.shape[1]

    reconstructions = np.array(
        [
            decoder_model.forward_prop(
                jnp.zeros_like(sample_activations[0])
                .at[i]
                .set(sample_activations[0][i])
            )
            for i in range(min(64, activation_size))
        ]
    )

    if reconstructions.ndim == 3 and reconstructions.shape[1] == 1:
        reconstructions = reconstructions.squeeze(1)

    if (current_avg := np.mean(reconstructions)) > 0:
        reconstructions = reconstructions * (0.5 / current_avg)

    first_linear = next(
        (
            layer
            for module in decoder_model.modules
            for layer in module.layers
            if hasattr(layer, "parameters") and hasattr(layer.parameters, "weights")
        ),
        None,
    )

    fig = sp.make_subplots(
        rows=len(reconstructions),
        cols=3,
        subplot_titles=[
            f"Unit {i} - Reconstruction"
            if j == 0
            else f"Unit {i} - Weight Distribution"
            if j == 1
            else f"Unit {i} - Weight Map"
            for i in range(len(reconstructions))
            for j in range(3)
        ],
        specs=[
            [{"type": "heatmap"}, {"type": "histogram"}, {"type": "heatmap"}]
            for _ in range(len(reconstructions))
        ],
    )

    for i in range(len(reconstructions)):
        fig.add_trace(
            go.Heatmap(
                z=reconstructions[i].flatten().reshape(28, 28),
                colorscale="gray",
                showscale=False,
                name=f"Unit {i} Reconstruction",
            ),
            row=i + 1,
            col=1,
        )

        if first_linear:
            fig.add_trace(
                go.Histogram(
                    x=first_linear.parameters.weights[i].flatten(),
                    nbinsx=20,
                    name=f"Unit {i} Weights",
                    showlegend=False,
                ),
                row=i + 1,
                col=2,
            )

            fig.add_trace(
                go.Heatmap(
                    z=np.tile(
                        first_linear.parameters.weights[i].reshape(-1, 1), (1, 10)
                    ),
                    colorscale="RdBu",
                    showscale=False,
                    name=f"Unit {i} Weight Map",
                ),
                row=i + 1,
                col=3,
            )
        else:
            fig.add_annotation(
                text="No weights found",
                xref=f"x{i * 3 + 2}",
                yref=f"y{i * 3 + 2}",
                x=0.5,
                y=0.5,
                showarrow=False,
                row=i + 1,
                col=2,
            )
            fig.add_annotation(
                text="No weights found",
                xref=f"x{i * 3 + 3}",
                yref=f"y{i * 3 + 3}",
                x=0.5,
                y=0.5,
                showarrow=False,
                row=i + 1,
                col=3,
            )

    fig.update_layout(
        title=f"Decoder Reconstructions and Weight Distributions from Layer {layer_number}",
        height=200 * len(reconstructions),
        width=1800,
        showlegend=False,
    )

    for i in range(len(reconstructions)):
        fig.update_xaxes(scaleanchor=f"y{i * 3 + 1}", scaleratio=1, row=i + 1, col=1)
        fig.update_yaxes(scaleanchor=f"x{i * 3 + 1}", scaleratio=1, row=i + 1, col=1)
        fig.update_xaxes(title_text="Weight Value", row=i + 1, col=2)
        fig.update_yaxes(title_text="Count", row=i + 1, col=2)
        fig.update_xaxes(title_text="Weight Magnitude", row=i + 1, col=3)
        fig.update_yaxes(title_text="Input Pixel", row=i + 1, col=3)

    save_and_show_plot(fig, output_path, not no_show)


if __name__ == "__main__":
    cli()
