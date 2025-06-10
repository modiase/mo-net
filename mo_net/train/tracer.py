from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

from mo_net.model.layer.linear import Linear, Parameters
from mo_net.model.model import Model
from mo_net.protos import RawGradientType, UpdateGradientType


class TracerStrategy(ABC):
    @abstractmethod
    def should_trace(self, iteration: int) -> bool: ...


class PerEpochTracerStrategy(TracerStrategy):
    def __init__(self, *, training_set_size: int, batch_size: int):
        self._training_set_size = training_set_size
        self._batch_size = batch_size

    def should_trace(self, iteration: int) -> bool:
        return iteration % (self._training_set_size / self._batch_size) == 0


class PerStepTracerStrategy(TracerStrategy):
    def should_trace(self, iteration: int) -> bool:
        return True


class SampleTracerStrategy(TracerStrategy):
    def __init__(self, *, sample_rate: float):
        if sample_rate < 0 or sample_rate > 1:
            raise ValueError("Sample rate must be between 0 and 1")
        self._sample_rate = sample_rate

    def should_trace(self, iteration: int) -> bool:
        del iteration  # unused
        return np.random.rand() < self._sample_rate


@dataclass(kw_only=True, frozen=True)
class TracerConfig:
    trace_activations: bool = True
    trace_biases: bool = True
    trace_raw_gradients: bool = True
    trace_updates: bool = True
    trace_strategy: TracerStrategy
    trace_weights: bool = True


class Tracer:
    def __init__(
        self,
        *,
        run_id: str,
        model: Model,
        tracer_config: TracerConfig,
    ):
        self.model = model
        self._linear_layers: Sequence[Linear] = (
            tuple(  # TODO: Consider reducing coupling to dense layers.
                layer
                for block in model.blocks
                for layer in block.layers
                if isinstance(layer, Linear)
            )
        )
        self._trace_logging_path = Path(f"trace_log_{run_id}.hdf5")
        self._tracer_config = tracer_config
        self._iterations = 0

        with h5py.File(self._trace_logging_path, "w") as f:
            f.create_group("weights")
            f.create_group("biases")
            f.create_group("raw_gradients")
            f.create_group("updates")
            f.attrs["layer_count"] = len(self._linear_layers)
            f.attrs["iterations"] = 0

    def post_batch(
        self,
        raw_gradient: RawGradientType,
        update: UpdateGradientType,
    ) -> None:
        if not self._tracer_config.trace_strategy.should_trace(self._iterations):
            self._iterations += 1
            return

        activations: Sequence[np.ndarray] = tuple(
            layer_activations
            for layer in self.model.grad_layers
            if (
                (layer_activations := layer.cache["output_activations"]) is not None
                and isinstance(layer_activations, np.ndarray)
            )
        )

        with h5py.File(self._trace_logging_path, "a") as f:
            f.attrs["iterations"] = self._iterations

            iter_group = f.create_group(f"iteration_{self._iterations}")
            iter_group.attrs["timestamp"] = datetime.now().isoformat()

            if self._tracer_config.trace_activations:
                activation_group = iter_group.create_group("activations")
                for i, activation in enumerate(activations):
                    layer_group = activation_group.create_group(f"layer_{i}")
                    hist_values, hist_bins = np.histogram(activation, bins=100)
                    layer_group.create_dataset("histogram_values", data=hist_values)
                    layer_group.create_dataset("histogram_bins", data=hist_bins)

                    layer_group.create_dataset(
                        "deciles", data=np.quantile(activation, np.linspace(0, 1, 11))
                    )

                    layer_group.attrs["mean"] = np.mean(activation)
                    layer_group.attrs["std"] = np.std(activation)
                    layer_group.attrs["min"] = np.min(activation)
                    layer_group.attrs["max"] = np.max(activation)

            linear_layer_params = tuple(
                layer.parameters for layer in self._linear_layers
            )
            if self._tracer_config.trace_weights:
                weights_group = iter_group.create_group("weights")

                for i, param in enumerate(linear_layer_params):
                    layer_group = weights_group.create_group(f"layer_{i}")
                    hist_values, hist_bins = np.histogram(param.weights, bins=100)
                    layer_group.create_dataset("histogram_values", data=hist_values)
                    layer_group.create_dataset("histogram_bins", data=hist_bins)

                    layer_group.create_dataset(
                        "deciles",
                        data=np.quantile(param.weights, np.linspace(0, 1, 11)),
                    )

                    layer_group.attrs["mean"] = np.mean(param.weights)
                    layer_group.attrs["std"] = np.std(param.weights)
                    layer_group.attrs["min"] = np.min(param.weights)
                    layer_group.attrs["max"] = np.max(param.weights)

            if self._tracer_config.trace_biases:
                biases_group = iter_group.create_group("biases")

                for i, param in enumerate(linear_layer_params):
                    layer_group = biases_group.create_group(f"layer_{i}")
                    hist_values, hist_bins = np.histogram(param.biases, bins=100)
                    layer_group.create_dataset("histogram_values", data=hist_values)
                    layer_group.create_dataset("histogram_bins", data=hist_bins)

                    layer_group.create_dataset(
                        "deciles", data=np.quantile(param.biases, np.linspace(0, 1, 11))
                    )

                    layer_group.attrs["mean"] = np.mean(param.biases)
                    layer_group.attrs["std"] = np.std(param.biases)
                    layer_group.attrs["min"] = np.min(param.biases)
                    layer_group.attrs["max"] = np.max(param.biases)

            if self._tracer_config.trace_raw_gradients:
                linear_layer_gradients = tuple(
                    gradient
                    for gradient in raw_gradient
                    if isinstance(gradient, Parameters)
                )
                raw_gradient_group = iter_group.create_group("raw_gradients")

                for i, grad in enumerate(linear_layer_gradients):
                    layer_group = raw_gradient_group.create_group(f"layer_{i}")

                    weights_group = layer_group.create_group("weights")
                    hist_values, hist_bins = np.histogram(grad.weights, bins=100)
                    weights_group.create_dataset("histogram_values", data=hist_values)
                    weights_group.create_dataset("histogram_bins", data=hist_bins)

                    weights_group.create_dataset(
                        "deciles", data=np.quantile(grad.weights, np.linspace(0, 1, 11))
                    )

                    weights_group.attrs["mean"] = np.mean(grad.weights)
                    weights_group.attrs["std"] = np.std(grad.weights)
                    weights_group.attrs["min"] = np.min(grad.weights)
                    weights_group.attrs["max"] = np.max(grad.weights)

                    biases_group = layer_group.create_group("biases")
                    hist_values, hist_bins = np.histogram(grad.biases, bins=100)
                    biases_group.create_dataset("histogram_values", data=hist_values)
                    biases_group.create_dataset("histogram_bins", data=hist_bins)

                    biases_group.create_dataset(
                        "deciles", data=np.quantile(grad.biases, np.linspace(0, 1, 11))
                    )

                    biases_group.attrs["mean"] = np.mean(grad.biases)
                    biases_group.attrs["std"] = np.std(grad.biases)
                    biases_group.attrs["min"] = np.min(grad.biases)
                    biases_group.attrs["max"] = np.max(grad.biases)

            if self._tracer_config.trace_updates:
                update_gradients = tuple(
                    gradient for gradient in update if isinstance(gradient, Parameters)
                )
                update_group = iter_group.create_group("updates")

                # Create weights and biases groups at the top level
                update_weights_group = update_group.create_group("weights")
                update_biases_group = update_group.create_group("biases")

                for i, update_gradient in enumerate(update_gradients):
                    # Create layer groups under weights and biases
                    weights_layer_group = update_weights_group.create_group(
                        f"layer_{i}"
                    )
                    hist_values, hist_bins = np.histogram(
                        update_gradient.weights, bins=100
                    )
                    weights_layer_group.create_dataset(
                        "histogram_values", data=hist_values
                    )
                    weights_layer_group.create_dataset("histogram_bins", data=hist_bins)

                    weights_layer_group.attrs["mean"] = np.mean(update_gradient.weights)
                    weights_layer_group.attrs["std"] = np.std(update_gradient.weights)
                    weights_layer_group.attrs["min"] = np.min(update_gradient.weights)
                    weights_layer_group.attrs["max"] = np.max(update_gradient.weights)

                    weights_layer_group.create_dataset(
                        "deciles",
                        data=np.quantile(
                            update_gradient.weights, np.linspace(0, 1, 11)
                        ),
                    )

                    biases_layer_group = update_biases_group.create_group(f"layer_{i}")
                    hist_values, hist_bins = np.histogram(
                        update_gradient.biases, bins=100
                    )
                    biases_layer_group.create_dataset(
                        "histogram_values", data=hist_values
                    )
                    biases_layer_group.create_dataset("histogram_bins", data=hist_bins)

                    biases_layer_group.create_dataset(
                        "deciles",
                        data=np.quantile(update_gradient.biases, np.linspace(0, 1, 11)),
                    )

                    biases_layer_group.attrs["mean"] = np.mean(update_gradient.biases)
                    biases_layer_group.attrs["std"] = np.std(update_gradient.biases)
                    biases_layer_group.attrs["min"] = np.min(update_gradient.biases)
                    biases_layer_group.attrs["max"] = np.max(update_gradient.biases)

        self._iterations += 1
