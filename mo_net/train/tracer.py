from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp

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
        del iteration  # unused
        return True


class SampleTracerStrategy(TracerStrategy):
    def __init__(self, *, sample_rate: float, key: jax.Array):
        if sample_rate < 0 or sample_rate > 1:
            raise ValueError("Sample rate must be between 0 and 1")
        self._sample_rate = sample_rate
        self._key = key

    def should_trace(self, iteration: int) -> bool:
        del iteration  # unused
        self._key = jax.random.split(self._key)[0]
        return jax.random.uniform(self._key, ()).item() < self._sample_rate


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
        self._linear_layers: Sequence[Linear] = tuple(
            layer
            for module in model.modules
            for layer in module.layers
            if isinstance(layer, Linear)
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

        activations: Sequence[jnp.ndarray] = tuple(
            layer_activations
            for layer in self.model.grad_layers
            if (
                (layer_activations := layer.cache["output_activations"]) is not None
                and isinstance(layer_activations, jnp.ndarray)
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
                    hist_values, hist_bins = jnp.histogram(activation, bins=100)
                    layer_group.create_dataset("histogram_values", data=hist_values)
                    layer_group.create_dataset("histogram_bins", data=hist_bins)

                    layer_group.create_dataset(
                        "deciles", data=jnp.quantile(activation, jnp.linspace(0, 1, 11))
                    )

                    layer_group.attrs["mean"] = jnp.mean(activation)
                    layer_group.attrs["std"] = jnp.std(activation)
                    layer_group.attrs["min"] = jnp.min(activation)
                    layer_group.attrs["max"] = jnp.max(activation)

            linear_layer_params = tuple(
                layer.parameters for layer in self._linear_layers
            )
            if self._tracer_config.trace_weights:
                weights_group = iter_group.create_group("weights")

                for i, param in enumerate(linear_layer_params):
                    layer_group = weights_group.create_group(f"layer_{i}")
                    hist_values, hist_bins = jnp.histogram(param.weights, bins=100)
                    layer_group.create_dataset("histogram_values", data=hist_values)
                    layer_group.create_dataset("histogram_bins", data=hist_bins)

                    layer_group.create_dataset(
                        "deciles",
                        data=jnp.quantile(param.weights, jnp.linspace(0, 1, 11)),
                    )

                    layer_group.attrs["mean"] = jnp.mean(param.weights)
                    layer_group.attrs["std"] = jnp.std(param.weights)
                    layer_group.attrs["min"] = jnp.min(param.weights)
                    layer_group.attrs["max"] = jnp.max(param.weights)

            if self._tracer_config.trace_biases:
                biases_group = iter_group.create_group("biases")

                for i, param in enumerate(linear_layer_params):
                    layer_group = biases_group.create_group(f"layer_{i}")
                    hist_values, hist_bins = jnp.histogram(param.biases, bins=100)
                    layer_group.create_dataset("histogram_values", data=hist_values)
                    layer_group.create_dataset("histogram_bins", data=hist_bins)

                    layer_group.create_dataset(
                        "deciles",
                        data=jnp.quantile(param.biases, jnp.linspace(0, 1, 11)),
                    )

                    layer_group.attrs["mean"] = jnp.mean(param.biases)
                    layer_group.attrs["std"] = jnp.std(param.biases)
                    layer_group.attrs["min"] = jnp.min(param.biases)
                    layer_group.attrs["max"] = jnp.max(param.biases)

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
                    hist_values, hist_bins = jnp.histogram(grad.weights, bins=100)
                    weights_group.create_dataset("histogram_values", data=hist_values)
                    weights_group.create_dataset("histogram_bins", data=hist_bins)

                    weights_group.create_dataset(
                        "deciles",
                        data=jnp.quantile(grad.weights, jnp.linspace(0, 1, 11)),
                    )

                    weights_group.attrs["mean"] = jnp.mean(grad.weights)
                    weights_group.attrs["std"] = jnp.std(grad.weights)
                    weights_group.attrs["min"] = jnp.min(grad.weights)
                    weights_group.attrs["max"] = jnp.max(grad.weights)

                    biases_group = layer_group.create_group("biases")
                    hist_values, hist_bins = jnp.histogram(grad.biases, bins=100)
                    biases_group.create_dataset("histogram_values", data=hist_values)
                    biases_group.create_dataset("histogram_bins", data=hist_bins)

                    biases_group.create_dataset(
                        "deciles",
                        data=jnp.quantile(grad.biases, jnp.linspace(0, 1, 11)),
                    )

                    biases_group.attrs["mean"] = jnp.mean(grad.biases)
                    biases_group.attrs["std"] = jnp.std(grad.biases)
                    biases_group.attrs["min"] = jnp.min(grad.biases)
                    biases_group.attrs["max"] = jnp.max(grad.biases)

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
                    hist_values, hist_bins = jnp.histogram(
                        update_gradient.weights, bins=100
                    )
                    weights_layer_group.create_dataset(
                        "histogram_values", data=hist_values
                    )
                    weights_layer_group.create_dataset("histogram_bins", data=hist_bins)

                    weights_layer_group.attrs["mean"] = jnp.mean(
                        update_gradient.weights
                    )
                    weights_layer_group.attrs["std"] = jnp.std(update_gradient.weights)
                    weights_layer_group.attrs["min"] = jnp.min(update_gradient.weights)
                    weights_layer_group.attrs["max"] = jnp.max(update_gradient.weights)

                    weights_layer_group.create_dataset(
                        "deciles",
                        data=jnp.quantile(
                            update_gradient.weights, jnp.linspace(0, 1, 11)
                        ),
                    )

                    biases_layer_group = update_biases_group.create_group(f"layer_{i}")
                    hist_values, hist_bins = jnp.histogram(
                        update_gradient.biases, bins=100
                    )
                    biases_layer_group.create_dataset(
                        "histogram_values", data=hist_values
                    )
                    biases_layer_group.create_dataset("histogram_bins", data=hist_bins)

                    biases_layer_group.create_dataset(
                        "deciles",
                        data=jnp.quantile(
                            update_gradient.biases, jnp.linspace(0, 1, 11)
                        ),
                    )

                    biases_layer_group.attrs["mean"] = jnp.mean(update_gradient.biases)
                    biases_layer_group.attrs["std"] = jnp.std(update_gradient.biases)
                    biases_layer_group.attrs["min"] = jnp.min(update_gradient.biases)
                    biases_layer_group.attrs["max"] = jnp.max(update_gradient.biases)

        self._iterations += 1
