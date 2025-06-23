from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict, cast

import jax.numpy as jnp
from jax import lax

from mo_net.model.layer.base import Hidden
from mo_net.protos import Activations, D, Dimensions, d


class Cache(TypedDict):
    input_activations: Activations | None
    max_indices: jnp.ndarray | None


class MaxPooling2D(Hidden):
    @dataclass(frozen=True, kw_only=True)
    class Serialized:
        pool_size: tuple[int, int]
        stride: tuple[int, int]
        input_dimensions: tuple[int, int, int]

        def deserialize(self, *, training: bool = False) -> MaxPooling2D:
            del training  # unused
            return MaxPooling2D(
                input_dimensions=self.input_dimensions,
                pool_size=self.pool_size,
                stride=self.stride,
            )

    Cache = Cache

    def __init__(
        self,
        *,
        input_dimensions: Dimensions,
        pool_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
    ):
        if not isinstance(pool_size, (int, tuple)):
            raise ValueError(
                f"Pool size must be an integer or a tuple, got {pool_size}"
            )
        pool_size = (pool_size, pool_size) if isinstance(pool_size, int) else pool_size
        if not isinstance(stride, (int, tuple)):
            raise ValueError(f"Stride must be an integer or a tuple, got {stride}")
        stride = (stride, stride) if isinstance(stride, int) else stride

        self._pool_w, self._pool_h = pool_size
        self._stride_w, self._stride_h = stride

        channels, in_h, in_w = input_dimensions

        super().__init__(
            input_dimensions=input_dimensions,
            output_dimensions=(
                channels,
                (in_h - self._pool_h) // self._stride_h + 1,
                (in_w - self._pool_w) // self._stride_w + 1,
            ),
        )
        self._cache: Cache = {
            "input_activations": None,
            "max_indices": None,
        }

    def _forward_prop(self, *, input_activations: Activations) -> Activations:
        batch_size, channels, _, _ = input_activations.shape
        _, h_out, w_out = self.output_dimensions

        # Ensure input is float type for JAX operations
        input_float = jnp.asarray(input_activations, dtype=jnp.float32)

        # Use JAX's reduce_window for max pooling
        output = lax.reduce_window(
            input_float,
            init_value=jnp.finfo(input_float.dtype).min,
            computation=lax.max,
            window_dimensions=(1, 1, self._pool_h, self._pool_w),
            window_strides=(1, 1, self._stride_h, self._stride_w),
            padding="VALID",
        )

        # For backward pass, we need to track which elements were selected
        # We'll compute this by comparing the pooled output with the input
        self._cache["input_activations"] = input_float  # Store float version
        self._cache["max_indices"] = output  # Store output for backward pass

        return Activations(output)

    def _backward_prop(self, *, dZ: D[Activations]) -> D[Activations]:
        input_activations = self._cache["input_activations"]
        pooled_output = self._cache["max_indices"]  # This is actually the pooled output
        if input_activations is None:
            raise ValueError("Input activations were not set during forward pass.")
        if pooled_output is None:
            raise ValueError("Pooled output was not set during forward pass.")

        batch_size, channels, in_h, in_w = input_activations.shape
        _, _, out_h, out_w = pooled_output.shape

        # Initialize gradient w.r.t input
        dX = jnp.zeros_like(input_activations)

        # Create a view of the input that matches the pooling windows
        # We'll iterate over output positions and find max locations
        for h_out in range(out_h):
            for w_out in range(out_w):
                # Determine the window boundaries
                h_start = h_out * self._stride_h
                h_end = h_start + self._pool_h
                w_start = w_out * self._stride_w
                w_end = w_start + self._pool_w

                # Get the window from input for all batches and channels
                window = input_activations[:, :, h_start:h_end, w_start:w_end]

                # Get the corresponding pooled values
                pooled_vals = pooled_output[:, :, h_out : h_out + 1, w_out : w_out + 1]

                # Create a mask for positions that equal the max
                mask = (window == pooled_vals).astype(jnp.float32)

                # Count how many positions have the max value (for handling ties)
                counts = jnp.sum(mask, axis=(2, 3), keepdims=True)
                counts = jnp.maximum(counts, 1.0)  # Avoid division by zero

                # Normalize the mask to distribute gradients equally
                mask = mask / counts

                # Get the gradients for this output position
                grad_out = cast(jnp.ndarray, dZ)[
                    :, :, h_out : h_out + 1, w_out : w_out + 1
                ]

                # Apply gradients to the window
                dX = dX.at[:, :, h_start:h_end, w_start:w_end].add(mask * grad_out)

        return d(Activations(dX))

    def serialize(self) -> MaxPooling2D.Serialized:
        channels, height, width = self.input_dimensions
        return MaxPooling2D.Serialized(
            pool_size=(self._pool_w, self._pool_h),
            stride=(self._stride_w, self._stride_h),
            input_dimensions=(channels, height, width),
        )
