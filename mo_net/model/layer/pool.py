from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict, cast

import jax.numpy as jnp
from jax import lax

from mo_net.model.layer.base import Hidden
from mo_net.protos import Activations, D, Dimensions


class Cache(TypedDict):
    input_activations: Activations | None
    max_indices: jnp.ndarray | None


class MaxPooling2D(Hidden):
    @dataclass(frozen=True, kw_only=True)
    class Serialized:
        pool_size: tuple[int, int]
        stride: tuple[int, int]
        input_dimensions: tuple[int, int, int]

        def deserialize(
            self,
            *,
            training: bool = False,
            freeze_parameters: bool = False,
        ) -> MaxPooling2D:
            del training, freeze_parameters  # unused
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

        input_float = jnp.asarray(input_activations, dtype=jnp.float32)

        output = lax.reduce_window(
            input_float,
            init_value=jnp.finfo(input_float.dtype).min,
            computation=lax.max,
            window_dimensions=(1, 1, self._pool_h, self._pool_w),
            window_strides=(1, 1, self._stride_h, self._stride_w),
            padding="VALID",
        )

        self._cache["input_activations"] = Activations(input_float)
        self._cache["max_indices"] = output

        return Activations(output)

    def _backward_prop(self, *, dZ: D[Activations]) -> D[Activations]:
        input_activations = self._cache["input_activations"]
        pooled_output = self._cache["max_indices"]
        if input_activations is None:
            raise ValueError("Input activations were not set during forward pass.")
        if pooled_output is None:
            raise ValueError("Pooled output was not set during forward pass.")

        _, _, out_h, out_w = pooled_output.shape

        dX = jnp.zeros_like(input_activations)

        for h_out in range(out_h):
            for w_out in range(out_w):
                h_start = h_out * self._stride_h
                h_end = h_start + self._pool_h
                w_start = w_out * self._stride_w
                w_end = w_start + self._pool_w

                window = input_activations[:, :, h_start:h_end, w_start:w_end]

                pooled_vals = pooled_output[:, :, h_out : h_out + 1, w_out : w_out + 1]

                mask = (window == pooled_vals).astype(jnp.float32)

                counts = jnp.sum(mask, axis=(2, 3), keepdims=True)
                counts = jnp.maximum(counts, 1.0)

                mask = mask / counts

                grad_out = cast(jnp.ndarray, dZ)[
                    :, :, h_out : h_out + 1, w_out : w_out + 1
                ]

                dX = dX.at[:, :, h_start:h_end, w_start:w_end].add(mask * grad_out)

        return cast(D[Activations], dX)

    def serialize(self) -> MaxPooling2D.Serialized:
        channels, height, width = self.input_dimensions
        return MaxPooling2D.Serialized(
            pool_size=(self._pool_w, self._pool_h),
            stride=(self._stride_w, self._stride_h),
            input_dimensions=(channels, height, width),
        )
