from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np

from mnist_numpy.model.layer.base import Hidden
from mnist_numpy.protos import Activations, D, Dimensions


class Cache(TypedDict):
    input_activations: Activations | None
    max_indices: np.ndarray | None


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
        self._pool_size_x, self._pool_size_y = pool_size
        self._stride_x, self._stride_y = stride
        output_dimensions = (
            input_dimensions[0],
            (input_dimensions[1] - self._pool_size_x) // self._stride_x + 1,
            (input_dimensions[2] - self._pool_size_y) // self._stride_y + 1,
        )
        super().__init__(
            input_dimensions=input_dimensions, output_dimensions=output_dimensions
        )
        self._cache: Cache = {
            "input_activations": None,
            "max_indices": None,
        }

    def _forward_prop(self, *, input_activations: Activations) -> Activations:
        batch_size = input_activations.shape[0]
        channels, h_out, w_out = self.output_dimensions

        output = np.zeros((batch_size, channels, h_out, w_out))

        self._cache["input_activations"] = input_activations
        self._cache["max_indices"] = np.zeros(
            (
                batch_size,
                channels,
                h_out,
                w_out,
                2,  # [y-idx, x-idx]
            ),
            dtype=int,
        )

        for h in range(h_out):
            for w in range(w_out):
                w_start, w_end = (
                    (w_start := w * self._stride_y),
                    w_start + self._pool_size_y,
                )
                h_start, h_end = (
                    (h_start := h * self._stride_x),
                    h_start + self._pool_size_x,
                )
                flattened_max_indices = np.argmax(
                    input_activations[:, :, h_start:h_end, w_start:w_end].reshape(
                        batch_size, channels, -1
                    ),
                    axis=2,
                )

                h_indices, w_indices = np.unravel_index(
                    flattened_max_indices.flatten(),
                    (self._pool_size_x, self._pool_size_y),
                )
                h_indices = h_indices.reshape(batch_size, channels)
                w_indices = w_indices.reshape(batch_size, channels)

                self._cache["max_indices"][:, :, h, w, 0] = h_indices
                self._cache["max_indices"][:, :, h, w, 1] = w_indices

                output[:, :, h, w] = input_activations[
                    np.arange(batch_size)[:, None],
                    np.arange(channels)[None, :],
                    h_start + h_indices,
                    w_start + w_indices,
                ]

        return output

    def _backward_prop(self, *, dZ: D[Activations]) -> D[Activations]:
        if (input_activations := self._cache["input_activations"]) is None:
            raise ValueError("Input activations were not set during forward pass.")
        if (max_indices := self._cache["max_indices"]) is None:
            raise ValueError("Max indices were not set during forward pass.")

        batch_size, channels = input_activations.shape[:2]
        h_out, w_out = self.output_dimensions[1], self.output_dimensions[2]

        dX = np.zeros_like(input_activations)

        for h in range(h_out):
            for w in range(w_out):
                h_start = h * self._stride_x
                w_start = w * self._stride_y

                h_indices = max_indices[:, :, h, w, 0]
                w_indices = max_indices[:, :, h, w, 1]

                h_abs = h_start + h_indices
                w_abs = w_start + w_indices

                batch_indices = np.arange(batch_size)[:, None]
                channel_indices = np.arange(channels)[None, :]

                dX[batch_indices, channel_indices, h_abs, w_abs] += dZ[:, :, h, w]

        return dX

    def serialize(self) -> MaxPooling2D.Serialized:
        return MaxPooling2D.Serialized(
            pool_size=self._pool_size_x,
            stride=self._stride_x,
            input_dimensions=self.input_dimensions,
        )
