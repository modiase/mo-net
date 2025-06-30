from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict, cast

import numpy as np
from numpy.lib.stride_tricks import as_strided

from mo_net.model.layer.base import Hidden
from mo_net.protos import Activations, D, Dimensions, d


class Cache(TypedDict):
    input_activations: Activations | None
    max_indices: np.ndarray | None


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

        input_windows = as_strided(
            input_activations,
            shape=(batch_size, channels, h_out, w_out, self._pool_h, self._pool_w),
            strides=(
                input_activations.strides[0],
                input_activations.strides[1],
                input_activations.strides[2] * self._stride_h,
                input_activations.strides[3] * self._stride_w,
                input_activations.strides[2],
                input_activations.strides[3],
            ),
            writeable=False,
        )

        h_indices, w_indices = [
            indices.reshape(batch_size, channels, h_out, w_out)
            for indices in np.unravel_index(
                np.argmax(
                    input_windows.reshape(batch_size, channels, h_out, w_out, -1),
                    axis=-1,
                ).flatten(),
                (self._pool_h, self._pool_w),
            )
        ]

        self._cache["input_activations"] = input_activations
        self._cache["max_indices"] = np.stack([h_indices, w_indices], axis=-1)

        return Activations(
            input_windows[
                np.arange(batch_size)[:, None, None, None],
                np.arange(channels)[None, :, None, None],
                np.arange(h_out)[None, None, :, None],
                np.arange(w_out)[None, None, None, :],
                h_indices,
                w_indices,
            ]
        )

    def _backward_prop(self, *, dZ: D[Activations]) -> D[Activations]:
        input_activations = self._cache["input_activations"]
        max_indices = self._cache["max_indices"]
        if input_activations is None:
            raise ValueError("Input activations were not set during forward pass.")
        if max_indices is None:
            raise ValueError("Max indices were not set during forward pass.")

        batch_size, channels, _, _ = input_activations.shape
        _, _, h_out, w_out = cast(np.ndarray, dZ).shape

        dX = np.zeros_like(input_activations)

        h_coords_flat, w_coords_flat = [
            coords.flatten()
            for coords in np.meshgrid(np.arange(h_out), np.arange(w_out), indexing="ij")
        ]

        h_indices_flat, w_indices_flat = [
            max_indices.reshape(batch_size, channels, h_out * w_out, 2)[:, :, :, i]
            for i in [0, 1]
        ]

        np.add.at(
            dX,
            (
                np.arange(batch_size)[:, None, None],
                np.arange(channels)[None, :, None],
                h_coords_flat * self._stride_h + h_indices_flat,
                w_coords_flat * self._stride_w + w_indices_flat,
            ),
            cast(np.ndarray, dZ).reshape(batch_size, channels, -1),
        )

        return d(Activations(dX))

    def serialize(self) -> MaxPooling2D.Serialized:
        channels, height, width = self.input_dimensions
        return MaxPooling2D.Serialized(
            pool_size=(self._pool_w, self._pool_h),
            stride=(self._stride_w, self._stride_h),
            input_dimensions=(channels, height, width),
        )
