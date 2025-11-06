from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypedDict, cast

import jax.numpy as jnp

from mo_net.model.layer.base import Hidden
from mo_net.protos import Activations, D, Dimensions


class Average(Hidden):
    @dataclass(frozen=True, kw_only=True)
    class Serialized:
        input_dimensions: tuple[int, ...]
        output_dimensions: tuple[int, ...]
        axis: tuple[int, ...]

        def deserialize(
            self,
            *,
            training: bool = False,
            freeze_parameters: bool = False,
        ) -> Average:
            del training, freeze_parameters  # unused
            return Average(
                input_dimensions=self.input_dimensions,
                axis=self.axis,
            )

    class Cache(TypedDict):
        input_shape: tuple[int, ...] | None

    def __init__(
        self,
        *,
        input_dimensions: Dimensions,
        axis: int | Sequence[int],
        layer_id: str | None = None,
    ):
        axis_tuple = (axis,) if isinstance(axis, int) else tuple(axis)
        if not axis_tuple:
            raise ValueError("Axis cannot be empty")
        if any(ax < 0 or ax >= len(input_dimensions) for ax in axis_tuple):
            raise IndexError(
                f"Axis {axis_tuple} is out of range for input dimensions {input_dimensions}"
            )
        output_dimensions = tuple(
            d for i, d in enumerate(input_dimensions) if i not in axis_tuple
        )
        super().__init__(
            layer_id=layer_id,
            input_dimensions=input_dimensions,
            output_dimensions=output_dimensions,
        )
        self._axis = axis_tuple
        self._cache: Average.Cache = {"input_shape": None}

    def _forward_prop(self, *, input_activations: Activations) -> Activations:
        self._cache["input_shape"] = input_activations.shape
        return Activations(
            jnp.mean(
                input_activations,
                axis=tuple(ax + 1 for ax in self._axis),
                keepdims=False,
            )
        )

    def _backward_prop(self, *, dZ: D[Activations]) -> D[Activations]:
        if self._cache["input_shape"] is None:
            raise ValueError("Input shape not cached during forward pass.")

        expanded_dZ = dZ
        for ax, factor in zip(
            sorted(ax + 1 for ax in self._axis),
            tuple(self._cache["input_shape"][ax + 1] for ax in self._axis),
            strict=True,
        ):
            expanded_dZ = jnp.expand_dims(cast(jnp.ndarray, expanded_dZ), ax) / factor  # type: ignore[reportArgumentType]

        return cast(
            D[Activations],
            Activations(
                jnp.broadcast_to(
                    cast(jnp.ndarray, expanded_dZ), self._cache["input_shape"]
                )
            ),
        )  # type: ignore[reportArgumentType]

    @property
    def axis(self) -> tuple[int, ...]:
        return self._axis

    def serialize(self) -> Average.Serialized:
        return Average.Serialized(
            input_dimensions=tuple(self._input_dimensions),
            output_dimensions=tuple(self._output_dimensions),
            axis=self._axis,
        )
