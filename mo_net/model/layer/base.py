from __future__ import annotations

import struct
import threading
from abc import ABC, abstractmethod
from collections.abc import MutableSet
from typing import IO, Final, Optional
from uuid import uuid4

import jax.numpy as jnp
from loguru import logger

from mo_net.protos import (
    Activations,
    CacheType_co,
    D,
    Dimensions,
    GradLayer,
    ParamType,
)


class _LayerRegistry:
    """Thread-safe registry for tracking layer IDs and ensuring uniqueness."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._used_names: MutableSet[str] = set()

    def generate_id(self, name: Optional[str] = None) -> str:
        with self._lock:
            if name is not None:
                if name in self._used_names:
                    raise ValueError(f"Layer name '{name}' is already in use")
                self._used_names.add(name)
                return name

            auto_name = uuid4().hex
            while auto_name in self._used_names:
                auto_name = uuid4().hex
            self._used_names.add(auto_name)
            return auto_name


_global_registry: Final[_LayerRegistry] = _LayerRegistry()


class BadLayerId(Exception): ...


class _Base(ABC):
    def __init__(
        self,
        *,
        input_dimensions: Dimensions,
        layer_id: str | None = None,
        output_dimensions: Dimensions,
    ):
        self._input_dimensions = input_dimensions
        self._output_dimensions = output_dimensions
        self._layer_id = _global_registry.generate_id(layer_id)

    def forward_prop(self, input_activations: Activations) -> Activations:
        # We wish to ensure that all inputs are at least 2D arrays such that the
        # leading dimension is always the 'batch' dimension.
        logger.trace(f"Forward propagating {self} (id: {self._layer_id}).")
        input_activations = Activations(jnp.atleast_2d(input_activations))
        if input_activations.shape[1:] != self.input_dimensions:
            raise ValueError(
                f"Input activations shape {input_activations.shape[1:]} does not match "
                f"input dimensions {self.input_dimensions}."
            )
        return self._forward_prop(input_activations=input_activations)

    @abstractmethod
    def _forward_prop(self, *, input_activations: Activations) -> Activations: ...

    @property
    def input_dimensions(self) -> Dimensions:
        return self._input_dimensions

    @property
    def output_dimensions(self) -> Dimensions:
        return self._output_dimensions

    @property
    def layer_id(self) -> str:
        """Unique identifier for this layer instance."""
        return self._layer_id

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} id={self.layer_id} "
            f"dimensions={(self.input_dimensions, self.output_dimensions)}>"
        )


class Hidden(_Base):
    def backward_prop(self, dZ: D[Activations]) -> D[Activations]:
        logger.trace(f"Backward propagating {self} (id: {self._layer_id}).")
        return self._backward_prop(dZ=dZ)

    @abstractmethod
    def _backward_prop(
        self,
        *,
        dZ: D[Activations],
    ) -> D[Activations]: ...


class ParametrisedHidden(Hidden, GradLayer[ParamType, CacheType_co]):
    @abstractmethod
    def write_serialized_parameters(self, buffer: IO[bytes]) -> None: ...

    @property
    @abstractmethod
    def parameter_nbytes(self) -> int: ...

    @property
    @abstractmethod
    def parameter_count(self) -> int: ...

    @abstractmethod
    def read_serialized_parameters(self, data: IO[bytes]) -> None: ...

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} id={self._layer_id} "
            f"dimensions={(self.input_dimensions, self.output_dimensions)} "
            f"parameter_count={self.parameter_count}>"
        )

    @staticmethod
    def get_layer_id(buffer: IO[bytes], peek: bool = False) -> str:
        start_pos = buffer.tell()
        header = struct.unpack(">I", buffer.read(4))[0]
        layer_id = buffer.read(header).decode()
        if peek:
            buffer.seek(start_pos)
        return layer_id

    def _write_header(self, buffer: IO[bytes]) -> None:
        header = struct.pack(">I", len(self._layer_id)) + self._layer_id.encode()
        buffer.write(header)
