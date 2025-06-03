from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from loguru import logger

from mo_net.protos import (
    Activations,
    D,
    Dimensions,
)


class _Base(ABC):
    def __init__(
        self,
        *,
        input_dimensions: Dimensions,
        output_dimensions: Dimensions,
    ):
        self._input_dimensions = input_dimensions
        self._output_dimensions = output_dimensions

    def forward_prop(self, input_activations: Activations) -> Activations:
        # We wish to ensure that all inputs are at least 2D arrays such that the
        # leading dimension is always the 'batch' dimension.
        logger.trace(f"Forward propagating {self}.")
        input_activations = Activations(np.atleast_2d(input_activations))
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


class Hidden(_Base):
    def backward_prop(self, dZ: D[Activations]) -> D[Activations]:
        logger.trace(f"Backward propagating {self}.")
        return self._backward_prop(dZ=dZ)

    @abstractmethod
    def _backward_prop(
        self,
        *,
        dZ: D[Activations],
    ) -> D[Activations]: ...
