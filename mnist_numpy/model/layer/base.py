from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import reduce
from itertools import chain

import numpy as np
from more_itertools import last

from mnist_numpy.protos import (
    Activations,
    D,
    Dimensions,
    TrainingStepHandler,
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
        self._training_step_handlers: Sequence[TrainingStepHandler] = ()

    def register_training_step_handler(self, handler: TrainingStepHandler) -> None:
        self._training_step_handlers = tuple(
            chain(self._training_step_handlers, (handler,))
        )

    def forward_prop(self, *, input_activations: Activations) -> Activations:
        # TODO: Check if this is correct - not sure I'm considering broadcasting correctly
        if last(np.atleast_2d(input_activations).shape) != last(self.input_dimensions):
            raise ValueError(
                f"Input activations shape {input_activations.shape} does not match "
                f"input dimensions {self.input_dimensions}."
            )
        return reduce(
            lambda acc, handler: handler(acc),  # type: ignore[operator]
            (handler.post_forward for handler in self._training_step_handlers),
            self._forward_prop(
                input_activations=(
                    reduce(
                        lambda acc, handler: handler(acc),
                        tuple(
                            handler.pre_forward
                            for handler in self._training_step_handlers
                        ),
                        input_activations,
                    )
                )
            ),
        )

    @abstractmethod
    def _forward_prop(self, *, input_activations: Activations) -> Activations: ...

    @property
    def input_dimensions(self) -> Dimensions:
        return self._input_dimensions

    @property
    def output_dimensions(self) -> Dimensions:
        return self._output_dimensions


class Hidden(_Base):
    def backward_prop(self, *, dZ: D[Activations]) -> D[Activations]:
        return reduce(
            lambda acc, handler: handler(acc),
            reversed(
                tuple(handler.post_backward for handler in self._training_step_handlers)
            ),
            self._backward_prop(
                dZ=reduce(
                    lambda acc, handler: handler(acc),
                    reversed(
                        tuple(
                            handler.pre_backward
                            for handler in self._training_step_handlers
                        )
                    ),
                    dZ,
                ),
            ),
        )

    @abstractmethod
    def _backward_prop(
        self,
        *,
        dZ: D[Activations],
    ) -> D[Activations]: ...
