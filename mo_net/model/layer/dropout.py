from __future__ import annotations

import typing
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypedDict

import numpy as np

from mo_net.model.layer.base import Hidden
from mo_net.protos import Activations, D, Dimensions

if typing.TYPE_CHECKING:
    from mo_net.model.model import Model


class Dropout(Hidden):
    """
    https://arxiv.org/abs/1207.0580
    """

    @dataclass(frozen=True, kw_only=True)
    class Serialized:
        input_dimensions: tuple[int, ...]
        keep_prob: float

        def deserialize(
            self,
            *,
            training: bool,
            freeze_parameters: bool = False,
        ) -> Dropout:
            del freeze_parameters  # unused
            return Dropout(
                input_dimensions=self.input_dimensions,
                keep_prob=self.keep_prob,
                training=training,
            )

    class Cache(TypedDict):
        mask: np.ndarray | None
        input_activations: Activations | None

    def __init__(
        self,
        *,
        input_dimensions: Dimensions,
        keep_prob: float,
        training: bool = False,
    ):
        super().__init__(
            input_dimensions=input_dimensions,
            output_dimensions=input_dimensions,
        )
        self._training = training

        if not 0.0 < keep_prob <= 1.0:
            raise ValueError(f"keep_prob must be in (0, 1], got {keep_prob}")

        self._keep_prob = keep_prob
        self._cache: Dropout.Cache = {
            "mask": None,
            "input_activations": None,
        }

    def _forward_prop(self, *, input_activations: Activations) -> Activations:
        if self._keep_prob == 1.0 or not self._training:
            return input_activations
        self._cache["input_activations"] = input_activations

        mask = np.random.binomial(1, self._keep_prob, size=input_activations.shape)
        self._cache["mask"] = mask

        return Activations(input_activations * mask / self._keep_prob)

    def _backward_prop(self, *, dZ: D[Activations]) -> D[Activations]:
        if self._keep_prob == 1.0 or not self._training:
            return dZ

        if self._cache["mask"] is None:
            raise ValueError("Mask not set during forward pass.")

        return dZ * self._cache["mask"] / self._keep_prob

    def serialize(self) -> Dropout.Serialized:
        return Dropout.Serialized(
            input_dimensions=tuple(self.input_dimensions),
            keep_prob=self._keep_prob,
        )

    @staticmethod
    def attach_dropout_layers(
        *,
        model: Model,
        keep_probs: Sequence[float],
        training: bool,
    ) -> None:
        if len(keep_probs) != len(model.hidden_modules):
            raise ValueError(
                f"Number of keep probabilities ({len(keep_probs)}) must match the number of hidden modules ({len(model.hidden_modules)})"
            )
        for module, keep_prob in zip(model.hidden_modules, keep_probs, strict=True):
            module.append_layer(
                Dropout(
                    input_dimensions=module.output_dimensions,
                    keep_prob=keep_prob,
                    training=training,
                )
            )
