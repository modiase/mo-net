from __future__ import annotations

import typing
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypedDict

import jax
import jax.numpy as jnp

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
                key=jax.random.PRNGKey(0),
            )

    class Cache(TypedDict):
        mask: jnp.ndarray | None
        input_activations: Activations | None

    def __init__(
        self,
        *,
        input_dimensions: Dimensions,
        keep_prob: float,
        key: jax.Array,
        training: bool = False,
    ):
        super().__init__(
            input_dimensions=input_dimensions,
            output_dimensions=input_dimensions,
        )
        self._training = training
        self._key = key
        self._backward_prop_fn = (
            self._backward_prop_training
            if training and keep_prob < 1.0
            else self._backward_prop_non_training
        )
        self._forward_prop_fn = (
            self._forward_prop_training
            if training and keep_prob < 1.0
            else self._forward_prop_non_training
        )

        if not 0.0 < keep_prob <= 1.0:
            raise ValueError(f"keep_prob must be in (0, 1], got {keep_prob}")

        self._keep_prob = keep_prob
        self._cache: Dropout.Cache = {
            "mask": None,
            "input_activations": None,
        }

    def _forward_prop_training(self, *, input_activations: Activations) -> Activations:
        self._cache["input_activations"] = input_activations

        self._key, subkey = jax.random.split(self._key)
        mask = jax.random.bernoulli(
            subkey, self._keep_prob, shape=input_activations.shape
        ).astype(jnp.float32)
        self._cache["mask"] = mask

        return Activations(input_activations * mask / self._keep_prob)

    def _forward_prop_non_training(
        self, *, input_activations: Activations
    ) -> Activations:
        return input_activations

    def _forward_prop(self, *, input_activations: Activations) -> Activations:
        return self._forward_prop_fn(input_activations=input_activations)

    def _backward_prop_training(self, *, dZ: D[Activations]) -> D[Activations]:
        if self._cache["mask"] is None:
            raise ValueError("Mask not set during forward pass.")

        return dZ * self._cache["mask"] / self._keep_prob

    def _backward_prop_non_training(self, *, dZ: D[Activations]) -> D[Activations]:
        return dZ

    def _backward_prop(self, *, dZ: D[Activations]) -> D[Activations]:
        return self._backward_prop_fn(dZ=dZ)

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
        key: jax.Array,
    ) -> None:
        if len(keep_probs) != len(model.hidden_modules):
            raise ValueError(
                f"Number of keep probabilities ({len(keep_probs)}) must match the number of hidden modules ({len(model.hidden_modules)})"
            )
        for module, keep_prob in zip(model.hidden_modules, keep_probs, strict=True):
            key, subkey = jax.random.split(key)
            module.append_layer(
                Dropout(
                    input_dimensions=module.output_dimensions,
                    keep_prob=keep_prob,
                    training=training,
                    key=subkey,
                )
            )
