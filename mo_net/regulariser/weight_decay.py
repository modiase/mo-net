from itertools import chain
from typing import Protocol, cast

import jax.numpy as jnp

from mo_net.model.layer.embedding import Embedding
from mo_net.model.layer.linear import Linear
from mo_net.model.model import Model
from mo_net.optimiser.base import Base as BaseOptimizer
from mo_net.protos import TrainingStepHandler, d


class WeightDecayRegulariser(TrainingStepHandler):
    """https://arxiv.org/pdf/1711.05101"""

    def __init__(self, *, lambda_: float, batch_size: int, layer: Linear):
        self._lambda = lambda_
        self._layer = layer
        self._batch_size = batch_size

    def after_compute_update(self, learning_rate: float) -> None:
        del learning_rate  # unused
        dP = self._layer.cache.get("dP", self._layer.empty_gradient())  # type: ignore[attr-defined]
        if dP is None:
            return

        dP_params: Linear.Parameters = cast(Linear.Parameters, dP)  # type: ignore[valid-type]
        self._layer.cache["dP"] = d(  # type: ignore[index]
            dP
            + Linear.Parameters(
                weights=self._lambda * self._layer.parameters.weights,
                biases=jnp.zeros_like(dP_params.biases),
            )
        )

    def compute_regularisation_loss(self) -> float:
        return (
            0.5
            * self._lambda
            * jnp.sum(self._layer.parameters.weights**2)
            / self._batch_size
        ).item()

    def __call__(self) -> float:
        return self.compute_regularisation_loss()

    @staticmethod
    def attach(
        *,
        lambda_: float,
        batch_size: int,
        optimiser: BaseOptimizer,
        model: Model,
    ) -> None:
        for layer in chain.from_iterable(module.layers for module in model.modules):
            if not isinstance(layer, Linear):
                continue
            layer_regulariser = WeightDecayRegulariser(
                lambda_=lambda_, batch_size=batch_size, layer=layer
            )
            optimiser.register_after_compute_update_handler(
                layer_regulariser.after_compute_update
            )
            model.register_loss_contributor(layer_regulariser)


class HasEmbeddingLayer(Protocol):
    @property
    def embedding_layer(self) -> Embedding: ...


class EmbeddingWeightDecayRegulariser(TrainingStepHandler):
    def __init__(self, *, lambda_: float, batch_size: int, layer: Embedding):
        self._lambda = lambda_
        self._layer = layer
        self._batch_size = batch_size

    def after_compute_update(self, learning_rate: float) -> None:
        del learning_rate  # unused
        dP = self._layer.cache.get("dP", self._layer.empty_gradient())  # type: ignore[attr-defined]
        if dP is None:
            return
        self._layer.cache["dP"] = d(
            dP
            + Embedding.Parameters(
                embeddings=self._lambda * self._layer.parameters.embeddings,
            )
        )

    def compute_regularisation_loss(self) -> float:
        return (
            0.5
            * self._lambda
            * jnp.sum(self._layer.parameters.embeddings**2)
            / self._batch_size
        ).item()

    def __call__(self) -> float:
        return self.compute_regularisation_loss()

    @staticmethod
    def attach(
        *,
        lambda_: float,
        batch_size: int,
        optimiser: BaseOptimizer,
        model: HasEmbeddingLayer,
    ) -> None:
        optimiser.register_after_compute_update_handler(
            EmbeddingWeightDecayRegulariser(
                lambda_=lambda_, batch_size=batch_size, layer=model.embedding_layer
            ).after_compute_update
        )
