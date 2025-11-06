from itertools import chain

import jax.numpy as jnp

from mo_net.model.layer.linear import Linear
from mo_net.model.model import Model
from mo_net.optimiser.base import Base as BaseOptimizer
from mo_net.protos import TrainingStepHandler, d


class L1Regulariser(TrainingStepHandler):
    def __init__(
        self, *, lambda_: float, batch_size: int, layer: Linear, key: jnp.ndarray
    ):
        self._lambda = lambda_
        self._layer = layer
        self._batch_size = batch_size
        self._key = key
        self._threshold = 0.01

    def after_compute_update(self, learning_rate: float) -> None:
        del learning_rate
        dP = self._layer.cache.get("dP", self._layer.empty_gradient())  # type: ignore[attr-defined]
        if dP is None:
            return

        self._layer.cache["dP"] = d(  # type: ignore[index]
            dP
            + Linear.Parameters(
                weights=self._lambda
                * jnp.sign(self._layer.parameters.weights)
                / self._batch_size,
                biases=jnp.zeros_like(dP.biases),  # type: ignore[reportAttributeAccessIssue]
            )
        )

    def compute_regularisation_loss(self) -> float:
        return (
            self._lambda
            * jnp.sum(jnp.abs(self._layer.parameters.weights))
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
        key: jnp.ndarray,
    ) -> None:
        for layer in chain.from_iterable(module.layers for module in model.modules):
            if not isinstance(layer, Linear):
                continue
            regulariser = L1Regulariser(
                lambda_=lambda_, batch_size=batch_size, layer=layer, key=key
            )
            optimiser.register_after_compute_update_handler(
                regulariser.after_compute_update
            )
            model.register_loss_contributor(regulariser)
