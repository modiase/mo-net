from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Final, Self, Sequence

import jax.numpy as jnp

from mo_net.model.layer.linear import Parameters
from mo_net.protos import RawGradientType, UpdateGradientType
from mo_net.train.exceptions import CheckFailed

# This value has been found empirically to be a good threshold for exploding
# gradients. Obviously, a Z score of 50 is an insanely high value, but it can be
# understood as recognising that the weights are not being modelled by a normal
# distribution since the likelihood of a Z score of 50 a random variable truly
# normally distributed is 1 - erf(50) which is approximately 0.
EPSILON: Final[float] = 1e-8
MAX_Z_SCORE_UPPER_BOUND: Final[float] = 50.0
MAX_Z_SCORE_LOWER_BOUND: Final[float] = 20.0


@dataclass(frozen=True, kw_only=True)
class WeightGradientRunningAverages:
    sums: jnp.ndarray
    sums_of_squares: jnp.ndarray

    @classmethod
    def from_weights(cls, weights_seq: Sequence[jnp.ndarray]) -> Self:
        return cls(
            sums=jnp.array([jnp.sum(weights) for weights in weights_seq]),
            sums_of_squares=jnp.array([jnp.sum(weights**2) for weights in weights_seq]),
        )

    @classmethod
    def from_weights_and_update(
        cls,
        running_average: WeightGradientRunningAverages,
        update: WeightGradientRunningAverages,
        update_count: int,
    ) -> Self:
        r = 1 / update_count
        return cls(
            sums=running_average.sums * (1 - r) + update.sums * r,
            sums_of_squares=running_average.sums_of_squares * (1 - r)
            + update.sums_of_squares * r,
        )

    @classmethod
    def none(cls) -> Self:
        return cls(sums=jnp.zeros(1), sums_of_squares=jnp.zeros(1))


class Monitor:
    def __init__(
        self,
        *,
        batches_per_epoch: int,
        history_max_len: int,
        warmup_epochs: int,
    ):
        self._L_history_max_len = history_max_len
        self._L_history: deque[float] = deque(maxlen=self._L_history_max_len)
        self._L_history_snapshot: Sequence[float] = ()
        self._running_update_count = 0
        self._running_weights: WeightGradientRunningAverages = (
            WeightGradientRunningAverages.none()
        )
        self._warmup_batches = warmup_epochs * batches_per_epoch

    def reset(self, restore_history: bool = False) -> None:
        if restore_history:
            self._L_history = deque(
                self._L_history_snapshot, maxlen=self._L_history_max_len
            )
        else:
            self._L_history.clear()
        self._running_update_count = 0
        self._running_weights = WeightGradientRunningAverages.none()

    def post_batch(
        self,
        raw_gradient: RawGradientType,
        update: UpdateGradientType,
    ) -> None | CheckFailed:
        del update  # unused
        linear_layer_gradients = [
            gradient for gradient in raw_gradient if isinstance(gradient, Parameters)
        ]
        self._running_update_count += 1
        self._running_weights = WeightGradientRunningAverages.from_weights_and_update(
            self._running_weights,
            WeightGradientRunningAverages.from_weights(
                tuple(param.weights for param in linear_layer_gradients)
            ),
            self._running_update_count,
        )
        ns = jnp.array([param.weights.size for param in linear_layer_gradients])
        means = self._running_weights.sums / ns
        variances = self._running_weights.sums_of_squares / ns - means**2

        weight_gradients_max_Z_scores = jnp.array(
            [
                jnp.max((weights - mean) / (jnp.sqrt(variance) + EPSILON))
                for (weights, mean, variance) in zip(
                    [param.weights for param in linear_layer_gradients],
                    means,
                    variances,
                    strict=True,
                )
            ]
        )
        if self._running_update_count == 0:
            return None

        if self._running_update_count > self._warmup_batches:
            if (
                weight_gradients_max_Z_score := jnp.max(weight_gradients_max_Z_scores)
            ) > max(
                MAX_Z_SCORE_UPPER_BOUND / jnp.log(jnp.log(self._running_update_count)),
                MAX_Z_SCORE_LOWER_BOUND,
            ):
                return CheckFailed(
                    message=f"Exploding gradients detected. {weight_gradients_max_Z_score=}"
                )
        return None

    def post_epoch(self, L: float) -> None | CheckFailed:
        self._L_history.append(L)
        if len(self._L_history) < self._L_history_max_len:
            return None
        if jnp.polyfit(range(len(self._L_history)), self._L_history, 1)[0] >= 0:
            return CheckFailed(message="Model is not learning.")
        return None

    def clear_history(self) -> None:
        self._L_history.clear()
        self._L_history_snapshot = ()
