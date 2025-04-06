from typing import TypeVar, cast

import numpy as np


def cross_entropy(Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
    Y_pred = np.clip(Y_pred, 1e-15, 1 - 1e-15)
    return -np.sum(Y_true * np.log(Y_pred))


_X = TypeVar("_X", bound=np.ndarray | float)


class _Softmax:
    def __call__(self, x: _X) -> _X:
        if isinstance(x, np.ndarray):
            return (exp_x := np.exp(x - np.max(x, axis=1, keepdims=True))) / np.sum(
                exp_x, axis=1, keepdims=True
            )
        else:
            # TODO: fix-types
            return cast(_X, 1.0)

    def deriv(self, x: _X) -> _X:
        # The derivative of softmax is more complex and depends on the specific context
        # For neural networks with cross-entropy loss, this derivative is often
        # simplified in the backpropagation calculation
        # This implementation is meant to be used with cross-entropy loss
        # where the combined derivative simplifies
        if isinstance(x, np.ndarray):
            # TODO: fix-types
            return cast(_X, np.ones_like(x))
        else:
            # TODO: fix-types
            return cast(_X, 0.0)


softmax = _Softmax()


class _ReLU:
    def __call__(self, x: _X) -> _X:
        if isinstance(x, np.ndarray):
            # TODO: fix-types
            return cast(_X, np.maximum(0, x))
        else:
            # TODO: fix-types
            return cast(_X, max(0, x))

    def deriv(self, x: _X) -> _X:
        if isinstance(x, np.ndarray):
            # TODO: fix-types
            return cast(_X, np.where(x > 0, 1, 0))
        else:
            # TODO: fix-types
            return cast(_X, 1 if x > 0 else 0)


ReLU = _ReLU()


class _Noop:
    def __call__(self, x: _X) -> _X:
        return x

    def deriv(self, x: _X) -> _X:
        if isinstance(x, np.ndarray):
            # TODO: fix-types
            return cast(_X, np.ones_like(x))
        else:
            # TODO: fix-types
            return cast(_X, 1.0)


noop = _Noop()
