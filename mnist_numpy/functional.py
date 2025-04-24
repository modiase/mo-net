from collections.abc import Callable
from typing import TypeVar

_T = TypeVar("_T")
Thunk = Callable[[], _T]


def evaluate(lazy: _T | Thunk[_T]) -> _T:
    if callable(lazy):
        return lazy()
    return lazy
