from collections.abc import Callable
from typing import TypeAlias, TypeVar

_T = TypeVar("_T")
Thunk: TypeAlias = Callable[[], _T]


def evaluate(lazy: _T | Thunk[_T]) -> _T:
    if callable(lazy):
        return lazy()
    return lazy
