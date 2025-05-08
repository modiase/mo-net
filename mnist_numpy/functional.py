from collections.abc import Callable
from typing import TypeVar, cast

_T = TypeVar("_T")
Thunk = Callable[[], _T]


def evaluate(lazy: _T | Thunk[_T]) -> _T:
    if callable(lazy):
        return cast(_T, lazy())
    return lazy
