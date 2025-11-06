from collections.abc import Callable
from typing import TypeVar, overload

_T = TypeVar("_T")
Thunk = Callable[[], _T]


@overload
def evaluate(lazy: Thunk[_T]) -> _T: ...


@overload
def evaluate(lazy: _T) -> _T: ...


def evaluate(lazy: _T | Thunk[_T]) -> _T:
    if callable(lazy) and not isinstance(lazy, type):
        return lazy()  # type: ignore[call-overload]
    return lazy  # type: ignore[return-value]
