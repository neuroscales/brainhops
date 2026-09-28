__all__ = ["cache", "partial", "PLACEHOLDER", "StrEnum"]
import sys
from functools import lru_cache
from functools import partial as _partial

import typing_extensions as tx

PLACEHOLDER = object()


def cache(func: tx.Callable) -> tx.Callable:
    """
    A [`functools.lru_cache`][] that is available in Python >= 3.9.

    The `maxsize` argument is fixed to `None`, so the cache is unbounded.
    """

    return lru_cache(maxsize=None)(func)


class partial(_partial):
    """
    A [`functools.partial`][] that allows placeholders in the saved
    arguments.

    Only available in Python >= 3.9.
    """

    def __new__(cls, func: tx.Callable, /, *args, **kwargs) -> tx.Self:
        return super().__new__(cls, func, **kwargs)

    def __init__(self, func: tx.Callable, /, *args, **kwargs) -> None:
        self._saved_args = args

    def __call__(self, *call_args, **kwargs) -> tx.Any:
        args = []

        saved_args = list(self._saved_args)
        call_args = list(call_args)
        while saved_args or call_args:
            if saved_args and saved_args[0] is PLACEHOLDER:
                saved_args.pop(0)
                arg = call_args.pop(0)
            elif saved_args:
                arg = saved_args.pop(0)
            else:
                arg = call_args.pop(0)
            args.append(arg)
        return super().__call__(*args, **kwargs)


if sys.version_info.minor >= 11:
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):
        def __str__(self) -> str:
            return str(self.value)
