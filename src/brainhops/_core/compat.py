__all__ = ["cache", "own_annotations", "partial", "PLACEHOLDER", "StrEnum"]
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


def own_annotations(
    owner: tx.Union[type, tx.Mapping[str, tx.Any]],
) -> tx.Dict[str, tx.Any]:
    """
    The annotations a class declares itself (not those it inherits), read
    from the class or from the namespace its metaclass receives.

    Up to Python 3.13 they are in `__annotations__`. From 3.14 (PEP
    649/749) they are lazy: a class (or its namespace) carries an
    annotate function instead, which is evaluated here in the
    `FORWARDREF` format, so a name that is not defined yet becomes a
    `ForwardRef` instead of raising.
    """
    namespace = owner if not isinstance(owner, type) else owner.__dict__
    if "__annotations__" in namespace:
        return dict(namespace["__annotations__"])
    if sys.version_info < (3, 14):
        return {}
    import annotationlib

    if isinstance(owner, type):
        return dict(
            annotationlib.get_annotations(
                owner, format=annotationlib.Format.FORWARDREF
            )
        )
    annotate = annotationlib.get_annotate_from_class_namespace(namespace)
    if annotate is None:
        return {}
    return dict(
        annotationlib.call_annotate_function(
            annotate, annotationlib.Format.FORWARDREF
        )
    )
