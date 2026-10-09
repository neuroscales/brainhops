__all__ = ["cache", "own_annotations", "partial", "PLACEHOLDER", "StrEnum"]
import sys
from functools import lru_cache
from functools import partial as _partial

import typing_extensions as tx

PLACEHOLDER = object()


def cache(func: tx.Callable) -> tx.Callable:
    """Wrap a function in an unbounded cache.

    The function is a replacement for `functools.cache` on Pythons that lack
    it, and is equivalent to [`lru_cache`][functools.lru_cache] with
    `maxsize=None`.
    """

    return lru_cache(maxsize=None)(func)


class partial(_partial):
    """Partial application that accepts placeholders.

    The class extends [`partial`][functools.partial] so that the saved
    positional arguments may contain the [`PLACEHOLDER`][] sentinel. When the
    partial object is called, each placeholder is replaced by the next call
    argument, in order, and the remaining call arguments are appended.
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
    """Return the annotations that a class declares itself.

    Inherited annotations are not included. The owner is either a class or
    the namespace mapping that a metaclass receives before the class is
    created. Up to Python 3.13, the annotations are read from
    `__annotations__` in the namespace. From Python 3.14 (PEP 649 and
    PEP 749), the lazy annotate function is evaluated in the `FORWARDREF`
    format, so that undefined names become forward references instead of
    raising an error. An empty dictionary is returned when the class has no
    annotations.
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
