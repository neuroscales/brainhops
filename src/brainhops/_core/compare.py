"""Comparison of values that may be arrays, and of objects by identity."""

__all__ = ["ComparesByIdMixin", "differs"]

import typing_extensions as tx


def differs(a: tx.Any, b: tx.Any) -> bool:
    """Return whether two values differ.

    The comparison is robust to arrays, for which `a != b` returns an array
    or raises an exception instead of returning a boolean. Two references to
    the same object never differ, and a comparison that raises is treated as
    a difference.
    """
    if a is b:
        return False
    try:
        return bool(a != b)
    except Exception:
        return True


class ComparesByIdMixin:
    """Mixin that makes a class and all its subclasses compare by identity.

    Equality never raises, and instances are hashable. Identity comparison
    is restored on every subclass, because a subclass that also derives
    from another data model may otherwise receive a value-based `__eq__`
    and `__hash__`.
    """

    __eq__ = object.__eq__
    __ne__ = object.__ne__
    __hash__ = object.__hash__

    def __init_subclass__(cls, **kwargs: tx.Any) -> None:
        super().__init_subclass__(**kwargs)
        for name in ("__eq__", "__ne__", "__hash__"):
            setattr(cls, name, getattr(object, name))
