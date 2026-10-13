"""Comparison of values that may be arrays."""

__all__ = ["differs"]

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
