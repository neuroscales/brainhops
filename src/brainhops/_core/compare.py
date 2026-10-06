"""Comparing values that may be arrays."""

__all__ = ["differs"]

# externals
import typing_extensions as tx


def differs(a: tx.Any, b: tx.Any) -> bool:
    """Whether two values differ, without trusting `!=` on arrays (whose
    `!=` is an array, or raises)."""
    if a is b:
        return False
    try:
        return bool(a != b)
    except Exception:
        return True
