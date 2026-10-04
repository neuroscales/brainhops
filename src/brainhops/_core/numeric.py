"""Helpers for numbers as files store them."""

__all__ = ["float32_repr"]

# externals
import numpy as np
import typing_extensions as tx


def float32_repr(value: tx.Any) -> float:
    """A value stored in single precision, as the shortest decimal that
    reads back to it (`0.3`, not `0.30000001192092896`)."""
    return float(str(np.float32(value)))
