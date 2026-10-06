"""
Helpers for numbers as files store them.

Many file formats store numbers in single precision, sometimes in
another unit than the one brainhops uses (milliseconds instead of
seconds, radians instead of degrees). Converted to double precision, such
a number reads as `0.30000001192092896` rather than as the `0.3` that was
written. The functions of this module recover the short decimal that
was meant.
"""

__all__ = ["float32_repr", "shortest_decimal"]

# stdlib
import math

# externals
import numpy as np
import typing_extensions as tx

# A double has at most 17 significant decimal digits.
_MAX_DIGITS = 17


def shortest_decimal(
    value: float,
    encode: tx.Optional[tx.Callable[[float], float]] = None,
) -> float:
    """
    Find the shortest decimal that a file stores as the same single
    precision number as `value`.

    A writer stores a value `d` as the single-precision number
    `float32(encode(d))`, where `encode` converts the value into the unit
    of the file. A reader that decodes that number in double precision
    obtains `value`, which is close to the number that was written, but
    rarely equal to it. This function returns the decimal with the fewest
    significant digits that the writer would store as the same number.
    That decimal is the value the reader reports: it reads back exactly as
    it was written, and writing it again stores the same bits.

    With the identity as `encode`, this is the question that the shortest
    round-trip representation of a single-precision number answers, and
    the result is the value that `str(numpy.float32(value))` prints.

    Parameters
    ----------
    value : float
        The decoded value, in double precision.
    encode : callable, optional
        The function that converts a value into what the file stores,
        before the rounding to single precision. By default, the identity.

    Returns
    -------
    float
        The shortest decimal that is stored as the same number. A value
        that is zero or not finite is returned as it is.

    Examples
    --------
    ```pycon
    >>> import math
    >>> import numpy as np
    >>> shortest_decimal(float(np.float32(0.3)))
    0.3
    >>> stored = np.float32(2000.7)  # a repetition time, in milliseconds
    >>> float(stored) * 1e-3
    2.000699951171875
    >>> shortest_decimal(float(stored) * 1e-3, lambda s: s / 1e-3)
    2.0007
    >>> radians = np.float32(math.radians(9))  # a flip angle
    >>> math.degrees(float(radians))
    9.000000250447817
    >>> shortest_decimal(math.degrees(float(radians)), math.radians)
    9.0
    ```
    """
    value = float(value)
    if not value or not math.isfinite(value):
        return value
    if encode is None:

        def encode(x: float) -> float:
            return x

    target = np.float32(encode(value))
    for digits in range(1, _MAX_DIGITS + 1):
        candidate = float(f"{value:.{digits}g}")
        if np.float32(encode(candidate)) == target:
            return candidate
    return value


def float32_repr(value: tx.Any) -> float:
    """
    The shortest decimal that is stored as the same single-precision
    number as `value`.

    Parameters
    ----------
    value : float
        A number, usually one read from a single-precision slot.

    Returns
    -------
    float
        The shortest decimal with the same single-precision value, such
        as `0.3` for the single-precision number `0.30000001192092896`.
    """
    return shortest_decimal(float(np.float32(value)))
