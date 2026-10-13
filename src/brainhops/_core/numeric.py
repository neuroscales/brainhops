"""Recovery of the short decimals behind numbers stored in files.

Files often store numbers in single precision, sometimes in a different
unit from the one used in memory, such as milliseconds instead of
seconds or radians instead of degrees. Once widened to double precision,
a stored `0.3` reads as `0.30000001192092896`. The functions in this
module recover the short decimal that was originally written.
"""

__all__ = ["float32_repr", "shortest_decimal"]

import math

import numpy as np
import typing_extensions as tx

# A double has at most 17 significant decimal digits, which bounds the
# search.
_MAX_DIGITS = 17


def shortest_decimal(
    value: float,
    encode: tx.Optional[tx.Callable[[float], float]] = None,
) -> float:
    """Return the shortest decimal that is stored as the same float32.

    A writer stores a decimal `d` as `float32(encode(d))`, where `encode`
    converts `d` into the unit of the file. A reader obtains the double
    `value`, which is close to `d` but rarely equal to it. The returned
    decimal is the shortest number that the writer could have written, and
    writing it again stores the same bits. With the default identity
    encoding, the result is the shortest representation of a float32 that
    round-trips, which is what `str(numpy.float32(value))` prints.

    Parameters
    ----------
    value : float
        Double decoded from the file.
    encode : callable, optional
        Conversion from the unit of `value` to the unit of the file. The
        default is the identity.

    Returns
    -------
    float
        The shortest decimal. Zero and non-finite values are returned as
        they are.

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
    """Return the shortest decimal that has the same float32 value.

    The value, which usually comes from a single-precision field, is first
    cast to float32. For example, `0.30000001192092896` becomes `0.3`.
    """
    return shortest_decimal(float(np.float32(value)))
