# This file is part of a port of torch-interpol 0.3.0 to NumPy and
# CuPy. Copyright (c) 2021 Yael Balbastre. Released under the MIT
# license, whose text is in the LICENSE file of this package.
"""Weights of the B-splines of orders 0 to 7.

A B-spline of order `p` is a piecewise polynomial of degree `p` that
is nonzero on the interval `(-(p + 1) / 2, (p + 1) / 2)`. Interpolation
at a coordinate `x` adds the `p + 1` samples nearest to `x`, each
weighted by the spline evaluated at its distance to `x`. In this
package, each of these samples is called a tap.

Whatever the coordinate, the distance of a given tap falls in the same
polynomial piece of the spline. [`tap_weight`][] therefore evaluates a
single polynomial for each tap, instead of selecting the piece sample
by sample.
"""

# stdlib
import math
from types import ModuleType

# dependencies
import typing_extensions as tx

MAX_ORDER = 7
"""The highest spline order that the package implements."""


def first_tap(order: int) -> float:
    """Return the offset that gives the first sample a coordinate reads.

    The first sample read at a coordinate `x` is `floor(x - offset)`,
    and the following ones are the next `order` samples.

    Parameters
    ----------
    order : int
        The spline order.

    Returns
    -------
    float
        The offset `(order - 1) / 2`.
    """
    return (order - 1) / 2


# The polynomial pieces of the splines, as functions of the absolute
# distance x, with piece 0 nearest to the centre. For even orders,
# piece 0 is even in x and also takes negative distances. The pieces
# are written with their coefficients, from the highest power of x to
# the constant term, and evaluated by Horner's rule. The outermost
# piece of the spline of order p is (c - x)**p / p! with
# c = (p + 1) / 2, and it is evaluated in that form, which is accurate
# near x = c, where the expanded polynomial would lose its precision.
_INNER_PIECES: tx.Dict[int, tx.Tuple[tx.Tuple[float, ...], ...]] = {
    1: (),
    2: ((-1.0, 0.0, 3 / 4),),
    3: ((1 / 2, -1.0, 0.0, 2 / 3),),
    4: (
        (1 / 4, 0.0, -5 / 8, 0.0, 115 / 192),
        (-1 / 6, 5 / 6, -5 / 4, 5 / 24, 55 / 96),
    ),
    5: (
        (-1 / 12, 1 / 4, 0.0, -1 / 2, 0.0, 11 / 20),
        (1 / 24, -3 / 8, 5 / 4, -7 / 4, 5 / 8, 17 / 40),
    ),
    6: (
        (-1 / 36, 0.0, 7 / 48, 0.0, -77 / 192, 0.0, 5887 / 11520),
        (
            1 / 48,
            -7 / 48,
            21 / 64,
            -35 / 288,
            -91 / 256,
            -7 / 768,
            7861 / 15360,
        ),
        (
            -1 / 120,
            7 / 60,
            -21 / 32,
            133 / 72,
            -329 / 128,
            1267 / 960,
            1379 / 7680,
        ),
    ),
    7: (
        (1 / 144, -1 / 36, 0.0, 1 / 9, 0.0, -1 / 3, 0.0, 151 / 315),
        (
            -1 / 240,
            1 / 20,
            -7 / 30,
            1 / 2,
            -7 / 18,
            -1 / 10,
            -7 / 90,
            103 / 210,
        ),
        (
            1 / 720,
            -1 / 36,
            7 / 30,
            -19 / 18,
            49 / 18,
            -23 / 6,
            217 / 90,
            -139 / 630,
        ),
    ),
}


def tap_weight(
    xp: ModuleType,
    order: int,
    tap: int,
    t: tx.Any,
    out: tx.Optional[tx.Any] = None,
    scratch: tx.Optional[tx.Any] = None,
) -> tx.Any:
    """Return the weight of one tap of a spline.

    The coordinate `x` reads the samples `i0 + tap`, with `i0` given by
    [`first_tap`][], and `t = x - i0` lies in the interval
    `[(order - 1) / 2, (order + 1) / 2)`. The weight of the sample
    `i0 + tap` is the spline evaluated at `t - tap`. The polynomial is
    evaluated in place in `out`, with `scratch` holding the distance,
    so that no other array is allocated.

    Parameters
    ----------
    xp : module
        The module `numpy` or `cupy` of the distances.
    order : int
        The spline order, from 0 to 7.
    tap : int
        The tap, from 0 to `order`.
    t : array
        The distances from the first sample to the coordinates.
    out : array, optional
        The array that receives the weights, with the shape of `t` and
        a floating data type. By default, a new array is allocated.
    scratch : array, optional
        An array of the shape and the data type of `out`, which is
        overwritten. By default, a new array is allocated.

    Returns
    -------
    array
        The weights, which are the array `out` when it is given.

    Raises
    ------
    NotImplementedError
    """
    if order not in _INNER_PIECES and order != 0:
        raise NotImplementedError(f"Spline order {order} is not implemented.")
    if out is None:
        out = xp.empty_like(t)
    if order == 0:
        out[...] = 1
        return out
    if scratch is None:
        scratch = xp.empty_like(out)
    # The distances t - tap of a given tap span an interval of length 1
    # centred on order / 2 - tap. Its bounds are bounds of pieces of the
    # spline, so that the whole interval lies in a single piece. The
    # absolute distance is x = t - tap when the centre is not negative,
    # and x = tap - t otherwise.
    centre = order / 2 - tap
    piece = int(math.floor(abs(centre)))
    sign = 1 if centre >= 0 else -1
    pieces = _INNER_PIECES[order]
    if piece == len(pieces):
        # Outermost piece: (c - x)**order / order!, where
        # c - x = c + sign * tap - sign * t.
        c = (order + 1) / 2
        if sign > 0:
            xp.subtract(c + tap, t, out=scratch)
        else:
            xp.add(t, c - tap, out=scratch)
        xp.multiply(scratch, 1 / math.factorial(order), out=out)
        for _ in range(order - 1):
            xp.multiply(out, scratch, out=out)
        return out
    if sign > 0:
        x = xp.subtract(t, tap, out=scratch) if tap else t
    else:
        x = xp.subtract(tap, t, out=scratch)
    coefficients = pieces[piece]
    xp.multiply(x, coefficients[0], out=out)
    for c in coefficients[1:-1]:
        if c:
            xp.add(out, c, out=out)
        xp.multiply(out, x, out=out)
    return xp.add(out, coefficients[-1], out=out)
