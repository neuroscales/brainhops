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
polynomial piece of the spline. [`_tap_weight`][] therefore evaluates a
single polynomial for each tap, instead of selecting the piece sample
by sample.
"""

# stdlib
import math

# dependencies
import typing_extensions as tx

_MAX_ORDER = 7
"""The highest spline order that the package implements."""


def _first_tap(order: int) -> float:
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


def _tap_weight(order: int, tap: int, t: tx.Any) -> tx.Any:
    """Return the weight of one tap of a spline.

    The coordinate `x` reads the samples `i0 + tap`, with `i0` given by
    [`_first_tap`][], and `t = x - i0` lies in the interval
    `[(order - 1) / 2, (order + 1) / 2)`. The weight of the sample
    `i0 + tap` is the spline evaluated at `t - tap`.

    Parameters
    ----------
    order : int
        The spline order, from 0 to 7.
    tap : int
        The tap, from 0 to `order`.
    t : array
        The distances from the first sample to the coordinates.

    Returns
    -------
    array
        The weights, with the shape of `t`.

    Raises
    ------
    NotImplementedError
    """
    if order == 0:
        return t * 0 + 1
    # The distances t - tap of a given tap span an interval of length 1
    # centred on order / 2 - tap. Its bounds are bounds of pieces of the
    # spline, so that the whole interval lies in a single piece.
    centre = order / 2 - tap
    piece = int(math.floor(abs(centre)))
    x = t - tap if centre >= 0 else tap - t
    return _piece(order, piece, x)


def _piece(order: int, piece: int, x: tx.Any) -> tx.Any:
    # The polynomial pieces of the splines, as functions of the absolute
    # distance x, with piece 0 nearest to the centre. For even orders,
    # piece 0 is even in x and also takes negative distances.
    if order == 1:
        return 1 - x
    if order == 2:
        if piece == 0:
            return 0.75 - x * x
        return 0.5 * (1.5 - x) ** 2
    if order == 3:
        if piece == 0:
            return (x * x * (x - 2.0) * 3.0 + 4.0) / 6.0
        return (2.0 - x) ** 3 / 6.0
    if order == 4:
        if piece == 0:
            x2 = x * x
            return x2 * (x2 * 0.25 - 0.625) + 115.0 / 192.0
        if piece == 1:
            return (
                x * (x * (x * (5.0 - x) / 6.0 - 1.25) + 5.0 / 24.0)
                + 55.0 / 96.0
            )
        return (x - 2.5) ** 4 / 24.0
    if order == 5:
        if piece == 0:
            x2 = x * x
            return x2 * (x2 * (0.25 - x / 12.0) - 0.5) + 0.55
        if piece == 1:
            return (
                x * (x * (x * (x * (x / 24.0 - 0.375) + 1.25) - 1.75) + 0.625)
                + 0.425
            )
        return (3.0 - x) ** 5 / 120.0
    if order == 6:
        if piece == 0:
            x2 = x * x
            return (
                x2 * (x2 * (7.0 / 48.0 - x2 / 36.0) - 77.0 / 192.0)
                + 5887.0 / 11520.0
            )
        if piece == 1:
            return (
                x
                * (
                    x
                    * (
                        x
                        * (
                            x * (x * (x / 48.0 - 7.0 / 48.0) + 0.328125)
                            - 35.0 / 288.0
                        )
                        - 91.0 / 256.0
                    )
                    - 7.0 / 768.0
                )
                + 7861.0 / 15360.0
            )
        if piece == 2:
            return (
                x
                * (
                    x
                    * (
                        x
                        * (
                            x * (x * (7.0 / 60.0 - x / 120.0) - 0.65625)
                            + 133.0 / 72.0
                        )
                        - 2.5703125
                    )
                    + 1267.0 / 960.0
                )
                + 1379.0 / 7680.0
            )
        return (x - 3.5) ** 6 / 720.0
    if order == 7:
        if piece == 0:
            x2 = x * x
            return (
                x2 * (x2 * (x2 * (x / 144.0 - 1.0 / 36.0) + 1.0 / 9.0) - 1 / 3)
                + 151.0 / 315.0
            )
        if piece == 1:
            return (
                x
                * (
                    x
                    * (
                        x
                        * (
                            x
                            * (x * (x * (0.05 - x / 240.0) - 7.0 / 30.0) + 0.5)
                            - 7.0 / 18.0
                        )
                        - 0.1
                    )
                    - 7.0 / 90.0
                )
                + 103.0 / 210.0
            )
        if piece == 2:
            return (
                x
                * (
                    x
                    * (
                        x
                        * (
                            x
                            * (
                                x * (x * (x / 720.0 - 1.0 / 36.0) + 7.0 / 30.0)
                                - 19.0 / 18.0
                            )
                            + 49.0 / 18.0
                        )
                        - 23.0 / 6.0
                    )
                    + 217.0 / 90.0
                )
                - 139.0 / 630.0
            )
        return (4.0 - x) ** 7 / 5040.0
    raise NotImplementedError(f"Spline order {order} is not implemented.")
