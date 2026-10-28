# This file is part of a port of torch-interpol 0.3.0 to NumPy and
# CuPy. Copyright (c) 2021 Yael Balbastre. Released under the MIT
# license, whose text is in the LICENSE file of this package.
"""Boundary conditions, which say how an array continues past its edges.

Spline interpolation near an edge reads samples that lie outside the
array. A boundary condition defines those samples from the ones inside.
In this module, a boundary condition maps an index that may lie outside
`[0, n - 1]` to an index inside it, and to a sign or a factor that
multiplies the sample read there.
"""

# stdlib
from types import ModuleType

# dependencies
import typing_extensions as tx

_BOUNDS = ("zero", "replicate", "dct1", "dct2", "dst1", "dst2", "dft")
"""The names that the package uses for the boundary conditions."""

_ALIASES = {
    "zero": "zero",
    "zeros": "zero",
    "grid-constant": "zero",
    "replicate": "replicate",
    "nearest": "replicate",
    "dct1": "dct1",
    "mirror": "dct1",
    "dct2": "dct2",
    "reflect": "dct2",
    "grid-mirror": "dct2",
    "dst1": "dst1",
    "antimirror": "dst1",
    "dst2": "dst2",
    "antireflect": "dst2",
    "dft": "dft",
    "grid-wrap": "dft",
}

_REFUSED = {
    "wrap": (
        "The name 'wrap' is ambiguous: torch-interpol uses it for the "
        "periodic boundary, which SciPy calls 'grid-wrap', while the "
        "'wrap' mode of SciPy is a different, shifted extension. Use "
        "'grid-wrap' or 'dft' for the periodic boundary."
    ),
    "constant": (
        "The 'constant' mode of SciPy returns the fill value as soon as "
        "a coordinate lies beyond the centre of an edge sample, which "
        "this package does not implement. Use 'grid-constant' or 'zero' "
        "for a zero extension of the array."
    ),
}


def _normalize_bound(bound: tx.Any) -> str:
    """Return the name that the package uses for a boundary condition.

    Parameters
    ----------
    bound : str
        A name of SciPy or of torch-interpol, from the table in the
        documentation of the package.

    Returns
    -------
    str
        One of the names in `_BOUNDS`.

    Raises
    ------
    ValueError
    """
    key = str(bound).lower()
    if key in _REFUSED:
        raise ValueError(_REFUSED[key])
    if key not in _ALIASES:
        raise ValueError(f"Unknown boundary condition: {bound!r}.")
    return _ALIASES[key]


def _bound_index(
    xp: ModuleType, i: tx.Any, n: int, bound: str
) -> tx.Tuple[tx.Any, tx.Optional[tx.Any]]:
    """Map indices that may lie outside an axis to indices inside it.

    The sign is the factor by which the sample at the returned index is
    multiplied. It is -1 where an antisymmetric boundary flips the
    sample, and 0 where the boundary reads zeros. It is None when every
    factor is 1. The indices are mapped in place where possible, so as
    to allocate few temporary arrays, which matters for large grids.

    Parameters
    ----------
    xp : module
        The module `numpy` or `cupy` of the indices.
    i : array
        The indices, of an integer data type. The array is overwritten.
    n : int
        The length of the axis.
    bound : str
        One of the names in `_BOUNDS`.

    Returns
    -------
    index : array
        The indices inside `[0, n - 1]`.
    sign : array or None
        The factors, as integers in {-1, 0, 1}, or None.
    """
    if bound == "replicate":
        return xp.clip(i, 0, n - 1, out=i), None
    if bound == "zero":
        inside = (i >= 0) & (i < n)
        return xp.clip(i, 0, n - 1, out=i), inside
    if bound == "dft":
        return xp.remainder(i, n, out=i), None
    if bound == "dct1":
        if n == 1:
            i[...] = 0
            return i, None
        # In one period of length 2 * (n - 1), the index p reads the
        # sample min(p, 2 * (n - 1) - p).
        n2 = 2 * (n - 1)
        xp.abs(i, out=i)
        xp.remainder(i, n2, out=i)
        return xp.minimum(i, n2 - i, out=i), None
    if bound in ("dct2", "dst2"):
        # In one period of length 2 * n, the index p reads the sample
        # min(p, 2 * n - 1 - p).
        n2 = 2 * n
        xp.remainder(i, n2, out=i)
        sign = None
        if bound == "dst2":
            # The sign changes at every edge that the index crosses,
            # that is, in the second half of every period.
            sign = 1 - 2 * (i >= n)
        return xp.minimum(i, n2 - 1 - i, out=i), sign
    if bound == "dst1":
        # The extended array has period 2 * (n + 1). Inside one period,
        # the samples are [x, 0, -x reversed, 0], starting at index 0.
        n2 = 2 * (n + 1)
        period = (i + 1) % n2 - 1
        flipped = period >= n
        index = xp.where(flipped, n2 - 2 - period, period)
        zero = (period == -1) | (period == n)
        sign = xp.where(flipped, -1, 1)
        sign = xp.where(zero, 0, sign)
        return xp.clip(index, 0, n - 1), sign
    raise ValueError(f"Unknown boundary condition: {bound!r}.")
