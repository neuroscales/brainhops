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
import numpy as np
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


def normalize_bound(bound: tx.Any) -> str:
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


def bound_index(
    xp: ModuleType,
    i: tx.Any,
    n: int,
    bound: str,
    weight: tx.Any,
    scratch: tx.Optional[tx.Any] = None,
    factor: tx.Optional[tx.Any] = None,
) -> tx.Any:
    """Map indices that may lie outside an axis to indices inside it.

    Interpolation near an edge reads samples outside the axis, and the
    boundary condition says which sample inside the axis each of them
    copies. An antisymmetric boundary also flips the sign of the copy,
    and the zero boundary replaces it with zero. This function brings
    the indices inside the axis and multiplies the weights of the
    samples by these factors, -1 or 0, where they apply. Both arrays
    are modified in place, so that no array of their size is allocated
    when the scratch arrays are given, which matters for large grids.

    Parameters
    ----------
    xp : module
        The module `numpy` or `cupy` of the indices.
    i : array
        The indices, as a contiguous array of the data type `intp`. The
        array is overwritten with the indices inside the axis.
    n : int
        The length of the axis.
    bound : str
        One of the names in `_BOUNDS`.
    weight : array
        The weights of the samples, with the shape of `i` and a floating
        data type. They are multiplied in place by the factors of the
        boundary.
    scratch : array, optional
        An integer array of the shape of `i`, which is overwritten. By
        default, a new array is allocated when the boundary needs one.
    factor : array, optional
        An array of the shape of `i` and the data type of `weight`,
        which is overwritten. By default, a new array is allocated when
        the boundary needs one.

    Returns
    -------
    array
        The array `i`, which holds the indices inside `[0, n - 1]`.

    Raises
    ------
    ValueError
    """
    if bound == "replicate":
        return xp.clip(i, 0, n - 1, out=i)
    if bound == "dft":
        return xp.remainder(i, n, out=i)
    if bound == "dct1" and n == 1:
        i[...] = 0
        return i
    if bound not in ("zero", "dct1", "dct2", "dst1", "dst2"):
        raise ValueError(f"Unknown boundary condition: {bound!r}.")
    if scratch is None:
        scratch = xp.empty_like(i)
    if factor is None:
        factor = xp.empty(i.shape, dtype=weight.dtype)
    if bound == "zero":
        # A negative index, read as an unsigned integer, is larger than
        # any length, so that a single comparison finds the indices
        # inside the axis.
        xp.less(i.view(np.uintp), n, out=factor)
        xp.multiply(weight, factor, out=weight)
        return xp.clip(i, 0, n - 1, out=i)
    if bound == "dct1":
        # In one period of length 2 * (n - 1), the index p reads the
        # sample min(p, 2 * (n - 1) - p).
        n2 = 2 * (n - 1)
        xp.abs(i, out=i)
        xp.remainder(i, n2, out=i)
        xp.subtract(n2, i, out=scratch)
        return xp.minimum(i, scratch, out=i)
    if bound in ("dct2", "dst2"):
        # In one period of length 2 * n, the index p reads the sample
        # min(p, 2 * n - 1 - p).
        n2 = 2 * n
        xp.remainder(i, n2, out=i)
        xp.subtract(n2 - 1, i, out=scratch)
        if bound == "dst2":
            # The sign changes at every edge that the index crosses,
            # that is, in the second half of every period, where p is
            # larger than 2 * n - 1 - p. The two are never equal.
            xp.subtract(scratch, i, out=factor)
            xp.sign(factor, out=factor)
            xp.multiply(weight, factor, out=weight)
        return xp.minimum(i, scratch, out=i)
    # The extended array of dst1 has period 2 * (n + 1). Inside one
    # period, the samples are [x, 0, -x reversed, 0], starting at index
    # 0. With q = (i + 1) mod 2 * (n + 1), the samples 1 <= q <= n are
    # x[q - 1], the samples q = 0 and q = n + 1 are zero, and the
    # samples q >= n + 2 are -x[2 * n + 1 - q]. The factor is therefore
    # sign(n + 1 - q) * sign(q), and the index is
    # min(q - 1, 2 * n + 1 - q), clipped to the axis.
    n2 = 2 * (n + 1)
    xp.add(i, 1, out=i)
    xp.remainder(i, n2, out=i)
    xp.subtract(n + 1, i, out=factor)
    xp.sign(factor, out=factor)
    xp.multiply(weight, factor, out=weight)
    xp.sign(i, out=factor)
    xp.multiply(weight, factor, out=weight)
    xp.subtract(n2 - 1, i, out=scratch)
    xp.subtract(i, 1, out=i)
    xp.minimum(i, scratch, out=i)
    return xp.clip(i, 0, n - 1, out=i)
