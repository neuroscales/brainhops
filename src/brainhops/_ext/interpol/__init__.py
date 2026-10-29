# This package is a port of torch-interpol 0.3.0 to NumPy and CuPy.
# Copyright (c) 2021 Yael Balbastre. Released under the MIT license,
# whose text is in the LICENSE file of this package.
"""B-spline interpolation of arrays, with NumPy or CuPy.

This package resamples arrays at arbitrary coordinates with B-splines of
order 0 to 7, and computes the spline coefficients of an array. It is a
port of [torch-interpol](https://github.com/balbasty/torch-interpol)
0.3.0, written for PyTorch, to array code that runs with NumPy or with
CuPy. The functions run with the package of their input, so that an
array on a GPU is processed on the GPU, and CuPy is only imported when
it is installed. The package does not depend on brainhops.

The functions also take dask arrays, whose blocks may be NumPy or CuPy
arrays, and then return dask arrays. Each task of the result runs the
same code on the part of the input that it needs, and the results equal
those of the whole arrays to rounding errors. Dask is only imported when
an argument is a dask array.

Compared with `scipy.ndimage.map_coordinates`, [`pull`][] takes one
order and one boundary condition per axis, and takes batch axes, whose
items it processes together rather than one at a time. Coordinates are
in voxels, the last axis of the grid holds them, and the sample `i` of
an axis lies at the coordinate `i`. As in torch-interpol, the grid has
as many output spatial axes as coordinates, and its leading axes are
batch axes that broadcast against those of the input.

Beyond the edges of the input, the values follow the boundary condition.
The argument `clip` of [`pull`][] sets margins, in voxels, beyond which
the values are zero instead. It replaces the argument `extrapolate` of
torch-interpol: `extrapolate=False` becomes `clip=0`, and
`extrapolate="edges"` becomes `clip=0.5`.

A spline of order 2 or more does not pass through its coefficients.
[`spline_coeff`][] computes the coefficients of the spline that passes
through the values of an array, and [`pull`][] calls it when
`prefilter` is true. These coefficients are exact for every boundary
condition: sampling them at the integer coordinates with the same order
and boundary gives back the values, to rounding errors.

The boundary condition says how the array continues beyond its edges,
which matters near the edges for every order above 0. Each condition
has several names: the name used by SciPy, when SciPy has the same
condition, the names used by torch-interpol, and the name of the
discrete transform whose implicit extension it is. The extensions of an
array `a b c d` are as follows.

| SciPy           | torch-interpol            | extension                   |
| --------------- | ------------------------- | --------------------------- |
| `grid-constant` | `zero`, `zeros`           | `0  0  0 [a b c d] 0  0  0` |
| `nearest`       | `replicate`, `nearest`    | `a  a  a [a b c d] d  d  d` |
| `mirror`        | `dct1`, `mirror`          | `d  c  b [a b c d] c  b  a` |
| `reflect`       | `dct2`, `reflect`         | `c  b  a [a b c d] d  c  b` |
| `grid-wrap`     | `dft`                     | `b  c  d [a b c d] a  b  c` |
|                 | `dst1`, `antimirror`      | `-b -a 0 [a b c d] 0 -d -c` |
|                 | `dst2`, `antireflect`     | `-c -b -a [a b c d] -d -c -b` |

SciPy also accepts `grid-mirror` for `reflect`, and the package accepts
it too. The SciPy modes `constant` and `wrap` are refused, because the
package implements neither of them. The SciPy mode `wrap` is a shifted
periodic extension, unlike the `wrap` of torch-interpol, which is
`grid-wrap`, so the name is refused rather than read in either sense.

The prefilter of torch-interpol applied the filter of `dct1` to the
zero boundary and the filter of `dct2` to the replicate boundary, and it
returned its input unchanged for an axis of length 1. Its coefficients
therefore did not interpolate the values near the edges, at every order
for the zero boundary and from order 4 for the replicate boundary. The
port computes them exactly, as explained in [`spline_coeff1d`][].
"""

from ._coeff import spline_coeff, spline_coeff1d
from ._pull import pull

__all__ = ["pull", "spline_coeff", "spline_coeff1d"]
