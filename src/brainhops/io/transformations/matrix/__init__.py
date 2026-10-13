"""
Plain affine matrices stored in text, NumPy or MATLAB files.

The file holds only numbers, so the caller states their meaning when the file
is read, as described in
[`MatrixAffine`][brainhops.io.transformations.matrix.MatrixAffine]. Each
container has its own reader: [`TxtMatrixAffine`][], [`CsvMatrixAffine`][],
[`TsvMatrixAffine`][], [`NpyMatrixAffine`][], [`NpzMatrixAffine`][] and
[`MatMatrixAffine`][], which hands the file to [`MatLegacyMatrixAffine`][] or
[`Mat73MatrixAffine`][] according to its MATLAB version.

!!! example "Reading a 1-based voxel-to-voxel matrix saved by MATLAB"
    ```python
    from brainhops.io.transformations import load

    xform = load(
        "M.mat", hint="matrix", variable="M",
        input="voxel", output="voxel", index_base=1,
    )
    ```
"""

__all__ = [
    "MatrixAffine",
    "TxtMatrixAffine",
    "CsvMatrixAffine",
    "TsvMatrixAffine",
    "NpyMatrixAffine",
    "NpzMatrixAffine",
    "MatMatrixAffine",
    "MatLegacyMatrixAffine",
    "Mat73MatrixAffine",
]

from ._xform import (
    CsvMatrixAffine,
    Mat73MatrixAffine,
    MatLegacyMatrixAffine,
    MatMatrixAffine,
    MatrixAffine,
    NpyMatrixAffine,
    NpzMatrixAffine,
    TsvMatrixAffine,
    TxtMatrixAffine,
)
