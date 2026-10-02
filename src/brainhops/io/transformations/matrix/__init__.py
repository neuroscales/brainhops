"""Plain affine matrices stored in text, NumPy (`.npy`, `.npz`) or MATLAB
(`.mat`) files.

The file stores only the numbers; what they mean (spaces, index base,
direction, vector convention) is given by the caller when reading. See
[`MatrixAffine`][brainhops.io.transformations.matrix.MatrixAffine], the
abstract base of one reader per container: [`TxtMatrixAffine`][],
[`CsvMatrixAffine`][], [`TsvMatrixAffine`][], [`NpyMatrixAffine`][],
[`NpzMatrixAffine`][], and [`MatMatrixAffine`][], which dispatches to
[`MatLegacyMatrixAffine`][] (MATLAB v4-v7) or [`Mat73MatrixAffine`][]
(MATLAB v7.3).

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
