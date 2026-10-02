"""Plain affine matrices stored in text, NumPy (`.npy`, `.npz`) or MATLAB
(`.mat`) files.

The file stores only the numbers; what they mean (spaces, index base,
direction, vector convention) is given by the caller when reading. See
[`MatrixAffine`][brainhops.io.transformations.matrix.MatrixAffine], the
abstract base of one reader per container: [`TextMatrixAffine`][],
[`NpyMatrixAffine`][], [`NpzMatrixAffine`][], [`MatMatrixAffine`][]
(MATLAB v4-v7) and [`Mat73MatrixAffine`][] (MATLAB v7.3).

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
    "TextMatrixAffine",
    "NpyMatrixAffine",
    "NpzMatrixAffine",
    "MatMatrixAffine",
    "Mat73MatrixAffine",
]

from ._xform import (
    Mat73MatrixAffine,
    MatMatrixAffine,
    MatrixAffine,
    NpyMatrixAffine,
    NpzMatrixAffine,
    TextMatrixAffine,
)
