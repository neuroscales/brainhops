"""Plain affine matrices stored in text, NumPy (`.npy`, `.npz`) or MATLAB
(`.mat`) files.

The file stores only the numbers; what they mean (spaces, index base,
direction, vector convention) is given by the caller when reading. See
[`MatrixAffine`][brainhops.io.transformations.matrix.MatrixAffine].

!!! example "Reading a 1-based voxel-to-voxel matrix saved by MATLAB"
    ```python
    from brainhops.io.transformations import load

    xform = load(
        "M.mat", hint="matrix", variable="M",
        input="voxel", output="voxel", index_base=1,
    )
    ```
"""

__all__ = ["MatrixAffine", "MatrixParser"]

from ._parser import MatrixParser
from ._xform import MatrixAffine
