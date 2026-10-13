"""
Plain affine matrices stored in text, NumPy or MATLAB files.

The file holds only numbers, so the caller states their meaning when the file
is read, as described in
[`MatrixAffine`][brainhops.io.transformations.matrix.MatrixAffine]. Each
container has its own reader: [`TxtMatrixAffine`][], [`CsvMatrixAffine`][],
[`TsvMatrixAffine`][], [`NpyMatrixAffine`][], [`NpzMatrixAffine`][] and
[`MatMatrixAffine`][], which hands the file to [`MatLegacyMatrixAffine`][] or
[`Mat73MatrixAffine`][] according to its MATLAB version.

A plain matrix file is short and is read in one pass. Its matrix, and
the conventions that the caller stated when the file was read, are held
as a [`MatrixAffineRaw`][] record by the metadata of the file, such as
[`TxtMatrixMetadata`][] or [`NpzMatrixMetadata`][], and the
transformation decodes its matrix and its coordinate systems from that
record. These formats read files and do not write them.

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
    "MatrixAffineRaw",
    "MatrixMetadata",
    "TxtMatrixMetadata",
    "CsvMatrixMetadata",
    "TsvMatrixMetadata",
    "NpyMatrixMetadata",
    "NpzMatrixMetadata",
    "MatLegacyMatrixMetadata",
    "Mat73MatrixMetadata",
]

from ._metadata import (
    CsvMatrixMetadata,
    Mat73MatrixMetadata,
    MatLegacyMatrixMetadata,
    MatrixMetadata,
    NpyMatrixMetadata,
    NpzMatrixMetadata,
    TsvMatrixMetadata,
    TxtMatrixMetadata,
)
from ._raw import MatrixAffineRaw
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

# Importing the private converters registers them with the data model,
# so that a conversion into one of these formats, such as
# `t.to(TxtMatrixAffine)`, relabels the transformation.
from . import _converters  # noqa: E402, F401  isort: skip
