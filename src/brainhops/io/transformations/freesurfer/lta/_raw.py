"""The record of an LTA file, which holds the whole file as it is parsed."""

__all__ = ["LtaRaw"]

import typing_extensions as tx
from bagof.magic import Factory

from ._enums import LtaMatrixType, LtaType, LtaValidity
from ._parser import (
    LtaReaderWriter,
    MatrixReaderWriter,
    VolumeInfoReaderWriter,
)

_2Ints = tx.Tuple[int, int]
_3Ints = tx.Tuple[int, int, int]
_3Floats = tx.Tuple[float, float, float]
_MatrixFloat = tx.Tuple[tx.Tuple[float, ...], ...]
_MatrixComplex = tx.Tuple[tx.Tuple[complex, ...], ...]
_Matrix = tx.Union[_MatrixFloat, _MatrixComplex]


class LtaRaw(LtaReaderWriter):
    """Record of an LTA file, which holds the content of the file as parsed.

    An LTA file is short and is read in one pass, so the record holds the
    whole file: its type, its matrix and the geometries of its source and
    destination volumes. The metadata of an LTA file,
    [`LtaMetadata`][brainhops.io.transformations.freesurfer.lta.LtaMetadata],
    holds a record, and an
    [`LtaTransformation`][brainhops.io.transformations.freesurfer.lta.LtaTransformation]
    reads its matrix and its coordinate systems from the record of its
    metadata. A record is never changed in place, so that several objects
    can share it.

    Reading and writing are implemented by `LtaReaderWriter`,
    `MatrixReaderWriter` and `VolumeInfoReaderWriter`.

    !!! note "Reference"
        https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/LtaFormat
    """

    class Affine(MatrixReaderWriter):
        """ASCII-encoded matrix.

        This encoding is used throughout FreeSurfer, not only in LTA files, and
        can hold any two-dimensional real or complex matrix. The first header
        value is 1 for a real matrix and 2 for a complex one, and the second
        and third values are the numbers of rows and columns. In an LTA file
        the matrix is always a real 4x4 matrix, but the parser handles any size
        and type. The `matrix` attribute is a tuple of tuples, not a NumPy
        array.

        !!! example "Example"
            A 4x4 real matrix:
            ```
            1 4 4
            +1.141600  +0.018630  +0.010876  -23.066311
            -0.019849  +1.142709  +0.150979  -29.566288
            -0.010058  -0.155729  +0.919673  +26.393215
            +0.000000  +0.000000  +0.000000  +1.000000
            ```

            A 4x4 complex matrix:
            ```
            2 4 4
            +1.141600 +0.0   +0.018630 +0.0   +0.010876 +0.0   -23.066311 +0.0
            -0.019849 +0.0   +1.142709 +0.0   +0.150979 +0.0   -29.566288 +0.0
            -0.010058 +0.0   -0.155729 +0.0   +0.919673 +0.0   +26.393215 +0.0
            +0.000000 +0.0   +0.000000 +0.0   +0.000000 +0.0   +1.000000 +0.0
            ```
        """

        matrix: _Matrix = ()

        @property
        def matrix_type(self) -> LtaMatrixType:
            """The type of the matrix, determined from its contents."""
            if not self.matrix:
                return LtaMatrixType.UNKNOWN_MATRIX
            if isinstance(self.matrix[0][0], complex):
                return LtaMatrixType.COMPLEX_MATRIX
            if isinstance(self.matrix[0][0], float):
                return LtaMatrixType.REAL_MATRIX
            return LtaMatrixType.UNKNOWN_MATRIX

        @property
        def dtype(self) -> tx.Optional[type]:
            """The Python type of the elements.

            The type is `float` for a real matrix, `complex` for a complex one,
            and `None` when the type is unknown.
            """
            if self.matrix_type == LtaMatrixType.COMPLEX_MATRIX:
                return complex
            if self.matrix_type == LtaMatrixType.REAL_MATRIX:
                return float
            return None

        @property
        def shape(self) -> _2Ints:
            """The number of rows and columns."""
            if not self.matrix:
                return (0, 0)
            if not self.matrix[0]:
                return (len(self.matrix), 0)
            return (len(self.matrix), len(self.matrix[0]))

    class VolumeInfo(VolumeInfoReaderWriter):
        """Geometry of a volume."""

        valid: LtaValidity = LtaValidity.VOLUME_INFO_INVALID
        filename: str = ""  # file name of the volume
        volume: _3Ints = (0, 0, 0)  # 3-D shape
        voxelsize: _3Floats = (1.0, 1.0, 1.0)  # voxel size
        xras: _3Floats = (1.0, 0.0, 0.0)  # columns of the phys2ras matrix
        yras: _3Floats = (0.0, 1.0, 0.0)
        zras: _3Floats = (0.0, 0.0, 1.0)
        cras: _3Floats = (0.0, 0.0, 0.0)

    class SrcVolumeInfo(VolumeInfo):
        """Geometry of the source volume."""

        NAME = "src"

    class DstVolumeInfo(VolumeInfo):
        """Geometry of the destination volume."""

        NAME = "dst"

    type: LtaType = LtaType.LINEAR_VOX_TO_VOX
    nxforms: int = 1
    mean: _3Floats = (0.0, 0.0, 0.0)
    sigma: float = 0.0
    affine: Affine = Factory(Affine)
    label: tx.Optional[int] = None
    src: tx.Optional[SrcVolumeInfo] = None
    dst: tx.Optional[DstVolumeInfo] = None
