__all__ = ["LTAStruct"]

# externals
import typing_extensions as _tx

# _ext
from bagof.magic import Factory

# internals
from ._enums import LTAMatrixType, LTAType, LTAValidity
from ._parser import LTAParser, MatrixParser, VolumeInfoParser

# type hints
_2Ints = _tx.Tuple[int, int]
_3Ints = _tx.Tuple[int, int, int]
_3Floats = _tx.Tuple[float, float, float]
_MatrixFloat = _tx.Tuple[_tx.Tuple[float, ...], ...]
_MatrixComplex = _tx.Tuple[_tx.Tuple[complex, ...], ...]
_Matrix = _tx.Union[_MatrixFloat, _MatrixComplex]


class LTAStruct(LTAParser):
    """
    In-memory representation of an LTA file.

    The parsing mechanisms are implemented in the parent classes:
    `LTAParser`, `MatrixParser`, and `VolumeInfoParser`.

    :: note "Reference"
        https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/LtaFormat
    """

    class Affine(MatrixParser):
        """A matrix, encoded in ASCII.

        This encoding is ubiquitous in Freesurfer (not only in LTA files)
        and can represent any 2D matrix of real or complex numbers, as
        indicated by the first value in the header (1 => real, 2 => complex).
        The second and third numbers indicate the number of rows and columns.

        In LTA files, they are always 4x4 real matrices, but the parser
        is flexible enough to handle any size and type.

        The `matrix` attribute is a tuple of tuples, not a NumPy array.

        :: example "Example"
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
        """

        matrix: _Matrix = ()

        @property
        def matrix_type(self) -> LTAMatrixType:
            """Determines the type of the matrix based on its contents."""
            if not self.matrix:
                return LTAMatrixType.UNKNOWN_MATRIX
            if isinstance(self.matrix[0][0], complex):
                return LTAMatrixType.COMPLEX_MATRIX
            if isinstance(self.matrix[0][0], float):
                return LTAMatrixType.REAL_MATRIX
            return LTAMatrixType.UNKNOWN_MATRIX

        @property
        def dtype(self) -> _tx.Optional[type]:
            """
            The Python type corresponding to the matrix type.

            Either `float` for real matrices, `complex` for complex matrices,
            or `None` if unknown.
            """
            if self.matrix_type == LTAMatrixType.COMPLEX_MATRIX:
                return complex
            if self.matrix_type == LTAMatrixType.REAL_MATRIX:
                return float
            return None

        @property
        def shape(self) -> _2Ints:
            """The shape of the matrix as a tuple (rows, columns)."""
            if not self.matrix:
                return (0, 0)
            if not self.matrix[0]:
                return (len(self.matrix), 0)
            return (len(self.matrix), len(self.matrix[0]))

    class VolumeInfo(VolumeInfoParser):
        """The geometry of a volume."""

        valid: LTAValidity = LTAValidity.VOLUME_INFO_INVALID
        filename: str = ""  # Filename of the volume
        volume: _3Ints = (0, 0, 0)  # 3D shape
        voxelsize: _3Floats = (1.0, 1.0, 1.0)  # Voxel size
        xras: _3Floats = (1.0, 0.0, 0.0)  # Columns of the phys2ras matrix
        yras: _3Floats = (0.0, 1.0, 0.0)  # "
        zras: _3Floats = (0.0, 0.0, 1.0)  # "
        cras: _3Floats = (0.0, 0.0, 0.0)  # "

    class SrcVolumeInfo(VolumeInfo):
        """The geometry of the source volume."""

        NAME = "src"

    class DstVolumeInfo(VolumeInfo):
        """The geometry of the destination volume."""

        NAME = "dst"

    type: LTAType = LTAType.LINEAR_VOX_TO_VOX
    nxforms: int = 1
    mean: _3Floats = (0.0, 0.0, 0.0)
    sigma: float = 0.0
    affine: Affine = Factory(Affine)  # Affine matrix
    label: _tx.Optional[int] = None  # Optional label
    src: _tx.Optional[SrcVolumeInfo] = None  # Source volume
    dst: _tx.Optional[DstVolumeInfo] = None  # Destination volume
