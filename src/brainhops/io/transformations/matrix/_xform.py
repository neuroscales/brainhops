import numpy as np
import typing_extensions as tx
from bagof.magic import HIDE_IF_NONE

from brainhops._core.typing import ArrayLike
from brainhops.datamodel import transformations as _xforms
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import Confidence, ParserContentError
from brainhops.io.common._arrays import (
    ArrayParser,
    CsvArrayParser,
    Mat73ArrayParser,
    MatArrayParser,
    MatLegacyArrayParser,
    NpyArrayParser,
    NpzArrayParser,
    TsvArrayParser,
    TxtArrayParser,
    is_numeric_array,
)
from brainhops.io.transformations.base import (
    AffineTransformationFormat,
    FileBasedTransformation,
)

from . import _conventions

_CONVENTIONS = (
    "input",
    "output",
    "source",
    "target",
    "index_base",
    "direction",
    "vector",
    "ndim",
)


class MatrixAffine(
    AffineTransformationFormat,
    ArrayParser,
    _xforms.Affine,
    FileBasedTransformation,
    repr=HIDE_IF_NONE,
):
    """
    Affine transformation stored as a bare matrix in a generic array container.

    The file holds a (3, 3), (3, 4) or (4, 4) matrix, or a (2, 3) or (3, 3)
    matrix in 2-D, and nothing else, so the caller gives its conventions as
    keyword arguments to `load` or `from_file`. This class is abstract, and
    each container has a registered subclass that mixes in its
    [`ArrayParser`][]:

    - [`TxtMatrixAffine`][]: whitespace-separated text (`.txt`, `.dat`, `.1D`),
      hint `"matrix.txt"`;
    - [`CsvMatrixAffine`][]: `.csv`, hint `"matrix.csv"`;
    - [`TsvMatrixAffine`][]: `.tsv`, hint `"matrix.tsv"`;
    - [`NpyMatrixAffine`][]: `.npy`, hint `"matrix.npy"`;
    - [`NpzMatrixAffine`][]: `.npz`, hint `"matrix.npz"`;
    - [`MatLegacyMatrixAffine`][]: MATLAB v4 to v7 `.mat`, read with
      `scipy.io`, hint `"matrix.mat"`;
    - [`Mat73MatrixAffine`][]: MATLAB v7.3 `.mat`, read with `h5py`, hints
      `"matrix.mat"`, `"mat.73"` and `"matrix.mat.73"`.

    [`MatMatrixAffine`][] reads a MATLAB file of any version by dispatching to
    the last two. Every reader answers to `hint="matrix"`, in which case the
    container is recognized from the content. In a container that holds several
    arrays, the matrix is selected with `variable=` (or `key=`), and defaults
    to the only 2-D numeric array.

    ## Conventions

    The conventions below are applied in order: transposition, inversion, index
    shift and image placement.

    - `vector="column"`: with `"row"`, the matrix is applied as `y = x @ A` and
      is transposed.
    - `ndim=None`: only used for a (3, 3) matrix, which is 3-D linear unless
      `ndim=2` makes it a 2-D homogeneous affine.
    - `direction="forward"`: with `"inverse"`, the file stores the map from
      output to input, and the matrix is inverted.
    - `input=None`, `output=None`: the spaces of the forward matrix, as
      `"voxel"` (or `"pixel"`, `"index"`), `"ras"`, `"lps"` or a coordinate
      system. `None` stands for a generic unnamed space.
    - `index_base=0`: 1 when voxel indices are 1-based, as in MATLAB and SPM,
      or an `(input, output)` pair. A 1-based voxel endpoint is shifted to
      0-based indices, in which an integer is the centre of a voxel. A single
      value applies to every voxel endpoint and requires at least one.
    - `source=None`, `target=None`: a `nibabel` image or header, or a brainhops
      image. The voxel input or output is mapped to the world space of that
      image through its voxel-to-world affine, and passing an image makes that
      endpoint default to `"voxel"`.

    ## Dispatch

    Any small numeric table reads as a matrix, so this reader scores
    `Confidence.WEAK` and loses to every positively recognized format, such as
    FLIRT or ITK. A text matrix whose file extension matches its separator
    scores higher, so that it is not read as a FLIRT matrix. Files larger than
    1 MiB are not sniffed. `hint="matrix"`, or a container hint, selects this
    reader explicitly.
    """

    HINTS = ("matrix",)
    SNIFF_CONFIDENCE: tx.ClassVar[float] = Confidence.WEAK
    # Plain matrices are tiny, so a large file is not claimed, and a huge
    # image is never read whole only to be rejected.
    SNIFF_LIMIT: tx.ClassVar[tx.Optional[int]] = 1 << 20

    raw_matrix: tx.Optional[ArrayLike] = None
    """The matrix exactly as stored, before any convention is applied."""

    variable: tx.Optional[str] = None
    """The `.npz` key or `.mat` variable that the matrix was read from."""

    vector: tx.Optional[str] = None
    """The vector convention that the raw matrix was read with."""

    direction: tx.Optional[str] = None
    """The direction that the raw matrix was read with."""

    index_base: tx.Optional[tx.Tuple[int, int]] = None
    """The `(input, output)` voxel index base used when reading."""

    @classmethod
    def _accepts_array(cls, array: np.ndarray) -> bool:
        return is_numeric_array(array) and array.ndim == 2

    @classmethod
    def _array_confidence(cls, array: np.ndarray, **kwargs) -> float:
        """Score `Confidence.WEAK` for an array that reads as an affine."""
        vector = kwargs.get("vector", "column")
        ndim = kwargs.get("ndim", None)
        if not _conventions.is_affine_matrix(array, vector, ndim):
            return Confidence.NO
        return cls.SNIFF_CONFIDENCE

    @classmethod
    def from_array(
        cls, array: np.ndarray, key: tx.Optional[str] = None, **kwargs
    ) -> tx.Self:
        """Apply the conventions to a matrix read from a file."""
        conventions = {
            name: kwargs.pop(name) for name in _CONVENTIONS if name in kwargs
        }
        try:
            vector = _conventions.check_vector(
                conventions.get("vector", "column")
            )
            direction = _conventions.check_direction(
                conventions.get("direction", "forward")
            )
            matrix, input, output, index_base = _conventions.to_affine(
                array, **conventions
            )
        except ValueError as e:
            raise ParserContentError(str(e)) from e
        return cls(
            matrix=matrix,
            input=input,
            output=output,
            raw_matrix=array,
            variable=key,
            vector=vector,
            direction=direction,
            index_base=index_base,
            **kwargs,
        )


@register_format
class TxtMatrixAffine(TxtArrayParser, MatrixAffine):
    """
    Affine stored in a whitespace-separated text file.

    This covers `.txt`, `.dat` and AFNI `.1D` files, with one matrix row per
    line. Comments start with `#`, and blank lines are skipped. See
    [`MatrixAffine`][].
    """

    HINTS = TxtArrayParser.HINTS


@register_format
class CsvMatrixAffine(CsvArrayParser, MatrixAffine):
    """
    Affine stored in a comma-separated `.csv` file. See [`MatrixAffine`][].
    """

    HINTS = CsvArrayParser.HINTS


@register_format
class TsvMatrixAffine(TsvArrayParser, MatrixAffine):
    """Affine stored in a tab-separated `.tsv` file. See [`MatrixAffine`][]."""

    HINTS = TsvArrayParser.HINTS


@register_format
class NpyMatrixAffine(NpyArrayParser, MatrixAffine):
    """
    Affine stored in a NumPy `.npy` file, read without unpickling.

    See [`MatrixAffine`][].
    """

    HINTS = NpyArrayParser.HINTS


@register_format
class NpzMatrixAffine(NpzArrayParser, MatrixAffine):
    """
    Affine stored in a NumPy `.npz` file, read without unpickling.

    The array is selected with `variable=` or `key=`, and defaults to the only
    2-D numeric array. See [`MatrixAffine`][].
    """

    HINTS = NpzArrayParser.HINTS


class MatMatrixAffine(MatArrayParser, MatrixAffine):
    """
    Affine stored in a MATLAB `.mat` file of any version.

    The variable is selected with `variable=`, and defaults to the only 2-D
    numeric array. The file is dispatched by content to
    [`MatLegacyMatrixAffine`][] or [`Mat73MatrixAffine`][], which returns an
    object of that class. Only these variants are registered, so that this
    class does not compete with them. See [`MatrixAffine`][].
    """

    HINTS = MatArrayParser.HINTS


@register_format
class MatLegacyMatrixAffine(MatLegacyArrayParser, MatMatrixAffine):
    """
    Affine stored in a MATLAB v4 or v5 to v7 `.mat` file, read with `scipy.io`.

    See [`MatMatrixAffine`][].
    """


@register_format
class Mat73MatrixAffine(Mat73ArrayParser, MatMatrixAffine):
    """
    Affine stored in a MATLAB v7.3 (HDF5) `.mat` file, read with `h5py`.

    MATLAB v7.3 stores arrays transposed, which the reader undoes. See
    [`MatMatrixAffine`][].
    """

    HINTS = Mat73ArrayParser.HINTS


MatMatrixAffine.VARIANTS = (MatLegacyMatrixAffine, Mat73MatrixAffine)
