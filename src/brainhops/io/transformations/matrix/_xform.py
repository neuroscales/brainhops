# dependencies
import numpy as np
import typing_extensions as tx

# externals
from bagof.magic import HIDE_IF_NONE

# core
from brainhops._core.typing import ArrayLike

# datamodel
from brainhops.datamodel import transformations as _xforms

# io
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import Confidence, ParserContentError
from brainhops.io.common.arrays import (
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

# Keyword arguments that interpret the matrix.
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
    An affine stored as a bare matrix in a generic array container.

    The file holds a `(3, 3)`, `(3, 4)` or `(4, 4)` matrix (in 2-D,
    `(2, 3)` or `(3, 3)`), and nothing else: no coordinate systems, no
    index base, no direction. The caller states those as keyword
    arguments to `load` / `from_file`; the defaults are listed below.

    Abstract: it reads no container, and is not decorated with
    `@register_format`, so it never takes part in dispatch. Each
    container has its own registered subclass, which mixes in that
    container's [`ArrayParser`][brainhops.io.base.arrays.ArrayParser]:

    - [`TxtMatrixAffine`][]: whitespace-separated text (`.txt`, `.dat`,
      `.1D`); hint `"matrix.txt"`.
    - [`CsvMatrixAffine`][]: comma-separated text (`.csv`); hint
      `"matrix.csv"`.
    - [`TsvMatrixAffine`][]: tab-separated text (`.tsv`); hint
      `"matrix.tsv"`.
    - [`NpyMatrixAffine`][]: NumPy `.npy`; hint `"matrix.npy"`.
    - [`NpzMatrixAffine`][]: NumPy `.npz`; hint `"matrix.npz"`.
    - [`MatLegacyMatrixAffine`][]: MATLAB v4 and v5-v7 `.mat`, read
      with `scipy.io`; hint `"matrix.mat"`.
    - [`Mat73MatrixAffine`][]: MATLAB v7.3 (HDF5) `.mat`, read with
      `h5py`; hints `"matrix.mat"`, `"mat.73"`, `"matrix.mat.73"`.

    [`MatMatrixAffine`][] (hint `"matrix.mat"`) is their unregistered
    parent: it reads any MATLAB version by dispatching to them.

    All of them answer to `hint="matrix"`, which then picks the
    container from the content. In a container that holds several
    arrays (`.npz`, `.mat`), select the matrix with `variable=` (or
    `key=`); by default, the only 2-D numeric array is read.

    **Conventions** (keyword arguments)

    - `vector="column"`: `"column"` if the matrix maps column vectors
      (`y = A @ x`), `"row"` if it maps row vectors (`y = x @ A`, the
      matrix is then transposed).
    - `ndim=None`: the number of spatial dimensions. Only needed for a
      `(3, 3)` matrix, which is read as a 3-D linear map unless
      `ndim=2` makes it a 2-D homogeneous affine.
    - `direction="forward"`: `"inverse"` if the file stores the map from
      `output` to `input`; it is then inverted.
    - `input=None`, `output=None`: the spaces the (forward) matrix maps
      from and to: `"voxel"` (or `"pixel"` / `"index"`), `"ras"`,
      `"lps"`, a `CoordinateSystem`, or `None` for a generic, unnamed
      space.
    - `index_base=0`: `1` if voxel indices are 1-based (MATLAB, SPM),
      or an `(input, output)` pair. A 1-based voxel endpoint is shifted
      to the 0-based, voxel-centred index space of the data model. A
      scalar applies to every voxel endpoint and requires at least one.
    - `source=None`, `target=None`: images (nibabel image or header, or
      brainhops image). When given, the voxel `input` (`output`) is
      mapped to that image's world space through its voxel-to-world
      affine, so the result is a world-to-world affine. Passing an image
      makes that endpoint default to `"voxel"`.

    The conventions are applied in this order: transposition, inversion,
    index shift, image placement. The raw matrix and the conventions it
    was read with are kept on the object (`raw_matrix`, `variable`,
    `vector`, `direction`, `index_base`); the container is the class's
    `CONTAINER`.

    **Dispatch**

    Any small numeric table reads as a matrix, so these readers score
    `WEAK` and lose to every format that recognizes a file positively:
    a FLIRT `.mat` (a text `(4, 4)`), an ITK `.mat` (MATLAB v4), an ITK
    `.tfm` / `.txt` / `.h5`. The exception is a text matrix whose file
    name has a text-array extension (`.txt`, `.csv`, `.tsv`, `.dat`,
    `.1D`) matching its separator, which scores `LIKELY` so it is not
    read as a FLIRT matrix.
    Files over 1 MiB are not sniffed. Pass `hint="matrix"` (or a
    container hint) to force these readers.
    """

    HINTS = ("matrix",)
    SNIFF_CONFIDENCE: tx.ClassVar[float] = Confidence.WEAK
    # Plain matrices are tiny: a larger file is not claimed, so a
    # multi-gigabyte image is never read whole just to be turned down.
    SNIFF_LIMIT: tx.ClassVar[tx.Optional[int]] = 1 << 20

    raw_matrix: tx.Optional[ArrayLike] = None
    """The matrix exactly as stored in the file, before any convention
    (transposition, inversion, index shift, image placement) is applied."""

    variable: tx.Optional[str] = None
    """The `.npz` key or `.mat` variable the matrix was read from."""

    vector: tx.Optional[str] = None
    """The vector convention the raw matrix was read with."""

    direction: tx.Optional[str] = None
    """The direction the raw matrix was read with."""

    index_base: tx.Optional[tx.Tuple[int, int]] = None
    """The `(input, output)` voxel index base the raw matrix was read
    with."""

    # --- ArrayParser hooks ----------------------------------------------

    @classmethod
    def _accepts_array(cls, array: np.ndarray) -> bool:
        """A matrix is a numeric 2-D array."""
        return is_numeric_array(array) and array.ndim == 2

    @classmethod
    def _array_confidence(cls, array: np.ndarray, **kwargs) -> float:
        """`WEAK` if the array reads as an affine under the requested
        `vector` and `ndim` conventions, else `NO`."""
        vector = kwargs.get("vector", "column")
        ndim = kwargs.get("ndim", None)
        if not _conventions.is_affine_matrix(array, vector, ndim):
            return Confidence.NO
        return cls.SNIFF_CONFIDENCE

    @classmethod
    def from_array(
        cls, array: np.ndarray, key: tx.Optional[str] = None, **kwargs
    ) -> tx.Self:
        """Apply the conventions to the raw matrix read from the file."""
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
    An affine stored as a bare matrix in whitespace-separated text
    (`.txt`, also `.dat` and AFNI-style `.1D`): one row per line, `#`
    starting a comment, blank lines skipped. See [`MatrixAffine`][] for
    the conventions.
    """

    HINTS = TxtArrayParser.HINTS


@register_format
class CsvMatrixAffine(CsvArrayParser, MatrixAffine):
    """
    An affine stored as a bare matrix in comma-separated text (`.csv`).
    See [`MatrixAffine`][] for the conventions.
    """

    HINTS = CsvArrayParser.HINTS


@register_format
class TsvMatrixAffine(TsvArrayParser, MatrixAffine):
    """
    An affine stored as a bare matrix in tab-separated text (`.tsv`).
    See [`MatrixAffine`][] for the conventions.
    """

    HINTS = TsvArrayParser.HINTS


@register_format
class NpyMatrixAffine(NpyArrayParser, MatrixAffine):
    """
    An affine stored as a bare matrix in a NumPy `.npy` file, read
    without unpickling. See [`MatrixAffine`][] for the
    conventions.
    """

    HINTS = NpyArrayParser.HINTS


@register_format
class NpzMatrixAffine(NpzArrayParser, MatrixAffine):
    """
    An affine stored as a bare matrix in a NumPy `.npz` archive, read
    without unpickling. Select the array with `variable=` (or `key=`);
    by default, the only 2-D numeric array. See
    [`MatrixAffine`][] for the conventions.
    """

    HINTS = NpzArrayParser.HINTS


class MatMatrixAffine(MatArrayParser, MatrixAffine):
    """
    An affine stored as a bare matrix in a MATLAB `.mat` file of any
    version. Select the variable with `variable=`; by default, the only
    2-D numeric variable. See [`MatrixAffine`][] for the conventions.

    It dispatches to [`MatLegacyMatrixAffine`][] (v4, v5-v7) or
    [`Mat73MatrixAffine`][] (v7.3) by content, and returns an object of
    that class. It is not registered itself: its two variants are, and
    answer to its hint `"matrix.mat"`, so registering it too would only
    put it in competition with its own subclasses.
    """

    HINTS = MatArrayParser.HINTS


@register_format
class MatLegacyMatrixAffine(MatLegacyArrayParser, MatMatrixAffine):
    """
    An affine stored as a bare matrix in a MATLAB v4 or v5-v7 `.mat`
    file, read with `scipy.io`. See [`MatMatrixAffine`][].
    """


@register_format
class Mat73MatrixAffine(Mat73ArrayParser, MatMatrixAffine):
    """
    An affine stored as a bare matrix in a MATLAB v7.3 (HDF5) `.mat`
    file, read with `h5py`. v7.3 stores arrays transposed, which is
    undone. See [`MatMatrixAffine`][].
    """

    HINTS = Mat73ArrayParser.HINTS


MatMatrixAffine.VARIANTS = (MatLegacyMatrixAffine, Mat73MatrixAffine)
