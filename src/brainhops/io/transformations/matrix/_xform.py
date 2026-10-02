# dependencies
import typing_extensions as tx

# datamodel
from brainhops.datamodel import transformations as _xforms

# io
from brainhops.io.base._base import register_format
from brainhops.io.transformations.base import (
    AffineTransformationFormat,
    FileBasedTransformation,
)

from ._parser import MatrixParser


@register_format
class MatrixAffine(
    AffineTransformationFormat,
    MatrixParser,
    _xforms.Affine,
    FileBasedTransformation,
):
    """
    An affine stored as a bare matrix in a text, NumPy or MATLAB file.

    The file holds a `(3, 3)`, `(3, 4)` or `(4, 4)` matrix (in 2-D,
    `(2, 3)` or `(3, 3)`), and nothing else: no coordinate systems, no
    index base, no direction. The caller states those as keyword
    arguments to `load` / `from_file`; the defaults are listed below.

    **Containers**

    - **text** (`.txt`, `.csv`, `.tsv`, `.dat`, ...): one row per line,
      values separated by whitespace, tabs, commas or semicolons, `#`
      starting a comment. Blank lines are skipped.
    - **NumPy** `.npy`, or `.npz` (select the array with `variable=`; by
      default the only 2-D numeric array). Read with
      `allow_pickle=False`.
    - **MATLAB** `.mat`: v4 and v5-v7 are read with `scipy.io.loadmat`,
      v7.3 (HDF5) with `h5py`, imported only then. Select the variable
      with `variable=`; by default the only 2-D numeric variable. v7.3
      stores arrays transposed, which is undone.

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
    was read with are kept on the object (`raw_matrix`, `container`,
    `variable`, `vector`, `direction`, `index_base`).

    **Dispatch**

    Any small numeric table reads as a matrix, so this reader scores
    `WEAK` and loses to every format that recognizes a file positively:
    a FLIRT `.mat` (a text `(4, 4)`), an ITK `.mat` (MATLAB v4), an ITK
    `.tfm` / `.txt`. The exception is a text matrix whose file name ends
    in `.txt`, `.csv`, `.tsv` or `.dat`, which scores `LIKELY` so it is
    not read as a FLIRT matrix. Pass `hint="matrix"` to force this reader.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (
        ".txt",
        ".csv",
        ".tsv",
        ".dat",
        ".npy",
        ".npz",
        ".mat",
    )
    HINTS = ("matrix",)
