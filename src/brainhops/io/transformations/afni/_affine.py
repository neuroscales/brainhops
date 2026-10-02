"""AFNI affine matrices, stored as text (`.aff12.1D`, `.1D`)."""

__all__ = ["AfniAffine"]

# stdlib
import os

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import HIDE_IF_NONE, Alias, KwOnly

# internals
from brainhops._core import path
from brainhops._core.peek import peekable_lines
from brainhops._core.typing import ArrayLike
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms
from brainhops.io.base._base import register_format
from brainhops.io.base._geometry import ras_conversion
from brainhops.io.base.afni import DICOM_TO_RAS
from brainhops.io.base.arrays import (
    ArrayContainerError,
    TxtArrayParser,
    is_numeric_array,
    read_text_array,
)
from brainhops.io.base.parsers import (
    Confidence,
    ParserContentError,
    SnifferContentError,
    TextFileParserWriter,
    WriterError,
)
from brainhops.io.transformations.base import WritableFileBasedTransformation

from ._formats import AfniAffineFormat
from ._utils import cardinal_to_real_matrix

_SIGNATURE = "DICOM-to-DICOM"
"""The words of the comment `3dAllineate` and `3dvolreg` write before
their matrices: `# 3dAllineate matrices (DICOM-to-DICOM, row-by-row):`."""

_COMMENT = "# AFNI matrices (DICOM-to-DICOM, row-by-row):"
"""The comment written before matrices that were not read from a file."""

_AFF12 = ".aff12.1D"
"""The file name ending of `-1Dmatrix_save` (`3dAllineate`, `3dvolreg`)."""


def _matrices(array: np.ndarray) -> tx.Optional[np.ndarray]:
    """
    The `(n, 3, 4)` matrices a numeric table holds, or `None`.

    AFNI reads (`3dAllineate -1Dmatrix_apply`, `cat_matvec`):

    - rows of 12 numbers, one matrix per row, its three rows one after
      the other (`-1Dmatrix_save`, `cat_matvec -ONELINE`);
    - a single matrix as 3 lines of 4 numbers (`cat_matvec`), or 4 lines
      ending in `0 0 0 1` (`cat_matvec -4x4`);
    - 9 numbers, as 3 lines of 3 or one line: a matrix with no shift.
    """
    if not is_numeric_array(array) or array.ndim != 2 or not array.size:
        return None
    array = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        return None
    rows, cols = array.shape
    if cols == 12:
        return array.reshape(rows, 3, 4)
    if (rows, cols) == (3, 4):
        return array[None]
    if (rows, cols) == (4, 4) and np.allclose(array[3], [0, 0, 0, 1]):
        return array[None, :3]
    if (rows, cols) in ((3, 3), (1, 9)):
        matrix = np.zeros((1, 3, 4))
        matrix[0, :, :3] = array.reshape(3, 3)
        return matrix
    return None


def _homogeneous(matrix: np.ndarray) -> np.ndarray:
    """A `(3, 4)` matrix, made `(4, 4)`."""
    out = np.eye(4)
    out[:3] = np.asarray(matrix, dtype=np.float64)[:3]
    return out


def _comment(lines: tx.Sequence[str]) -> tx.Optional[str]:
    """The comment lines before the first value, or `None`."""
    comments = []
    for line in lines:
        text = line.strip()
        if not text:
            continue
        if not text.startswith("#"):
            break
        comments.append(text)
    return "\n".join(comments) or None


def _as_lines(lines: tx.Iterable[str]) -> tx.List[str]:
    if isinstance(lines, peekable_lines):
        lines = list(lines)
    return [line for line in lines if isinstance(line, str)]


def _format(value: float) -> str:
    """A value, with every digit of a double: `repr` of the float, made
    more compact for integers (`1` rather than `1.0`)."""
    value = float(value)
    if value == 0:
        return "0"
    if value.is_integer() and abs(value) < 1e15:
        return str(int(value))
    return repr(value)


def _pop_alias(kwargs: tx.Dict[str, tx.Any], *names: str) -> tx.Any:
    """Pop every name from `kwargs`, and return the first one set."""
    found = None
    for name in names:
        value = kwargs.pop(name, None)
        if found is None:
            found = value
    return found


@register_format
class AfniAffine(
    AfniAffineFormat,
    TxtArrayParser,
    TextFileParserWriter,
    _xforms.Affine,
    WritableFileBasedTransformation,
    repr=HIDE_IF_NONE,
):
    """
    An affine transformation stored in an AFNI text matrix file
    (`.aff12.1D`, `.1D`).

    The file holds the 12 numbers of the `(3, 4)` matrix `M` that maps
    the base dataset's DICOM (LPS) coordinates to the source dataset's,
    `x_source = M @ x_base`: one matrix per row, as written by
    `3dAllineate -1Dmatrix_save`, `3dvolreg -1Dmatrix_save`,
    `align_epi_anat.py` and `cat_matvec -ONELINE`, or a single matrix
    as 3 lines of 4 numbers (`cat_matvec`). This is the direction the data
    model resamples with (base to source, "pull"), so `matrix` is `M`
    itself, from `LPSmm` to `LPSmm`.

    **Several matrices.** A file may hold one matrix per volume of the
    source (`3dvolreg`, or `3dAllineate` on a time series). They are all
    kept, raw, in `afni_matrices`, and returned in world coordinates by
    `matrices`; this affine is the one of volume `volume` (the first one
    by default). The data model has no affine that changes along the
    time axis, so a file of several matrices is read one volume at a
    time: `load(path, volume=12)`. It is written back whole.

    **Obliquity.** AFNI computes on each dataset's cardinal grid and
    ignores its obliquity, so `M` maps cardinal coordinates. When the
    `base` and `source` images are given (`base=`/`reference=`,
    `source=`/`moving=`), `matrix` maps their true coordinates instead:
    `matrix = c2r(source) @ M @ inv(c2r(base))`, where `c2r` is
    [`cardinal_to_real`][brainhops.io.transformations.afni.cardinal_to_real].
    Without them, the two are taken to be the same, as they are for
    datasets that are not oblique (templates, most `3dQwarp` bases).

    **Writing.** A matrix read from a file, and left untouched, is
    written back as it was read: every row, and the comment above them.
    Otherwise `matrix` is converted from its endpoints to DICOM (an
    `RASmm` endpoint is flipped; an endpoint without orientation is taken
    to be RAS), and to cardinal coordinates when `base` and `source` are
    known (as fields, or as writer options), and written as one row of
    12 numbers. The writer option `oneline=False` writes a single matrix
    as 3 lines of 4 numbers instead.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (_AFF12, ".1D")
    HINTS = ("aff12",)

    # Volume registration of a long time series writes one row per
    # volume: allow a few megabytes before turning a file down unread.
    SNIFF_LIMIT: tx.ClassVar[tx.Optional[int]] = 16 << 20

    _input: KwOnly[_systems.CoordinateSystem] = _systems.LPSmm()
    _output: KwOnly[_systems.CoordinateSystem] = _systems.LPSmm()

    afni_matrices: tx.Optional[ArrayLike] = None
    """The `(n, 3, 4)` matrices exactly as stored in the file, from the
    base's to the source's cardinal DICOM coordinates, one per row."""

    volume: int = 0
    """The row of `afni_matrices` that this affine is (negative values
    count from the end)."""

    comment: tx.Optional[str] = None
    """The comment lines above the matrices, written back unchanged."""

    base: tx.Annotated[
        tx.Optional[tx.Any], Alias(("base", "reference", "ref"))
    ] = None
    """The base (reference) image: an AFNI image or header, a NIfTI image
    or header, or a brainhops image. Used for its obliquity only."""

    source: tx.Annotated[
        tx.Optional[tx.Any], Alias(("source", "moving", "mov", "src"))
    ] = None
    """The source (moving) image. Used for its obliquity only."""

    # --- matrices -----------------------------------------------------

    def _stack(self) -> tx.Optional[np.ndarray]:
        matrices = getattr(self, "afni_matrices", None)
        if matrices is None:
            return None
        matrices = np.asarray(matrices, dtype=np.float64)
        return matrices.reshape(-1, 3, 4)

    def _world(self, raw: np.ndarray) -> np.ndarray:
        """A stored `(3, 4)` matrix, in true coordinates if the images
        are known: `c2r(source) @ raw @ inv(c2r(base))`."""
        matrix = _homogeneous(raw)
        if self.base is not None:
            matrix = matrix @ np.linalg.inv(cardinal_to_real_matrix(self.base))
        if self.source is not None:
            matrix = cardinal_to_real_matrix(self.source) @ matrix
        return matrix[:3]

    @property
    def nvolumes(self) -> int:
        """The number of matrices in the file (0 if not read from one)."""
        stack = self._stack()
        return 0 if stack is None else len(stack)

    @property
    def matrices(self) -> tx.Optional[np.ndarray]:
        """Every matrix of the file, as a `(n, 3, 4)` array in the
        coordinates `matrix` is in (true ones when `base` and `source`
        are known)."""
        stack = self._stack()
        if stack is None:
            return None
        return np.stack([self._world(raw) for raw in stack])

    @property
    def matrix(self) -> tx.Optional[np.ndarray]:
        """The `(3, 4)` base-to-source matrix of volume `volume`, in LPS
        millimetres, unless set explicitly."""
        if getattr(self, "_matrix", None) is not None:
            return self._matrix
        stack = self._stack()
        if stack is None:
            return None
        try:
            raw = stack[int(self.volume)]
        except IndexError:
            raise IndexError(
                f"Volume {self.volume} is out of range: the file holds "
                f"{len(stack)} matrices."
            ) from None
        return self._world(raw)

    @matrix.setter
    def matrix(self, value: tx.Optional[np.ndarray]) -> None:
        self._matrix = value

    # --- sniff --------------------------------------------------------

    @classmethod
    def _accepts_array(cls, array: np.ndarray) -> bool:
        return _matrices(array) is not None

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        name = getattr(file, "name", None)
        if isinstance(name, (str, os.PathLike)):
            kwargs["_aff12"] = os.fspath(name).endswith(_AFF12)
        return super().sniff_fileobj(file, error=error, **kwargs)

    @classmethod
    def sniff_lines(
        cls,
        lines: tx.Iterable[str],
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """
        Score text as AFNI matrices.

        The table must hold matrices AFNI reads (see the class
        documentation). Then the comment `3dAllineate` and `3dvolreg`
        write (`... (DICOM-to-DICOM, row-by-row):`), or a `.aff12.1D`
        name, make it `CERTAIN`. A `.1D` name makes rows of 12 numbers,
        or 3 lines of 4, `LIKELY`, and the other layouts (a `(4, 4)` or
        `(3, 3)` matrix, which any software may save) `WEAK`, so that
        the generic matrix reader keeps them. Rows of 12 numbers in a
        file named otherwise are `WEAK`.
        """
        named = kwargs.pop("_named", False)
        aff12 = kwargs.pop("_aff12", False)
        lines = _as_lines(lines)
        try:
            array = read_text_array(lines, separators=cls.SEPARATORS)
        except ArrayContainerError as e:
            return cls._reject(error, str(e))
        matrices = _matrices(array)
        if matrices is None:
            return cls._reject(
                error, f"A {array.shape} table is not an AFNI matrix file."
            )
        comment = _comment(lines) or ""
        if aff12 or _SIGNATURE in comment:
            return Confidence.CERTAIN
        # Rows of 12 numbers, and 3 lines of 4, are AFNI's own layouts.
        # A square matrix is what any software saves, so a `.1D` file
        # that holds one is left to the generic matrix reader.
        own = array.shape[1] == 12 or array.shape == (3, 4)
        if named:
            return Confidence.LIKELY if own else Confidence.WEAK
        if array.shape[1] == 12:
            return Confidence.WEAK
        return cls._reject(
            error, "A matrix in a file not named .1D is not claimed by AFNI."
        )

    @staticmethod
    def _reject(
        error: tx.Union[bool, tx.Type[Exception]], message: str
    ) -> float:
        if error:
            if error is True:
                error = SnifferContentError
            raise error(message)
        return Confidence.NO

    # --- from ---------------------------------------------------------

    @classmethod
    def from_lines(cls, lines: tx.Iterable[str], **kwargs) -> tx.Self:
        """Read the matrices, and the comment above them."""
        lines = _as_lines(lines)
        kwargs.setdefault("comment", _comment(lines))
        return super().from_lines(lines, **kwargs)

    @classmethod
    def from_array(
        cls, array: np.ndarray, key: tx.Optional[str] = None, **kwargs
    ) -> tx.Self:
        """Build the affine from the table read from the file."""
        matrices = _matrices(array)
        if matrices is None:
            raise ParserContentError(
                f"A {np.shape(array)} table is not an AFNI matrix file: "
                f"expected rows of 12 numbers, or 3 lines of 4."
            )
        base = _pop_alias(kwargs, "base", "reference", "ref")
        source = _pop_alias(kwargs, "source", "moving", "mov", "src")
        obj = cls(afni_matrices=matrices, base=base, source=source, **kwargs)
        if not -len(matrices) <= int(obj.volume) < len(matrices):
            raise ParserContentError(
                f"Volume {obj.volume} is out of range: the file holds "
                f"{len(matrices)} matrices."
            )
        return obj

    # --- to -----------------------------------------------------------

    def _is_as_read(self) -> bool:
        """Whether the matrices are those read from the file: none was
        set explicitly, and both endpoints are still in LPS."""
        return (
            self._stack() is not None
            and getattr(self, "_matrix", None) is None
            and np.array_equal(ras_conversion(self.input), DICOM_TO_RAS)
            and np.array_equal(ras_conversion(self.output), DICOM_TO_RAS)
        )

    def _rows(self, base: tx.Any = None, source: tx.Any = None) -> np.ndarray:
        """The `(n, 3, 4)` matrices to write, in cardinal DICOM."""
        if self._is_as_read():
            return self._stack()
        matrix = getattr(self, "matrix", None)
        if matrix is None:
            raise WriterError(
                "This affine has no matrix, so there is nothing to write."
            )
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != (3, 4):
            raise WriterError(
                f"An AFNI matrix maps 3-D coordinates, and this one has "
                f"shape {matrix.shape}."
            )
        to_ras_in = ras_conversion(self.input)
        to_ras_out = ras_conversion(self.output)
        lps = (
            DICOM_TO_RAS
            @ to_ras_out
            @ _homogeneous(matrix)
            @ np.linalg.inv(to_ras_in)
            @ DICOM_TO_RAS
        )
        base = self.base if base is None else base
        source = self.source if source is None else source
        if base is not None:
            lps = lps @ cardinal_to_real_matrix(base)
        if source is not None:
            lps = np.linalg.inv(cardinal_to_real_matrix(source)) @ lps
        return lps[None, :3]

    def to_lines(
        self,
        oneline: bool = True,
        base: tx.Any = None,
        source: tx.Any = None,
        **kwargs,
    ) -> tx.Iterator[str]:
        """
        The lines of the AFNI matrix file.

        Parameters
        ----------
        oneline : bool
            Write each matrix as one row of 12 numbers (`-1Dmatrix_save`),
            after a comment line. `False` writes a single matrix as 3
            lines of 4 numbers (`cat_matvec`'s default layout).
        base, source : image, optional
            The base and source images, whose obliquity is undone when a
            matrix in true coordinates is written. Default: the fields of
            the same names.
        """
        base = _pop_alias({"base": base, **kwargs}, "base", "reference", "ref")
        source = _pop_alias(
            {"source": source, **kwargs}, "source", "moving", "mov", "src"
        )
        # Everything is computed before the first line is handed out, so
        # that a matrix that cannot be written is refused before the file
        # is opened.
        rows = self._rows(base, source)
        comment = self.comment if self._is_as_read() else None
        lines = []
        if not oneline:
            if len(rows) != 1:
                raise WriterError(
                    f"Only a single matrix can be written as 3 lines of 4 "
                    f"numbers, and there are {len(rows)}: use oneline=True."
                )
            lines += comment.splitlines() if comment else []
            lines += [" ".join(_format(v) for v in row) for row in rows[0]]
            return iter(lines)
        lines += (comment or _COMMENT).splitlines()
        lines += [
            " " + " ".join(_format(v) for v in matrix.ravel())
            for matrix in rows
        ]
        return iter(lines)

    def to_text(self, **kwargs) -> str:
        """The content of the AFNI matrix file."""
        return "\n".join(self.to_lines(**kwargs)) + "\n"

    def to_filename(self, filename: path.FilenameLike, **kwargs) -> None:
        """Write the matrix file. Its content is built first, so a matrix
        that cannot be written leaves no file behind."""
        text = self.to_text(**kwargs)
        if isinstance(filename, str):
            filename = path.Path(filename)
        with filename.open("w") as f:
            f.write(text)
