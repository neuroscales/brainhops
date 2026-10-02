"""
MRtrix linear transforms: the 3x4 / 4x4 text files of `mrtransform
-linear` and `mrregister -rigid` / `-affine`.

What they hold, and the MRtrix3 sources it was checked against, is
described in the package docstring,
[`brainhops.io.transformations.mrtrix`][].
"""

__all__ = ["MrtrixLinearTransform"]

# stdlib
from collections import OrderedDict

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import Factory

# datamodel
from brainhops.datamodel import systems as _systems
from brainhops.datamodel import transformations as _xforms

# io
from brainhops.io.base._base import register_format
from brainhops.io.base.arrays import TxtArrayParser, _as_lines, _reject
from brainhops.io.base.parsers import (
    Confidence,
    TextFileParserWriter,
    UnrepresentableTransformationError,
)
from brainhops.io.transformations.base import (
    AffineTransformationFormat,
    WritableFileBasedTransformation,
)

from ._formats import MrtrixTransformationFormat

_NDIM = 3
_HISTORY = "command_history"
_CENTRE = "centre"
_VERSION = "(version="
"""What MRtrix appends to every `command_history` entry (`core/app.cpp`)."""


def _comment_keyval(lines: tx.Iterable[str]) -> tx.Dict[str, str]:
    """
    The `# key: value` comments of an MRtrix text file.

    MRtrix writes its key-values as comments, one line per entry
    (`File::KeyValue::write` with the prefix `"# "`). The lines of a
    repeated key are joined by newlines, as in an image header. Other
    comments are dropped.
    """
    keyval: tx.Dict[str, str] = OrderedDict()
    for line in lines:
        if "#" not in line:
            continue
        comment = line.split("#", 1)[1].strip()
        key, sep, value = comment.partition(":")
        key = key.strip()
        if not sep or not key or " " in key:
            continue
        value = value.strip()
        if key in keyval:
            keyval[key] += "\n" + value
        else:
            keyval[key] = value
    return keyval


def _mrtrix_signature(keyval: tx.Mapping[str, str]) -> float:
    """How surely the comments of a text matrix were written by MRtrix."""
    history = keyval.get(_HISTORY, "")
    if _VERSION in history:
        return Confidence.CERTAIN
    if _HISTORY in keyval or _CENTRE in keyval:
        return Confidence.LIKELY
    return Confidence.NO


@register_format
class MrtrixLinearTransform(
    MrtrixTransformationFormat,
    AffineTransformationFormat,
    TxtArrayParser,
    TextFileParserWriter,
    _xforms.Affine,
    WritableFileBasedTransformation,
):
    """
    An affine stored in an MRtrix linear transform file (a 3x4 or 4x4
    text matrix).

    The matrix maps scanner RAS points of the *template* image to scanner
    RAS points of the *moving* image, in millimetres: the "reverse"
    convention `mrtransform -linear` and `mrregister` use. It is read as
    it is, as an affine from `RASmm` to `RASmm`, since that is also the
    direction of the data model (template points to the points to
    sample).

    The `# key: value` comments MRtrix writes (`command_history`,
    `centre`) are kept and written back.

    Plain text matrices look all alike, so a file is claimed only when
    its comments say MRtrix wrote it (a `command_history` entry, or a
    `centre`). Otherwise, ask for it: `hint="mrtrix.linear"`.
    """

    HINTS = ("linear",)
    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".txt",)
    # MRtrix splits values on spaces, commas, semicolons and tabs
    # (`load_matrix_2D_vector`, `core/math/math.h`).
    SEPARATORS: tx.ClassVar[tx.Optional[str]] = r"[\s,;]+"
    SNIFF_LIMIT: tx.ClassVar[tx.Optional[int]] = 1 << 20

    _input: _systems.CoordinateSystem = _systems.RASmm()
    _output: _systems.CoordinateSystem = _systems.RASmm()

    _keyval: tx.Dict[str, str] = Factory(OrderedDict, repr=False)
    """The `# key: value` comments of the file, written back."""

    @classmethod
    def from_instance(cls, other: tx.Any, *args, **kwargs) -> tx.Self:
        """Copy another affine, which must map scanner RAS to scanner RAS:
        an affine between other spaces (or unnamed ones) is not given a
        meaning it did not have."""
        src = getattr(other, "input", None)
        dst = getattr(other, "output", None)
        if not (
            isinstance(src, _systems.RASmm) and isinstance(dst, _systems.RASmm)
        ):
            raise TypeError(
                "An MRtrix linear transform maps scanner RAS to scanner RAS "
                "world coordinates; set the input and output of this affine "
                "to RASmm to say that it does."
            )
        return super().from_instance(other, *args, **kwargs)

    # --- ArrayParser hooks ----------------------------------------------

    @classmethod
    def _accepts_array(cls, array: np.ndarray) -> bool:
        """MRtrix reads 3 or 4 rows of 4 columns (`load_transform`)."""
        return array.ndim == 2 and array.shape in ((3, 4), (4, 4))

    @classmethod
    def from_array(
        cls, array: np.ndarray, key: tx.Optional[str] = None, **kwargs
    ) -> tx.Self:
        """The affine of the first three rows, as MRtrix reads it: a
        fourth row is not read."""
        matrix = np.asarray(array, dtype=np.float64)[:_NDIM]
        return cls(matrix=matrix, **kwargs)

    # --- sniff / read ---------------------------------------------------

    @classmethod
    def sniff_lines(
        cls,
        lines: tx.Iterable[str],
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score text as an MRtrix transform: a 3x4 or 4x4 matrix whose
        comments carry MRtrix's signature."""
        kwargs.pop("_named", None)
        lines = _as_lines(lines)
        score = super().sniff_lines(lines, error=error, _named=True, **kwargs)
        if not score:
            return score
        score = _mrtrix_signature(_comment_keyval(lines))
        if not score:
            return _reject(
                error,
                "A text matrix without the comments MRtrix writes; read it "
                "with hint='mrtrix.linear' if it is an MRtrix transform.",
            )
        return score

    @classmethod
    def from_lines(cls, lines: tx.Iterable[str], **kwargs) -> tx.Self:
        lines = _as_lines(lines)
        kwargs.setdefault("keyval", _comment_keyval(lines))
        return super().from_lines(lines, **kwargs)

    # --- write ----------------------------------------------------------

    def to_lines(self, **kwargs) -> tx.Iterator[str]:
        """
        The lines MRtrix writes for this transform (`save_transform`):
        the `# key: value` comments, three rows of four values, and
        `0 0 0 1`.

        Raises
        ------
        UnrepresentableTransformationError
            If the affine does not map scanner RAS to scanner RAS, or is
            not 3-D.
        """
        if kwargs:
            raise TypeError(
                f"Unknown MRtrix writer option(s): {', '.join(kwargs)}"
            )
        src, dst = self.input, self.output
        if not (
            isinstance(src, _systems.RASmm) and isinstance(dst, _systems.RASmm)
        ):
            raise UnrepresentableTransformationError(
                "An MRtrix linear transform maps scanner RAS to scanner RAS "
                "world coordinates; set the input and output of this affine "
                "to RASmm to say that it does."
            )
        matrix = self.homogeneous_matrix
        matrix = np.eye(4) if matrix is None else np.asarray(matrix, float)
        if matrix.shape != (_NDIM + 1, _NDIM + 1):
            raise UnrepresentableTransformationError(
                f"An MRtrix linear transform is 3-D, and this affine has "
                f"shape {matrix.shape[0] - 1}x{matrix.shape[1] - 1}."
            )
        for key, value in self._keyval.items():
            for line in str(value).split("\n"):
                yield f"# {key}: {line}"
        for row in matrix[:_NDIM]:
            yield " ".join(repr(float(v)) for v in row)
        yield "0 0 0 1"

    def to_text(self, **kwargs) -> str:
        """The lines of `to_lines`, ending with a newline as MRtrix's."""
        return super().to_text(**kwargs) + "\n"
