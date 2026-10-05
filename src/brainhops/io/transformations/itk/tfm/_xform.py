# dependencies
import typing_extensions as tx

# io
from brainhops.io.base._base import register_format

# locals
from .._xform import ItkTransform
from ._parser import TfmTransformParser


@register_format
class TfmTransform(
    TfmTransformParser,
    ItkTransform,
):
    """A transformation stored in an ITK text (`.tfm`) file."""

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".tfm",)
    HINTS = ("tfm",)
