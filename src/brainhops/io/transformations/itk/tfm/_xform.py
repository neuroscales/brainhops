# dependencies
import typing_extensions as tx

# io
from brainhops.io.base._base import register_format

# locals
from .._xform import ITKTransform
from ._parser import TFMTransformParser


@register_format
class TFMTransform(
    TFMTransformParser,
    ITKTransform,
):
    """A transformation stored in an ITK text (`.tfm`) file."""

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".tfm",)
    HINTS = ("tfm",)
