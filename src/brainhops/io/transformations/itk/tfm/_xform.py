import typing_extensions as tx

from brainhops.io.base._base import register_format

from .._xform import ItkTransform
from ._parser import TfmTransformReader


@register_format
class TfmTransform(
    TfmTransformReader,
    ItkTransform,
):
    """Transformation stored in an ITK text `.tfm` file."""

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".tfm",)
    HINTS = ("tfm",)
