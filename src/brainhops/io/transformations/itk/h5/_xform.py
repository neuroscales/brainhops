# dependencies
import typing_extensions as tx

# io
from brainhops.io.base._base import register_format

# locals
from .._xform import ItkTransform
from ._parser import H5TransformParser


@register_format
class H5Transform(
    H5TransformParser,
    ItkTransform,
):
    """A transformation stored in an ITK binary (`.h5`) file."""

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".h5", ".hdf5")
    HINTS = ("h5",)
