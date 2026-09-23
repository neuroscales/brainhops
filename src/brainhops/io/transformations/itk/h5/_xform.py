# dependencies
import typing_extensions as tx

# io
from brainhops.io.base._base import register_format

# locals
from .._xform import ITKTransform
from ._parser import H5TransformParser


@register_format
class H5Transform(
    H5TransformParser,
    ITKTransform,
):
    """A transformation stored in an ITK binary (`.h5`) file."""

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".h5", ".hdf5")
    HINTS = ("h5",)
