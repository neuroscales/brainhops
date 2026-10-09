import typing_extensions as tx

from brainhops.io.base._base import register_format

from .._xform import ItkTransform
from ._parser import H5TransformParser


@register_format
class H5Transform(
    H5TransformParser,
    ItkTransform,
):
    """Transformation stored in an ITK HDF5 (`.h5`) file."""

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".h5", ".hdf5")
    HINTS = ("h5",)
