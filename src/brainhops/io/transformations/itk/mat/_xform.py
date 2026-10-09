import typing_extensions as tx

from brainhops.io.base._base import register_format
from brainhops.io.transformations.base import WritableFileBasedTransformation

from .._xform import ItkTransform
from ._parser import MatTransformParser


@register_format
class MatTransform(
    MatTransformParser,
    ItkTransform,
    WritableFileBasedTransformation,
):
    """Transformation stored in an ITK binary MATLAB (`.mat`) file.

    ANTs writes every linear transform in this format. FSL FLIRT also writes
    `.mat` files, but as plain text, so the two formats are told apart by their
    content.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".mat",)
    HINTS = ("mat",)
