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

    ANTs writes every linear transform in this format. FLIRT `.mat` files are
    plain text, and the two are told apart by their content.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".mat",)
    HINTS = ("mat",)
