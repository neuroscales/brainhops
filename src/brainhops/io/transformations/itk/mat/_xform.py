# dependencies
import typing_extensions as tx

# io
from brainhops.io.base._base import register_format

# locals
from .._xform import ItkTransform
from ._parser import MatTransformParser


@register_format
class MatTransform(
    MatTransformParser,
    ItkTransform,
):
    """A transformation stored in an ITK binary MATLAB (`.mat`) file.

    This is the file that ANTs writes for every linear transform:
    `<prefix>0GenericAffine.mat`, but also `Rigid.mat`, `Affine.mat`,
    `Similarity.mat`, `Translation.mat` and
    `DerivedInitialMovingTranslation.mat`.

    FSL FLIRT also writes `.mat` files, but as text. The two are told
    apart by content: this reader claims only files that start with a
    MATLAB v4 header naming an ITK transform.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".mat",)
    HINTS = ("mat",)
