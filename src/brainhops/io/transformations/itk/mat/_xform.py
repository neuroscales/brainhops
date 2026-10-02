# dependencies
import typing_extensions as tx

# io
from brainhops.io.base._base import register_format
from brainhops.io.transformations.base import WritableFileBasedTransformation

# locals
from .._xform import ITKTransform
from ._parser import MatTransformParser


@register_format
class MatTransform(
    MatTransformParser,
    ITKTransform,
    WritableFileBasedTransformation,
):
    """A transformation stored in an ITK binary MATLAB (`.mat`) file.

    This is the file that ANTs writes for every linear transform:
    `<prefix>0GenericAffine.mat`, but also `Rigid.mat`, `Affine.mat`,
    `Similarity.mat`, `Translation.mat` and
    `DerivedInitialMovingTranslation.mat`.

    FSL FLIRT also writes `.mat` files, but as text. The two are told
    apart by content: this reader claims only files that start with a
    MATLAB v4 header naming an ITK transform.

    !!! note "What is written"
        `save` writes a single block, as ANTs does (see
        [`to_bytes`][brainhops.io.transformations.itk.mat.MatTransformParser.to_bytes]).
        A block read from an ITK file is written back as it was read,
        center included. An affine is written as an `AffineTransform`
        centered on the origin (`fixed` is zero), since a brainhops
        affine has no center:
        `MatTransform([affine]).save("out0GenericAffine.mat")`. ITK and
        ANTs read back the same matrix, since `y = A (x - c) + c + t`
        is `y = A x + t` when `c = 0`.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".mat",)
    HINTS = ("mat",)
