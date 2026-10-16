"""FSL FLIRT linear transforms.

A FLIRT `.mat` file holds a (4, 4) affine that maps the scaled millimetre
coordinates of the moving image to the scaled millimetre coordinates of the
reference image.

A FLIRT file is short and is read in one pass. Its matrix is held as a
[`FlirtRaw`][] record by [`FlirtMetadata`][], the metadata of the file, and
a [`FlirtTransform`][] holds that metadata. The file holds no image
geometry, so the moving and reference images that place the matrix in
world coordinates are options of the transformation, `moving=` and
`reference=`, which are never part of the record. brainhops does not
write FLIRT files.
"""

__all__ = ["FlirtMetadata", "FlirtRaw", "FlirtTransform"]

import typing_extensions as tx

from brainhops._core.lazy import lazy_exports
from brainhops.io.base._base import register_format
from brainhops.io.metadata._base import MetadataFormat
from brainhops.io.transformations.fsl._formats import FslAffineFormat


@register_format("._xform:FlirtTransform")
class FlirtTransformStub(FslAffineFormat):
    """Hints and file names of [`FlirtTransform`][], which derives from it.

    Dispatch reads this class to tell whether a file may be a FLIRT
    matrix before the reader of FLIRT matrices is imported.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".mat",)
    HINTS = ("flirt",)


@register_format("._metadata:FlirtMetadata")
class FlirtMetadataStub(MetadataFormat):
    """Hints and file names of [`FlirtMetadata`][], which derives from it."""

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".mat",)
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("fsl", "flirt")


__getattr__, __dir__ = lazy_exports(
    __name__,
    {
        "FlirtMetadata": "._metadata",
        "FlirtRaw": "._raw",
        "FlirtTransform": "._xform",
    },
)
