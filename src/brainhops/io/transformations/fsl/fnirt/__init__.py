"""FSL FNIRT non-linear transforms.

FNIRT stores either a deformation field (`--fout`) or a spline
coefficient field (`--cout`) in a NIfTI file, and the intent code of the
file tells the two apart. A single reader handles both kinds of field,
and it writes each back as the kind that it read.
"""

__all__ = ["FnirtWarpField"]

from brainhops._core.lazy import lazy_exports
from brainhops.io.base._base import register_format
from brainhops.io.transformations.fsl._formats import FslTransformationFormat
from brainhops.io.transformations.nifti._stubs import (
    NiftiBasedTransformationStub,
)


@register_format("._base:FnirtWarpField")
class FnirtWarpFieldStub(
    FslTransformationFormat, NiftiBasedTransformationStub
):
    """Hints and file names of [`FnirtWarpField`][], which derives from it.

    Dispatch reads this class to tell whether a file may be a FNIRT field
    before the reader of FNIRT fields is imported.
    """

    HINTS = ("fnirt",)


__getattr__, __dir__ = lazy_exports(__name__, {"FnirtWarpField": "._base"})
