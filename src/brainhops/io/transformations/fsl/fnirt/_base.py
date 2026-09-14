# dependencies
import typing_extensions as tx

# core
from brainhops._core import path

# io
from brainhops.io.transformations.nifti.base import NiftiBasedTransformation

# FNIRT NIfTI intent codes (see $FSLDIR/src/fnirt/fnirt_file_writer.cpp).
FSL_FNIRT_DISPLACEMENT_FIELD = 2006
FSL_CUBIC_SPLINE_COEFFICIENTS = 2007
FSL_DCT_COEFFICIENTS = 2008
FSL_QUADRATIC_SPLINE_COEFFICIENTS = 2009

# A FNIRT reader claims an FSL-intent NIfTI over the generic RAS
# coordinate field reader, which also scores such files as certain. The
# two are the same container and score identically, so an explicit
# priority is what separates them.
_FNIRT_PRIORITY = 10


class FNIRTTransformation(NiftiBasedTransformation):
    """Base for FNIRT non-linear transformations stored in a NIfTI file.

    A FNIRT transformation maps reference-image coordinates to
    moving-image coordinates, both in FSL scaled-mm coordinates. Turning
    it into a world-space transformation needs the moving image, and the
    reference image. The reference defaults to the geometry of the FNIRT
    file itself, because the field is defined on the reference grid. The
    moving image must be supplied.

    Abstract: it is not decorated with `@register_format`, so it never
    takes part in dispatch. Concrete FNIRT transformations inherit from
    it and register themselves.
    """

    moving: tx.Optional[tx.Any] = None
    """The moving (source) image, a nibabel image or header."""

    reference: tx.Optional[tx.Any] = None
    """The reference image. Defaults to the FNIRT file's own geometry."""

    PRIORITY: tx.ClassVar[int] = _FNIRT_PRIORITY

    # --- image keyword handling ---------------------------------------
    #
    # `NiftiParser.from_file` forwards its keyword arguments to
    # `nibabel.load`, which rejects `moving=`/`reference=`. These
    # overrides peel the image keywords off before delegating, then set
    # them on the parsed object.

    @classmethod
    def _pop_images(cls, kwargs: dict) -> tx.Tuple[tx.Any, tx.Any]:
        moving = kwargs.pop("moving", None)
        if moving is None:
            moving = kwargs.pop("src", None)
        reference = kwargs.pop("reference", None)
        if reference is None:
            reference = kwargs.pop("ref", None)
        return moving, reference

    @classmethod
    def _with_images(
        cls, obj: tx.Self, moving: tx.Any, reference: tx.Any
    ) -> tx.Self:
        if moving is not None:
            obj.moving = moving
        if reference is not None:
            obj.reference = reference
        return obj

    @classmethod
    def from_file(cls, file: path.FileLike, **kwargs) -> tx.Self:
        moving, reference = cls._pop_images(kwargs)
        obj = super().from_file(file, **kwargs)
        return cls._with_images(obj, moving, reference)

    @classmethod
    def from_fileobj(cls, fileobj: tx.BinaryIO, **kwargs) -> tx.Self:
        moving, reference = cls._pop_images(kwargs)
        obj = super().from_fileobj(fileobj, **kwargs)
        return cls._with_images(obj, moving, reference)

    @classmethod
    def from_bytes(cls, data: bytes, **kwargs) -> tx.Self:
        moving, reference = cls._pop_images(kwargs)
        obj = super().from_bytes(data, **kwargs)
        return cls._with_images(obj, moving, reference)
