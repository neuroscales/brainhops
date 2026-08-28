"""
Generic representation of transforms and their interactions.

Mostly based on OME-NGFF, but with more flexibility.
Should be able to accommodate a large variety of existing transform formats:

- OME-NGFF
- FreeSurfer LTA (affine)
- Freesurfer XFM (nonlinear)
- ANTs
- SPM
- nitorch
- ...

(FSL defines its transformations with respect to the fixed and moving
images, but their metadata is not stored in the transform, which means
that the fixed and moving images must be accessible when applying the
transform on some third image. This is very inconvenient and not a use
case I am fond of supporting).

"""
__all__ = [
    "axes",
    "base",
    "enums",
    "hierarchy",
    "images",
    "orientation",
    "systems",
    "transformations",
    "typing",
    "units",
]

# trigger registration
from . import _xform_adaptors as _
from . import _xform_composers as _  # noqa: F811
from . import _xform_converters as _  # noqa: F401, F811
from . import (
    axes,
    base,
    enums,
    hierarchy,
    images,
    orientation,
    systems,
    transformations,
    units,
)
