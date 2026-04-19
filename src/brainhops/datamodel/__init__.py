"""
Generic representation of transforms and their interactions.

Mostly based on OME-NGFF, but with more flexibility.
Should be able to accomodate a large variety of existing transform formats:
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
__all__ = ["axes", "images", "orientation", "systems", "transforms", "typing"]

from . import axes
from . import images
from . import orientation
from . import systems
from . import transforms
from . import typing

# trigger registration
from . import _xform_adaptors as _
from . import _xform_composers as _
from . import _xform_converters as _