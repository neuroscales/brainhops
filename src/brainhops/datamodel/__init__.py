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
    "kinds",
    "images",
    "orientation",
    "systems",
    "transformations",
    "units",
]

# dependencies
import typing_extensions as tx

# internals
from brainhops._core.lazy import lazy_exports

# The submodules are imported on first access. Each one imports what it
# builds on, and registers what it defines as it is imported, so no
# submodule needs another one to have been imported first.
__getattr__, __dir__ = lazy_exports(
    __name__, globals(), {name: "." + name for name in __all__}
)

if tx.TYPE_CHECKING:
    from . import (
        axes,
        base,
        enums,
        images,
        kinds,
        orientation,
        systems,
        transformations,
        units,
    )
