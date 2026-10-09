"""Generic representation of coordinate systems and transformations.

The model follows OME-NGFF but is more flexible, so that it can represent
the transformations of many formats:

- OME-NGFF
- FreeSurfer LTA (affine)
- Freesurfer XFM (nonlinear)
- ANTs
- SPM
- nitorch
- ...

FSL transformations are not supported, because applying one requires the
metadata of its fixed and moving images, which it does not store.
"""

__all__ = [
    "axes",
    "base",
    "enums",
    "kinds",
    "images",
    "orientations",
    "systems",
    "transformations",
    "units",
]

# Importing the modules registers their classes.
from . import (
    axes,
    base,
    enums,
    images,
    kinds,
    orientations,
    systems,
    transformations,
    units,
)
