"""Readers and writers of transformation file formats.

The entry points are [`load`][], which reads a transformation from a file
and detects its format, and [`sniff`][], which predicts that format without
parsing the file.
"""

__all__ = [
    "FileBasedTransformation",
    "base",
    "elastix",
    "freesurfer",
    "itk",
    "load",
    "matrix",
    "niftyreg",
    "sniff",
]

from brainhops._core.dependencies import has_abczarr_driver

from . import base, elastix, freesurfer, itk, matrix, niftyreg
from .base import (
    FileBasedTransformation,
    load,
    sniff,
)

# A format is visible to `load` only once its module has been imported,
# because the format registry holds imported classes only.
try:
    from . import nifti

    __all__ += ["nifti"]
except ImportError:
    pass

try:
    from . import spm

    __all__ += ["spm"]
except ImportError:
    pass

try:
    from . import fsl

    __all__ += ["fsl"]
except ImportError:
    pass

# The X5 reader needs the optional h5py dependency.
try:
    from . import x5

    __all__ += ["x5"]
except ImportError:
    pass

# abczarr alone cannot open a store, so the OME-Zarr field reader is
# registered only when at least one backend driver is also installed. The
# same gating applies to brainhops.io.images.zarr.
if has_abczarr_driver():
    from . import zarr

    __all__ += ["zarr"]
