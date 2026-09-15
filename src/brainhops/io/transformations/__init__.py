"""Readers and writers for transformation formats."""

__all__ = [
    "FileBasedTransformation",
    "WritableFileBasedTransformation",
    "base",
    "freesurfer",
    "itk",
    "load",
    "sniff",
]

# internals
from brainhops._core.dependencies import has_abczarr_driver

from . import base, freesurfer, itk
from .base import (
    FileBasedTransformation,
    WritableFileBasedTransformation,
    load,
    sniff,
)

# Formats must be imported for them to register themselves: the registry
# only holds classes that have actually been imported, so a format left
# unimported would silently be invisible to `load`.
try:
    from . import nifti

    __all__ += ["nifti"]
except ImportError:  # nibabel is optional
    pass

try:
    from . import spm

    __all__ += ["spm"]
except ImportError:  # nibabel is optional
    pass

try:
    from . import fsl

    __all__ += ["fsl"]
except ImportError:  # nibabel is optional
    pass

# The OME-Zarr field reader needs abczarr and at least one backend driver.
# abczarr alone cannot open a store, so the reader is registered only when a
# driver is present. This mirrors how io.images gates io.images.zarr.
if has_abczarr_driver():
    from . import zarr

    __all__ += ["zarr"]
