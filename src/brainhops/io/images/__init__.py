"""Readers and writers for image formats."""

__all__ = [
    "FileBasedImage",
    "WritableFileBasedImage",
    "base",
    "load",
    "sniff",
]

# internals
from brainhops._core.dependencies import HAS_ABCZARR, HAS_NIBABEL

from . import base
from .base import FileBasedImage, WritableFileBasedImage, load, sniff

# Formats must be imported for them to register themselves: the registry
# only ever holds classes that have actually been imported, so a lazily
# imported format would silently be invisible to `load`.
if HAS_NIBABEL:
    from . import nifti

    __all__ += ["nifti"]

if HAS_ABCZARR:
    from . import zarr

    __all__ += ["zarr"]
