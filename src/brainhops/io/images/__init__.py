"""Readers and writers for image formats."""

__all__ = [
    "FileBasedImage",
    "WritableFileBasedImage",
    "base",
    "load",
    "sniff",
]

# internals
from brainhops._core.dependencies import (
    HAS_NIBABEL,
    HAS_PILLOW,
    has_abczarr_driver,
)

from . import base
from .base import FileBasedImage, WritableFileBasedImage, load, sniff

# Formats must be imported for them to register themselves: the registry
# only ever holds classes that have actually been imported, so a lazily
# imported format would silently be invisible to `load`.
if HAS_NIBABEL:
    from . import nifti

    __all__ += ["nifti"]

# Raster images (PNG, JPEG, ...) are read and written with Pillow.
if HAS_PILLOW:
    from . import pillow

    __all__ += ["pillow"]

# The Zarr reader needs abczarr and at least one of its backend drivers.
# abczarr alone cannot open a store, so the reader is registered only when a
# driver is present.
if has_abczarr_driver():
    from . import zarr

    __all__ += ["zarr"]
