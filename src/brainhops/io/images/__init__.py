"""Readers and writers for image formats."""

__all__ = [
    "FileBasedImage",
    "WritableFileBasedImage",
    "afni",
    "base",
    "load",
    "mrtrix",
    "sniff",
]

# internals
from brainhops._core.dependencies import (
    HAS_NIBABEL,
    HAS_PILLOW,
    HAS_TIFFFILE,
    has_abczarr_driver,
)
from brainhops.io.base._dispatch import register_missing_format

from . import afni, base, mrtrix
from .base import FileBasedImage, WritableFileBasedImage, load, sniff

# Formats must be imported for them to register themselves: the registry
# only ever holds classes that have actually been imported, so a lazily
# imported format would silently be invisible to `load`.
if HAS_NIBABEL:
    from . import freesurfer, nifti

    __all__ += ["freesurfer", "nifti"]

# Raster images (PNG, JPEG, ...) are read and written with Pillow.
if HAS_PILLOW:
    from . import pillow

    __all__ += ["pillow"]
else:
    register_missing_format(["pillow"], "Pillow", "pillow")

# TIFF images are read and written with tifffile. Without it, Pillow
# (whose TIFF sniff is weaker) reads them as raster images.
if HAS_TIFFFILE:
    from . import tiff

    __all__ += ["tiff"]
else:
    register_missing_format(["tiff", "tifffile"], "tifffile", "tiff")

# The Zarr reader needs abczarr and at least one of its backend drivers.
# abczarr alone cannot open a store, so the reader is registered only when a
# driver is present.
if has_abczarr_driver():
    from . import zarr

    __all__ += ["zarr"]
