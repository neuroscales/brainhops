"""Readers and writers for image formats."""

__all__ = [
    "FileBasedImage",
    "WritableFileBasedImage",
    "afni",
    "base",
    "load",
    "mrtrix",
    "nrrd",
    "sniff",
]

# dependencies
import typing_extensions as tx

# internals
from brainhops._core.dependencies import (
    HAS_ABCZARR,
    HAS_NIBABEL,
    HAS_OPENSLIDE,
    HAS_PILLOW,
    HAS_TIFFFILE,
)
from brainhops._core.lazy import lazy_exports

# The formats are declared here, ahead of import, so that `load` finds
# every one of them while the subpackages below are imported lazily: a
# format's module is imported when dispatch first needs it, or when it is
# first accessed.
from . import _entries  # noqa: F401

_EXPORTS = {
    "FileBasedImage": ".base",
    "WritableFileBasedImage": ".base",
    "load": ".base",
    "sniff": ".base",
    "afni": ".afni",
    "base": ".base",
    "mrtrix": ".mrtrix",
    "nrrd": ".nrrd",
}

# MINC, MGH and NIfTI images are read with nibabel.
if HAS_NIBABEL:
    __all__ += ["freesurfer", "minc", "nifti"]
    _EXPORTS.update(freesurfer=".freesurfer", minc=".minc", nifti=".nifti")

# Raster images (PNG, JPEG, ...) are read and written with Pillow.
if HAS_PILLOW:
    __all__ += ["pillow"]
    _EXPORTS.update(pillow=".pillow")

# TIFF images are read and written with tifffile.
if HAS_TIFFFILE:
    __all__ += ["tiff"]
    _EXPORTS.update(tiff=".tiff")

# Whole-slide images are read with OpenSlide.
if HAS_OPENSLIDE:
    __all__ += ["openslide"]
    _EXPORTS.update(openslide=".openslide")

# The Zarr reader needs abczarr and at least one of its backend drivers.
# Whether a driver is present is only known by importing abczarr, which
# is left to the first use of the reader.
if HAS_ABCZARR:
    __all__ += ["zarr"]
    _EXPORTS.update(zarr=".zarr")

__getattr__, __dir__ = lazy_exports(__name__, globals(), _EXPORTS)

if tx.TYPE_CHECKING:
    from . import (
        afni,
        base,
        freesurfer,
        minc,
        mrtrix,
        nifti,
        nrrd,
        openslide,
        pillow,
        tiff,
        zarr,
    )
    from .base import FileBasedImage, WritableFileBasedImage, load, sniff
