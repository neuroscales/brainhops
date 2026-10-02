"""Readers and writers for image formats."""

__all__ = [
    "FileBasedImage",
    "WritableFileBasedImage",
    "base",
    "load",
    "mrtrix",
    "nrrd",
    "sniff",
]

# internals
from brainhops._core.dependencies import (
    HAS_H5PY,
    HAS_NIBABEL,
    HAS_OPENSLIDE,
    HAS_PILLOW,
    HAS_TIFFFILE,
    has_abczarr_driver,
)
from brainhops.io.base._dispatch import register_missing_format

from . import base, mrtrix, nrrd
from .base import FileBasedImage, WritableFileBasedImage, load, sniff

# Formats must be imported for them to register themselves: the registry
# only ever holds classes that have actually been imported, so a lazily
# imported format would silently be invisible to `load`.
if HAS_NIBABEL:
    from . import freesurfer, minc, nifti

    __all__ += ["freesurfer", "minc", "nifti"]
    # MINC2 is an HDF5 file, read only with h5py.
    if not HAS_H5PY:
        register_missing_format(
            ["minc2", "minc.2", "minc.minc2"], "h5py", "minc"
        )
else:
    register_missing_format(
        ["minc", "minc1", "minc2", "minc.1", "minc.2"], "nibabel", "minc"
    )

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

# Whole-slide images are read with OpenSlide (openslide-python and the
# OpenSlide library). Without it, the TIFF-based slides are read by the
# TIFF reader.
if HAS_OPENSLIDE:
    from . import openslide

    __all__ += ["openslide"]
else:
    register_missing_format(
        [
            "openslide",
            "aperio",
            "svs",
            "hamamatsu",
            "ndpi",
            "vms",
            "vmu",
            "mirax",
            "mrxs",
            "3dhistech",
            "leica",
            "scn",
            "philips",
            "ventana",
            "bif",
            "sakura",
            "svslide",
            "trestle",
            "zeiss",
            "czi",
            "dicom-wsi",
            "generic-tiff",
        ],
        "openslide-python",
        "openslide",
    )

# The Zarr reader needs abczarr and at least one of its backend drivers.
# abczarr alone cannot open a store, so the reader is registered only when a
# driver is present.
if has_abczarr_driver():
    from . import zarr

    __all__ += ["zarr"]
