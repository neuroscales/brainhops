"""Readers and writers for image file formats."""

__all__ = [
    "FileBasedImage",
    "afni",
    "base",
    "load",
    "mrtrix",
    "nrrd",
    "sniff",
]

from brainhops._core.dependencies import (
    HAS_H5PY,
    HAS_NIBABEL,
    HAS_OPENSLIDE,
    HAS_PILLOW,
    HAS_TIFFFILE,
    has_abczarr_driver,
)
from brainhops.io.base._dispatch import register_missing_format

from . import afni, base, mrtrix, nrrd
from .base import FileBasedImage, load, sniff

# Formats register themselves when their module is imported, and the
# registry only knows imported classes, so every available format is
# imported eagerly to make it visible to `load`.
if HAS_NIBABEL:
    from . import freesurfer, minc, nifti

    __all__ += ["freesurfer", "minc", "nifti"]
    # MINC2 files are HDF5 files and need h5py to be read.
    if not HAS_H5PY:
        register_missing_format(
            ["minc2", "minc.2", "minc.minc2"], "h5py", "minc"
        )
else:
    register_missing_format(
        ["minc", "minc1", "minc2", "minc.1", "minc.2"], "nibabel", "minc"
    )

# Raster formats (PNG, JPEG, ...) are read and written through Pillow.
if HAS_PILLOW:
    from . import pillow

    __all__ += ["pillow"]
else:
    register_missing_format(["pillow"], "Pillow", "pillow")

# Without tifffile, Pillow still reads TIFF files as raster images, but
# with weaker sniffing.
if HAS_TIFFFILE:
    from . import tiff

    __all__ += ["tiff"]
else:
    register_missing_format(["tiff", "tifffile"], "tifffile", "tiff")

# OpenSlide needs both the Python package and the C library. Without it,
# TIFF-based slides fall back to the TIFF reader.
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

# abczarr cannot open stores without at least one backend driver, so the
# Zarr format is registered only when a driver is available.
if has_abczarr_driver():
    from . import zarr

    __all__ += ["zarr"]
