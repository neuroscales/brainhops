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
    HAS_H5PY,
    HAS_NIBABEL,
    HAS_OPENSLIDE,
    HAS_PILLOW,
    HAS_TIFFFILE,
)
from brainhops._core.lazy import lazy_exports
from brainhops.io.base._formats import register_missing_format

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
    __all__ += ["pillow"]
    _EXPORTS.update(pillow=".pillow")
else:
    register_missing_format(["pillow"], "Pillow", "pillow")

# TIFF images are read and written with tifffile. Without it, Pillow
# (whose TIFF sniff is weaker) reads them as raster images.
if HAS_TIFFFILE:
    __all__ += ["tiff"]
    _EXPORTS.update(tiff=".tiff")
else:
    register_missing_format(["tiff", "tifffile"], "tifffile", "tiff")

# Whole-slide images are read with OpenSlide (openslide-python and the
# OpenSlide library). Without it, the TIFF-based slides are read by the
# TIFF reader.
if HAS_OPENSLIDE:
    __all__ += ["openslide"]
    _EXPORTS.update(openslide=".openslide")
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
# Whether a driver is present may need abczarr imported, which is left to
# the first dispatch (see `brainhops.io.base._formats`).
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
