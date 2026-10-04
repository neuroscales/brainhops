"""Readers and writers for transformation formats."""

__all__ = [
    "FileBasedTransformation",
    "WritableFileBasedTransformation",
    "base",
    "elastix",
    "freesurfer",
    "itk",
    "load",
    "matrix",
    "niftyreg",
    "sniff",
]

# dependencies
import typing_extensions as tx

# internals
from brainhops._core.dependencies import HAS_ABCZARR, HAS_H5PY, HAS_NIBABEL
from brainhops._core.lazy import lazy_exports

_EXPORTS = {
    "FileBasedTransformation": ".base",
    "WritableFileBasedTransformation": ".base",
    "load": ".base",
    "sniff": ".base",
    "base": ".base",
    "elastix": ".elastix",
    "freesurfer": ".freesurfer",
    "itk": ".itk",
    "matrix": ".matrix",
    "niftyreg": ".niftyreg",
}

# The NIfTI, SPM and FSL formats are read with nibabel.
if HAS_NIBABEL:
    __all__ += ["nifti", "spm", "fsl"]
    _EXPORTS.update(nifti=".nifti", spm=".spm", fsl=".fsl")

# The X5 reader needs h5py.
if HAS_H5PY:
    __all__ += ["x5"]
    _EXPORTS.update(x5=".x5")

# The OME-Zarr field reader needs abczarr and at least one of its backend
# drivers, as io.images.zarr does (see `brainhops.io.base._formats`).
if HAS_ABCZARR:
    __all__ += ["zarr"]
    _EXPORTS.update(zarr=".zarr")

__getattr__, __dir__ = lazy_exports(__name__, globals(), _EXPORTS)

if tx.TYPE_CHECKING:
    from . import (
        base,
        elastix,
        freesurfer,
        fsl,
        itk,
        matrix,
        nifti,
        niftyreg,
        spm,
        x5,
        zarr,
    )
    from .base import (
        FileBasedTransformation,
        WritableFileBasedTransformation,
        load,
        sniff,
    )
