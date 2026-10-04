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

# The formats are declared here, ahead of import, so that `load` finds
# every one of them while the subpackages below are imported lazily: a
# format's module is imported when dispatch first needs it, or when it is
# first accessed.
from . import _entries  # noqa: F401

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
# drivers. Whether a driver is present is only known by importing
# abczarr, which is left to the first use of the reader. This mirrors how
# io.images gates io.images.zarr.
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
