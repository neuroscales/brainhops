"""Concrete file formats shared by images and transformations.

NIfTI requires nibabel, and Zarr requires `abczarr` with a backend driver.
"""

__all__ = [
    "FileBasedObject",
    "WritableFileBasedObject",
    "TextFileBasedObject",
    "BinaryFileBasedObject",
    "WritableTextFileBasedObject",
    "WritableBinaryFileBasedObject",
    "format_registry",
    "load",
    "save",
    "sniff",
    "afni",
    "mrtrix",
    "parsers",
    "freesurfer",
    "register_format",
    "ImageSpec",
    "Parser",
    "OperationSpec",
    "SourceSpec",
    "TransformationSpec",
    "format_hints",
    "parser_for",
    "register_parser",
]

from brainhops._core.dependencies import (
    HAS_NIBABEL,
    has_abczarr_driver,
)

from . import afni, freesurfer, mgh, minc, mrtrix

if HAS_NIBABEL:
    from ..common import nifti

    __all__ += ["mgh", "minc", "nifti"]

# abczarr cannot open a store without a backend driver.
if has_abczarr_driver():
    from . import zarr

    __all__ += ["zarr"]
