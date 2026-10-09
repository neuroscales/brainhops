"""Format-independent machinery shared by every kind of file-based object."""

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

# TODO: stop importing from common here; import from brainhops.io.common.
from ..common import afni, freesurfer, mgh, minc, mrtrix
from . import parsers
from ._base import (
    BinaryFileBasedObject,
    FileBasedObject,
    TextFileBasedObject,
    WritableBinaryFileBasedObject,
    WritableFileBasedObject,
    WritableTextFileBasedObject,
    format_registry,
    register_format,
)
from ._load import load, sniff
from ._save import save
from .specs import (
    ImageSpec,
    OperationSpec,
    Parser,
    SourceSpec,
    TransformationSpec,
    format_hints,
    parser_for,
    register_parser,
)

if HAS_NIBABEL:
    from ..common import nifti

    __all__ += ["mgh", "minc", "nifti"]

# abczarr cannot open a store without a backend driver.
if has_abczarr_driver():
    from ..common import zarr

    __all__ += ["zarr"]
