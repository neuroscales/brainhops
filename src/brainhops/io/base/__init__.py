"""The format-independent base shared by every file-based object, whatever
its kind."""

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
    "parsers",
    "raster",
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
    HAS_PILLOW,
    HAS_TIFFFILE,
    has_abczarr_driver,
)

from . import parsers, raster
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
    from . import nifti

    __all__ += ["nifti"]

if HAS_PILLOW:
    from . import pillow

    __all__ += ["pillow"]

if HAS_TIFFFILE:
    from . import tiff

    __all__ += ["tiff"]

# The Zarr store adapter needs abczarr and at least one backend driver.
# abczarr alone cannot open a store, so the adapter is exposed only when a
# driver is present.
if has_abczarr_driver():
    from . import zarr

    __all__ += ["zarr"]
