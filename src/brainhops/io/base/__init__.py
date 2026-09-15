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
    "sniff",
    "parsers",
    "register_format",
]

from brainhops._core.dependencies import HAS_NIBABEL, has_abczarr_driver

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

if HAS_NIBABEL:
    from . import nifti

    __all__ += ["nifti"]

# The Zarr store adapter needs abczarr and at least one backend driver.
# abczarr alone cannot open a store, so the adapter is exposed only when a
# driver is present.
if has_abczarr_driver():
    from . import zarr

    __all__ += ["zarr"]
