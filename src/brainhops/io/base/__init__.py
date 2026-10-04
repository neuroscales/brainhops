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

# dependencies
import typing_extensions as tx

# internals
from brainhops._core.dependencies import HAS_ABCZARR, HAS_NIBABEL
from brainhops._core.lazy import lazy_exports

_EXPORTS = {
    "FileBasedObject": "._base",
    "WritableFileBasedObject": "._base",
    "TextFileBasedObject": "._base",
    "BinaryFileBasedObject": "._base",
    "WritableTextFileBasedObject": "._base",
    "WritableBinaryFileBasedObject": "._base",
    "format_registry": "._base",
    "register_format": "._base",
    "load": "._load",
    "sniff": "._load",
    "save": "._save",
    "afni": ".afni",
    "mrtrix": ".mrtrix",
    "parsers": ".parsers",
    "freesurfer": ".freesurfer",
    "ImageSpec": ".specs",
    "Parser": ".specs",
    "OperationSpec": ".specs",
    "SourceSpec": ".specs",
    "TransformationSpec": ".specs",
    "format_hints": ".specs",
    "parser_for": ".specs",
    "register_parser": ".specs",
}

# The NIfTI, MGH and MINC helpers need nibabel, which is optional.
if HAS_NIBABEL:
    __all__ += ["mgh", "minc", "nifti"]
    _EXPORTS.update(mgh=".mgh", minc=".minc", nifti=".nifti")

# The Zarr store adapter needs abczarr. Whether a backend driver is
# present too is only known by importing abczarr and its drivers, which
# is left to the first use of a Zarr store.
if HAS_ABCZARR:
    __all__ += ["zarr"]
    _EXPORTS.update(zarr=".zarr")

__getattr__, __dir__ = lazy_exports(__name__, globals(), _EXPORTS)

if tx.TYPE_CHECKING:
    from . import afni, freesurfer, mgh, minc, mrtrix, nifti, parsers, zarr
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
