"""Format-independent machinery shared by every kind of file-based object."""

__all__ = [
    "FileBasedObject",
    "Format",
    "FormatDispatcher",
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

from . import parsers
from ._base import (
    BinaryFileBasedObject,
    FileBasedObject,
    Format,
    FormatDispatcher,
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
