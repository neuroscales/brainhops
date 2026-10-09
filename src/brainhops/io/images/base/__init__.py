"""Format-independent base classes and loaders for file-based images."""

__all__ = [
    "FileBasedImage",
    "WritableFileBasedImage",
    "load",
    "sniff",
]

from ._base import FileBasedImage, WritableFileBasedImage
from ._load import load, sniff
