"""Format-independent base classes and loaders for file-based images."""

__all__ = [
    "FileBasedImage",
    "load",
    "sniff",
]

from ._base import FileBasedImage
from ._load import load, sniff
