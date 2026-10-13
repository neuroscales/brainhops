"""Format-independent base classes and loaders for file-based images."""

__all__ = [
    "ImageFormat",
    "load",
    "sniff",
]

from ._base import ImageFormat
from ._load import load, sniff
