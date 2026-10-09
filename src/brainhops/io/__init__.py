"""Transformations, images, meshes and streamlines stored in files.

Each file format also implements the generic interface of the object it holds,
whether the file is on a local disk or in a cloud store, so a file-backed
object can be used wherever the generic object is expected.
"""

__all__ = [
    "FileBasedObject",
    "WritableFileBasedObject",
    "base",
    "images",
    "load",
    "save",
    "sniff",
    "transformations",
    "vectors",
]

from . import base, images, transformations, vectors
from .base import FileBasedObject, WritableFileBasedObject, load, save, sniff
