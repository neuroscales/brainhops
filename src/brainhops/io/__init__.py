"""
This module implements a generic API for interacting with
transformations, images, meshes, or streamlines that are stored in
some format on disk or on the cloud.

Each format also implements the more general
transformations/images/meshes/streamlines API that does not assume that
the data is stored at a specific path. In other words, a file-backed
object is a special case of a generic object, and the file-backed API
is a special case of the generic API.
"""

__all__ = [
    "FileBasedObject",
    "WritableFileBasedObject",
    "base",
    "images",
    "load",
    "sniff",
    "transformations",
    "vectors",
]

from . import base, images, transformations, vectors
from .base import FileBasedObject, WritableFileBasedObject, load, sniff
