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
    "save",
    "sniff",
    "transformations",
    "vectors",
]

# dependencies
import typing_extensions as tx

# internals
from brainhops._core.lazy import lazy_exports

# The subpackages are imported on first access. `load`, `save` and
# `sniff` import the formats they dispatch to themselves, so they find
# every format whether or not `images` and `transformations` were
# accessed first.
__getattr__, __dir__ = lazy_exports(
    __name__,
    globals(),
    {
        "FileBasedObject": ".base",
        "WritableFileBasedObject": ".base",
        "base": ".base",
        "images": ".images",
        "load": ".base",
        "save": ".base",
        "sniff": ".base",
        "transformations": ".transformations",
        "vectors": ".vectors",
    },
)

if tx.TYPE_CHECKING:
    from . import base, images, transformations, vectors
    from .base import (
        FileBasedObject,
        WritableFileBasedObject,
        load,
        save,
        sniff,
    )
