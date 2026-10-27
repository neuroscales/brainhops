"""
Brainhops separates its generic representation of object types
(images, transformations, coordinate systems), implemented in the
[`datamodel`][] module, from the file formats that store them,
implemented in the [`io`][] module.

All operations can be run on CPU or GPU backends, and can be distributed
across multiple threads or processes using [`dask`][_dask]. The
computational backend is selected in the [`backends`][] module.

Modules
-------
backends
    Computational backend selection (numpy, cupy, dask)
datamodel
    General representation of images, transformations, coordinate
    systems, metadata, etc.
errors
    Exceptions raised by brainhops.
io
    Input/output of images, transformations, metadata, etc.
"""

__all__ = [
    "backends",
    "datamodel",
    "errors",
    "io",
]

import typing_extensions as tx

from . import backends, datamodel, errors, io

if tx.TYPE_CHECKING:
    import dask as _dask  # noqa: F401

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0+unknown"
