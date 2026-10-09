"""
Shared machinery for formats stored in HDF5 files.

ITK `.h5` transforms and BIDS X5 files both keep their transforms under
a root `TransformGroup`, so only their content tells them apart. A
format therefore only says how to recognise and decode an open
`h5py.File`, and the classes of this module handle paths, streams,
bytes and lazy datasets. h5py is optional, so this module is only
imported by the formats that need it.
"""

__all__ = [
    "Hdf5Reader",
    "Hdf5ReaderWriter",
    "DelayedH5Array",
    "H5Like",
]

from ._delayed import DelayedH5Array
from ._parsers import H5Like, Hdf5Reader, Hdf5ReaderWriter
