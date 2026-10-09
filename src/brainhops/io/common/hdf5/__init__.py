"""
Shared plumbing for formats stored in HDF5 files.

Several transformation formats are HDF5 containers -- ITK's binary
transforms (`.h5`) and BIDS X5 (`.x5`) among them -- and they cannot be
told apart by their container: both are HDF5 files, and both even keep
their transforms under a root group named `TransformGroup`. What
differs is their *content*, so each format only says how to recognise
and decode an open [`h5py.File`][h5py.File], and the mixins here deal
with everything else: paths, open binary streams, bytes held in memory,
and lazily read datasets.

- [`Hdf5Parser`][] turns every input that `brainhops` accepts into an
  open `h5py.File`, and hands it to the format's `sniff_h5` and
  `from_h5`.
- [`Hdf5ParserWriter`][] does the same for writing, through the
  format's `to_h5`.
- [`DelayedH5Array`][] is a dataset that can still be read after its
  file was closed, by reopening it on demand.

`h5py` is an optional dependency, so this package is imported only by
the formats that need it, and only when it is installed.
"""

__all__ = [
    "Hdf5Parser",
    "Hdf5ParserWriter",
    "DelayedH5Array",
    "H5Like",
]

from ._delayed import DelayedH5Array
from ._parsers import H5Like, Hdf5Parser, Hdf5ParserWriter
