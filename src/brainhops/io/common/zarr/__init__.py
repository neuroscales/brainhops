"""Records, metadata and reader adapters of Zarr stores.

A Zarr store is a directory, or a prefix of keys in a remote store, opened
through `abczarr`, and not a file that can be read as a stream. The
classes of this package hold the document of a Zarr array as a record,
read and write it as metadata, and route the file-based reader and writer
interface to store locations, so that the image and transformation
formats stored as Zarr are loaded and saved like any other format.

[`ZarrRaw`][] holds the document of an array, `zarr.json` in Zarr version
3 or `.zarray` and `.zattrs` in version 2, as the typed model of abczarr,
together with the bytes of each document as read. [`ZarrMetadata`][]
holds such a record. [`ZarrReader`][] and [`ZarrReaderWriter`][] are the
stateless adapters that the Zarr formats inherit.
"""

__all__ = [
    "StoreLike",
    "ZarrMetadata",
    "ZarrRaw",
    "ZarrReader",
    "ZarrReaderWriter",
]

from ._metadata import ZarrMetadata
from ._parsers import StoreLike, ZarrReader, ZarrReaderWriter
from ._raw import ZarrRaw
