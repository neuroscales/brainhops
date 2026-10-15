"""Metadata read from the headers of files.

A file format that records metadata defines a subclass of
[`MetadataFormat`][], which holds the header of a file as the format
stores it, and the common vocabulary of
[`Metadata`][brainhops.datamodel.metadata.Metadata], decoded from the
header. The `load` method of [`MetadataFormat`][] reads the header of a
file in any registered format, without reading its data. The fields of
the vocabulary that a format can store are listed in its constant
`SUPPORTED_FIELDS`, and `consolidate` encodes the fields that changed
into a copy of the header.

A file format that holds data, such as an image, keeps its metadata in a
`metadata` field and the data as stored in the file in a `rawdata`
field. The `rawdata` field of such an object is usually a lazy proxy that
reads the file on demand. The `data` of the object is a cached view of
that `rawdata` field, decoded by a pair of functions that the format
defines. On write, the record of the metadata is the base of the header,
the geometry of the object is encoded over it, and the data is written
last.

A format whose `rawdata` field is a proxy overrides `from_filename` or
`from_file`, because the generic `from_filename` opens the file, passes
the stream to `from_fileobj` and closes the stream when that method
returns, which would leave the proxy without a file to read.

[`BidsSidecar`][] is the metadata of a BIDS JSON sidecar, a file of
metadata only, and [`from_bids`][] and [`to_bids`][] convert generic
metadata from and to the JSON object of a sidecar.
"""

__all__ = ["MetadataFormat", "BidsSidecar", "from_bids", "to_bids"]

from ._base import MetadataFormat
from ._bids import BidsSidecar, from_bids, to_bids
