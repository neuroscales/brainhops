"""Metadata read from the headers of files.

A file format that records metadata defines a subclass of
[`MetadataFormat`][], which holds the header of a file as the format
stores it. The `load` method of [`MetadataFormat`][] reads the header of
a file in any registered format, without reading its data.

A file format that holds data, such as an image, keeps its metadata in a
`metadata` field and the data as stored in the file in a `raw` field. The
`raw` field of such an object is usually a lazy proxy that reads the file
on demand. The `data` of the object is a cached view of that `raw` field,
decoded by a pair of functions that the format defines: reading `data`
returns the decoded `raw`, and setting `data` stores the encoded value in
`raw` and deletes the cached view. On write, the record of the metadata
is the base of the header, the geometry of the object is encoded over it,
and the data is written last.

A format whose `raw` field is a proxy overrides `from_filename` or
`from_file`, because the generic `from_filename` opens the file, passes
the stream to `from_fileobj` and closes the stream when that method
returns, which would leave the proxy without a file to read.
"""

__all__ = ["MetadataFormat"]

from ._base import MetadataFormat
