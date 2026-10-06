"""
Reading the metadata of a file without its data.

A file stores its metadata in a raw record (a NIfTI header, the
attributes of a Zarr array, the JSON of an x5 node, ...), which the
metadata class of the format decodes with `from_raw` and encodes with
`to_raw` and `update_raw`. [`MetadataParser`][] adds the file side: its
`from_*` methods read the raw record of a file, and nothing else, then
build the metadata with `from_raw`. The metadata class of a format lists
it first among its bases, as the image class of a format lists its
parser.

The parsers own no registry: the dispatcher among the formats is
[`FileBasedMetadata`][brainhops.io.metadata.FileBasedMetadata],
whose `load` reads the metadata of a file in any of these formats.
"""

__all__ = ["MetadataParser"]

# internals
from .parsers import FileParser


class MetadataParser(FileParser):
    """
    Reads the metadata of a file of one format.

    A format implements `from_fileobj`, which reads the raw record of an
    open file and builds the metadata with `from_raw`, and the sniffers
    that recognise its files; a path is opened by `from_filename`, in
    binary mode, and handed to `from_fileobj`, and bytes are wrapped in a
    stream and handed to it too (`FileParser.from_bytes` does so for a
    class that implements `from_fileobj`). A format that reads paths
    otherwise (MGH reads its tags lazily from a path) overrides
    `from_filename` too.
    The parser only reads, as a `FileParser` does: a format whose record
    is an object of its own on disk (the attributes of a Zarr array)
    defines its own `to_file`, and the record of any other format is
    written by the writer of its images or transformations, along with
    the data. The parser of a format stored in HDF5 is
    [`Hdf5MetadataParser`][brainhops.io.base.hdf5.Hdf5MetadataParser].

    The parser of a format owns no registry: the class of the format
    also derives from
    [`FileBasedMetadata`][brainhops.io.metadata.FileBasedMetadata],
    and registers into its registry with
    [`register_format`][brainhops.io.base.register_format].
    """

    _READ_MODE = "rb"
