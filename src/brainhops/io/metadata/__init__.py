"""The metadata of file formats, and codecs between format-agnostic
metadata and files that only hold metadata, such as BIDS JSON sidecars.

The format-agnostic model itself lives in
[`brainhops.datamodel.metadata`][]. This package holds the base of the
metadata of every file format,
[`FileBasedMetadata`][brainhops.io.metadata.FileBasedMetadata], as
[`brainhops.io.images`][] holds `FileBasedImage`, and
[`OpaqueMetadata`][brainhops.io.metadata.OpaqueMetadata], for the formats
that store none. The metadata class of each format lives next to that
format's parser.
"""

__all__ = ["FileBasedMetadata", "OpaqueMetadata", "from_bids", "to_bids"]

from ._base import FileBasedMetadata, OpaqueMetadata
from .bids import from_bids, to_bids
