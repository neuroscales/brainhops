"""Codecs between format-agnostic metadata and files that only hold
metadata, such as BIDS JSON sidecars.

The format-agnostic model itself lives in
[`brainhops.datamodel.metadata`][]; the metadata of each file format
lives next to that format's parser.
"""

__all__ = ["from_bids", "to_bids"]

from .bids import from_bids, to_bids
