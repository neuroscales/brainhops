"""Metadata read from the headers of files.

A file format that records metadata defines a subclass of
[`MetadataFormat`][], which holds the header of a file as the format
stores it. The `load` method of [`MetadataFormat`][] reads the header of
a file in any registered format, without reading its data.
"""

__all__ = ["MetadataFormat"]

from ._base import MetadataFormat
