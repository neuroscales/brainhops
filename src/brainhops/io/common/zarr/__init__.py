"""
Adapt the file-based reader and writer surface to a Zarr store.

A Zarr node is a directory-backed store rather than a single file, so the
file-handle machinery of the parser base does not apply to it. This mixin
opens the store through abczarr instead. It inherits the common parser
class so that dispatch, `load`, and `save` reach it the same way they reach
every other reader, and it routes their file operations to a store path.

The mixin is object-agnostic. An image reader and a transformation reader
both mix it in and share its store handling. The public entry points
pair an open-node door with a location door in each direction:
[from_node][ZarrParser.from_node] and [to_node][ZarrParser.to_node]
operate on an opened node, and [from_store][ZarrParser.from_store] and
[to_store][ZarrParser.to_store] operate on a store location.
"""

__all__ = [
    "ZarrParser",
    "ZarrParserWriter",
    "StoreLike",
]

from ._parsers import StoreLike, ZarrParser, ZarrParserWriter
