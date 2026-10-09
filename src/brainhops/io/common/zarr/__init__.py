"""Parser and writer support for Zarr stores.

A Zarr store is a directory opened through `abczarr`, not a file that can be
read as a stream. The classes in this module route the file-based parser
interface to store locations, so that image and transformation formats stored
as Zarr are loaded and saved like any other format.
"""

__all__ = [
    "ZarrParser",
    "ZarrParserWriter",
    "StoreLike",
]

from ._parsers import StoreLike, ZarrParser, ZarrParserWriter
