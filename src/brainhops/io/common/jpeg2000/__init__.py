"""
Reading the header of JPEG 2000 files.

JPEG 2000 stores an image as a codestream, which is a wavelet
decomposition of the image cut into tiles and coded in quality layers. A
codestream starts with a main header, made of marker segments that give
the size of the image area on a reference grid, the bit depth of each
component, the number of decomposition levels and the comments. It is
stored either alone, as a raw codestream (`.j2k`, `.j2c`, `.jpc`), or in
a JP2 file (`.jp2`, `.jpx`, `.jpf`), which is a sequence of boxes. The
boxes of a JP2 file add a capture and a display resolution, a palette,
and XML and UUID boxes such as GMLJP2 and XMP.

Everything but the coded image is held as a [`Jpeg2000Raw`][] record,
and [`Jpeg2000Metadata`][] reads and writes that record without
decoding the image. The record is parsed by brainhops itself, and it
keeps the bytes of the file up to its first tile, so that its header can
be written again. [`Jpeg2000Image`][brainhops.io.images.jpeg2000.Jpeg2000Image]
holds such metadata and a lazy array of the samples, which Pillow
decodes. A record whose comments, resolutions, boxes or container
change is written by rewriting these parts of the file, and the coded
image is copied as it is.
"""

__all__ = [
    "Jpeg2000Metadata",
    "Jpeg2000Raw",
]

from ._metadata import Jpeg2000Metadata
from ._raw import Jpeg2000Raw
