"""The metadata of a raster image read with Pillow."""

# dependencies
import typing_extensions as tx
from bagof.magic import NoRepr

# internals
from brainhops.io.base._base import register_format
from brainhops.io.metadata import MetadataFormat, MetadataReader

# this format
from ._raw import PillowRaw
from ._utils import EXTENSIONS


@register_format
class PillowMetadata(MetadataReader, MetadataFormat):
    """Metadata of a raster image, which is its header as Pillow reads it.

    The metadata of a raster file, such as a PNG or a JPEG file, is held
    as a [`PillowRaw`][] record: the format of the file, the mode and the
    size of a frame, its `info` dictionary, and the number of frames.
    [`PillowImage`][brainhops.io.images.pillow.PillowImage] holds such
    metadata and a lazy array of the pixels of the frame. Reading the
    metadata never decodes the pixels, and a file cut after its header can
    still be read. The metadata keeps no open file.

    Pillow writes a header only together with the pixels that follow it,
    so the metadata cannot be written alone. The image writes it.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = EXTENSIONS
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("pillow",)
    RAW: tx.ClassVar[type] = PillowRaw

    raw: NoRepr[tx.Optional[PillowRaw]] = None
    """The record of the image, or `None` for metadata without a record."""
