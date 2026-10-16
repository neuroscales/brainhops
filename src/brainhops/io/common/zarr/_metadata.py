"""The metadata of a Zarr array."""

# dependencies
import typing_extensions as tx
from abczarr import ZarrArray, ZarrNode
from bagof.magic import NoRepr

# internals
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import Confidence, ParserContentError
from brainhops.io.metadata import MetadataFormat

# this format
from ._parsers import ZarrReader, opened_node
from ._raw import ZarrRaw


@register_format
class ZarrMetadata(ZarrReader, MetadataFormat):
    """Metadata of a Zarr array, which is the document of the array.

    The metadata of a Zarr array is the document that its store keeps,
    `zarr.json` in Zarr version 3, or `.zarray` and `.zattrs` in version
    2, held as a [`ZarrRaw`][] record.
    [`ZarrImage`][brainhops.io.images.zarr.ZarrImage] uses that record as
    the base of the document that it writes. Reading the metadata never
    reads a chunk, so a store whose chunks were removed can still be
    read. The metadata keeps no handle on the store.

    A store is a directory or a remote prefix of keys, not a file, so the
    metadata is read from a store location with [`from_store`][] or from
    an opened array with [`from_node`][], and bytes and streams are
    refused. The metadata writes nothing by itself: the documents are
    written by the image, with its chunks.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".zarr",)

    raw: NoRepr[tx.Optional[ZarrRaw]] = None
    """The record of the array, or `None` for metadata without a record."""

    @classmethod
    def from_node(cls, node: tx.Any, **kwargs) -> "ZarrMetadata":
        """Read the metadata of an opened Zarr array.

        Parameters
        ----------
        node : ZarrArray or native Zarr array
            The opened array.
        **kwargs : Any
            Refused, since the metadata has no reader option.

        Returns
        -------
        ZarrMetadata
            The metadata, which holds the record of the array.

        Raises
        ------
        TypeError
            If an option is given.
        ParserTypeError
            If `node` is a path, or an object that cannot be wrapped.
        ParserContentError
            If the node is a group.
        """
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"Unknown options of ZarrMetadata: {names}")
        node = opened_node(node)
        if not isinstance(node, ZarrArray):
            raise ParserContentError(
                "This Zarr store is a group, not a plain array, so it has "
                "no array document to read."
            )
        return cls.from_raw(ZarrRaw.from_node(node))

    @classmethod
    def _score_store(cls, node: ZarrNode) -> float:
        # Any array has the metadata of an array, and a group has none.
        if isinstance(node, ZarrArray):
            return Confidence.LIKELY
        return Confidence.NO
