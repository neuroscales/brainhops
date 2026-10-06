# stdlib

# dependencies
import typing_extensions as tx
from abczarr import ZarrArray, ZarrNode, create
from bagof.magic import Factory

# internals
from brainhops._core.dependencies import da
from brainhops._core.properties import smartproperty
from brainhops._core.typing import ArrayProtocol

# backends
from brainhops.backends import get_array_backend
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.metadata._field import MetadataField
from brainhops.datamodel.metadata._report import apply_loss_policy
from brainhops.datamodel.transformations import Transformation
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    Confidence,
    ParserContentError,
    WriterError,
)
from brainhops.io.base.zarr import (
    StoreLike,
    ZarrParserWriter,
    _as_node,
)
from brainhops.io.images.base import WritableFileBasedImage
from brainhops.io.metadata._sync import parent_post_init, sync_metadata

from ._metadata import (
    ZarrMetadata,
    ZarrRaw,
    node_attributes,
    write_attributes,
)


@register_format
class ZarrImage(ZarrParserWriter, WritableFileBasedImage, SingleScaleImage):
    """
    An image that is encoded by a plain Zarr array.

    A plain Zarr array carries no world geometry, so the image is read with
    an identity geometry unless a voxel-to-world transformation is supplied
    through the `transformation` argument. An OME-Zarr pyramid, whose group
    carries a multiscale geometry, is read by
    [OmeZarrImage][brainhops.io.images.zarr.OmeZarrImage] instead.

    The array is held as a handle and its data is read on first access, so
    opening the image does not read the array.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".zarr",)

    metadata: MetadataField[
        ZarrMetadata,
        Factory(),
        tx.Doc(
            """
            The metadata of the array: the vocabulary, stored as a sidecar
            under its attribute `"brainhops"`, and its other attributes as
            `extra`; the attributes are the raw record (`metadata.raw`).
            See [`ZarrMetadata`][brainhops.io.images.zarr.ZarrMetadata].
            """
        ),
    ]

    def __post_init__(self, arguments: tx.Any = None) -> None:
        parent_post_init(super(), arguments)
        self._sync_metadata()

    def _sync_metadata(self) -> None:
        """Read the metadata from the attributes of the array, when it
        was not read from this array yet."""
        node = self.node
        if node is not None:
            sync_metadata(
                self,
                ZarrMetadata,
                lambda: ZarrRaw(node_attributes(node), node),
                same=lambda held: held.node is node,
                image=self,
            )

    def _write_metadata(self, node: tx.Any, on_loss: tx.Any) -> None:
        metadata = self.metadata
        if metadata is None:
            return
        metadata, report = ZarrMetadata.writable(metadata)
        record = metadata.update_raw(image=self, on_loss=report)
        metadata.check_raw(record, image=self, on_loss=report)
        apply_loss_policy(report, on_loss, stacklevel=4)
        write_attributes(node, record.attrs, metadata.attributes)

    @smartproperty(cache=True)
    def data(self) -> tx.Optional[ArrayProtocol]:
        node = self.node
        if node is not None:
            backend = get_array_backend()
            if backend is da:
                return node.to_dask(chunks="chunks")
            return backend.asarray(node[...])
        return None

    # --- sniff --------------------------------------------------------

    @classmethod
    def _score_store(cls, node: ZarrNode) -> float:
        # A plain array is very likely wanted as an image. A group is not an
        # array, so it is left to the OME reader.
        if isinstance(node, ZarrArray):
            return Confidence.LIKELY
        return Confidence.NO

    # --- load ---------------------------------------------------------

    @classmethod
    def from_node(
        cls,
        node: tx.Any,
        transformation: tx.Optional[Transformation] = None,
        **kwargs,
    ) -> tx.Self:
        """
        Read the image from an opened Zarr array.

        A plain array carries no world geometry, so `transformation` supplies
        the voxel-to-world placement to read it with. Without one the image
        is read with an identity geometry.

        A group is refused: it carries no array to read, and reading one as
        an image would fail later and more obscurely.
        """
        image = super().from_node(node, **kwargs)
        if not isinstance(image.node, ZarrArray):
            raise ParserContentError(
                "This Zarr store is a group, not a plain array, so it cannot "
                "be read as a single-scale image."
            )
        if transformation is not None:
            image.transformation = transformation
        return image

    # --- save ---------------------------------------------------------

    def to_node(self, node: tx.Any, **kwargs) -> ZarrNode:
        """
        Write the image into an opened Zarr array, and return it.

        The metadata is written into the attributes of the array; what it
        cannot hold is reported according to `on_loss`.
        """
        on_loss = kwargs.pop("on_loss", None)
        wrapped = _as_node(node)
        data = self.data
        if not isinstance(wrapped, ZarrArray):
            raise WriterError(
                "A plain Zarr image is written into an array node, not a "
                "group. Pass a store path to to_store instead."
            )
        if data is not None:
            wrapped[...] = data
        self._write_metadata(wrapped, on_loss)
        return wrapped

    def to_store(self, location: StoreLike, **kwargs) -> None:
        # An opened array is written into as it stands. A location names a
        # store that does not exist yet, so the array is created there.
        # `create` takes the data and describes the array from it; the
        # array-only `create_array` needs a shape and a dtype instead, so it
        # cannot be handed data alone.
        node = _as_node(location)
        if node is not None:
            self.to_node(node, **kwargs)
            return
        on_loss = kwargs.pop("on_loss", None)
        data = self.data
        if data is not None:
            node = create(location, data=data, overwrite=True, **kwargs)
            self._write_metadata(node, on_loss)
