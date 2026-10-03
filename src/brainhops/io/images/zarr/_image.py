# stdlib
import copy

# dependencies
import typing_extensions as tx
from abczarr import ZarrArray, ZarrNode, create

# internals
from brainhops._core.dependencies import da
from brainhops._core.properties import smartproperty
from brainhops._core.typing import ArrayProtocol

# backends
from brainhops.backends import get_array_backend
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.metadata import (
    ConversionReport,
    FormatMetadata,
    apply_loss_policy,
    metadata_annotation,
)
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

from ._metadata import ZarrMetadata


def metadata_field(cls: tx.Type[FormatMetadata], doc: str) -> tx.Any:
    """The annotation of the narrowed `metadata` field of a Zarr class."""
    return metadata_annotation(cls, doc, default=cls)


def sync_record(
    cls: tx.Type[FormatMetadata],
    metadata: tx.Optional[FormatMetadata],
    node: tx.Any,
    read: tx.Callable[[], tx.Any],
    image: tx.Any = None,
) -> FormatMetadata:
    """
    The metadata of an object read from `node`.

    A Zarr record is rebuilt from the attributes of the node on each
    read, so the metadata remembers the node its record was read from:
    a metadata read from `node` already (as `replace()` carries it) is
    kept as it is. Otherwise the record is read (`read()`) and decoded,
    and the fields that changed in `metadata` (all of them for one
    built in memory) are set over the decoded ones, as changes (see
    `FormatMetadata.with_record`).
    """
    if (
        metadata is not None
        and metadata.raw is not None
        and metadata.__dict__.get("_source") is node
    ):
        return metadata
    if metadata is None:
        metadata = cls()
    synced = metadata.with_record(read(), image=image)
    synced.__dict__["_source"] = node
    return synced


def node_attributes(node: tx.Any) -> tx.Dict[str, tx.Any]:
    """The attributes of a node, as plain JSON."""
    try:
        attrs = node.attrs
        return {key: attrs[key] for key in attrs}
    except Exception:
        return {}


def write_attributes(
    node: tx.Any,
    attrs: tx.Mapping[str, tx.Any],
    before: tx.Optional[tx.Mapping[str, tx.Any]] = None,
) -> None:
    """Write `attrs` onto a node, and remove the keys of `before` (the
    record that was read) that are no longer in it."""
    current = node_attributes(node)
    for key in before or {}:
        if key not in attrs and key in current:
            del node.attrs[key]
    for key, value in attrs.items():
        if current.get(key) != value:
            node.attrs[key] = value


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

    metadata: metadata_field(
        ZarrMetadata,
        """
        The metadata of the array: the vocabulary, stored as a sidecar
        under its attribute `"brainhops"`, and its other attributes as
        `extra`; the attributes are the record (`metadata.raw`). See
        [`ZarrMetadata`][brainhops.io.images.zarr.ZarrMetadata].
        """,
    )

    def __post_init__(self) -> None:
        parent = getattr(super(), "__post_init__", None)
        if parent is not None:
            parent()
        self._sync_metadata()

    def _sync_metadata(self) -> None:
        """Read the metadata from the attributes of the array, when it
        was not read from this array yet."""
        node = self.node
        if node is None:
            return
        self.metadata = sync_record(
            ZarrMetadata,
            self.metadata,
            node,
            lambda: node_attributes(node),
            self,
        )

    def _write_metadata(self, node: tx.Any, on_loss: tx.Any) -> None:
        metadata = self.metadata
        if metadata is None:
            return
        if not isinstance(metadata, ZarrMetadata):
            metadata = ZarrMetadata.from_other(metadata)
        before = metadata.raw
        report = ConversionReport(source=metadata.format, target="zarr")
        record = metadata.write_raw(
            copy.deepcopy(before) if before is not None else {},
            image=self,
            report=report,
        )
        apply_loss_policy(report, on_loss, stacklevel=4)
        write_attributes(node, record, before)

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
