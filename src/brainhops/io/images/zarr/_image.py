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

    @smartproperty
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
        wrapped = _as_node(node)
        data = self.data
        if not isinstance(wrapped, ZarrArray):
            raise WriterError(
                "A plain Zarr image is written into an array node, not a "
                "group. Pass a store path to to_store instead."
            )
        if data is not None:
            wrapped[...] = data
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
        data = self.data
        if data is not None:
            create(location, data=data, overwrite=True, **kwargs)
