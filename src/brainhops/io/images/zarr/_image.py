import typing_extensions as tx
from abczarr import ZarrArray, ZarrNode, create

from brainhops._core.dependencies import da
from brainhops._core.properties import smartproperty
from brainhops._core.typing import ArrayProtocol
from brainhops.backends import get_array_backend
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.transformations import Transformation
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    Confidence,
    ParserContentError,
    WriterError,
)
from brainhops.io.common.zarr import (
    StoreLike,
    ZarrParserWriter,
    _as_node,
)
from brainhops.io.images.base import WritableFileBasedImage


@register_format
class ZarrImage(ZarrParserWriter, WritableFileBasedImage, SingleScaleImage):
    """An image stored as a plain Zarr array.

    A plain array has no world geometry, so the image is read with an identity
    geometry unless a voxel-to-world `transformation` is given. An OME-Zarr
    pyramid, a group with multiscale geometry, is read by
    [`OmeZarrImage`][brainhops.io.images.zarr.OmeZarrImage] instead. Opening
    the image does not read the array: the data is read on first access.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".zarr",)

    @smartproperty(cache=True)
    def data(self) -> tx.Optional[ArrayProtocol]:
        node = self.node
        if node is not None:
            backend = get_array_backend()
            if backend is da:
                return node.to_dask(chunks="chunks")
            return backend.asarray(node[...])
        return None

    @classmethod
    def _score_store(cls, node: ZarrNode) -> float:
        # A plain array is very likely meant as an image. A group is not an
        # array, and is left to the OME-Zarr reader.
        if isinstance(node, ZarrArray):
            return Confidence.LIKELY
        return Confidence.NO

    @classmethod
    def from_node(
        cls,
        node: tx.Any,
        transformation: tx.Optional[Transformation] = None,
        **kwargs,
    ) -> tx.Self:
        """Read an image from an open Zarr array.

        Without a `transformation`, which places voxels in the world, the image
        has an identity geometry.

        Raises
        ------
        ParserContentError
            If the node is a group: there is no array to read, and reading the
            group would fail later, more obscurely.
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
        # An open array is written into as is, and a location that names no
        # existing store gets a new array. `create` describes the array from
        # its data, whereas `create_array` would need a shape and dtype.
        node = _as_node(location)
        if node is not None:
            self.to_node(node, **kwargs)
            return
        data = self.data
        if data is not None:
            create(location, data=data, overwrite=True, **kwargs)
