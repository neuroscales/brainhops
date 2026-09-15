# dependencies
import abczarr
import typing_extensions as tx
from abczarr.abc.sync import ZarrNode

# internals
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
from brainhops.io.base.zarr import ZarrParser
from brainhops.io.images.base import WritableFileBasedImage


@register_format
class ZarrImage(ZarrParser, WritableFileBasedImage, SingleScaleImage):
    """An image that is encoded by a plain Zarr array.

    A plain Zarr array carries no world geometry, so the image is read with
    an identity geometry unless a voxel-to-world transformation is supplied
    through the `transformation` argument. An OME-Zarr pyramid, whose group
    carries a multiscale geometry, is read by
    [OmeZarrImage][brainhops.io.images.zarr.OmeZarrImage] instead.

    The array is held as a handle and its data is read on first access, so
    opening the image does not read the array.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".zarr",)

    @property
    def data(self) -> tx.Optional[ArrayProtocol]:
        cached = getattr(self, "_data", None)
        if cached is not None:
            return cached
        node = getattr(self, "_node", None)
        if node is not None:
            self._data = get_array_backend().asarray(node[...])
            return self._data
        return None

    @data.setter
    def data(self, value: tx.Optional[ArrayProtocol]) -> None:
        self._data = value

    @classmethod
    def _score_store(cls, node: ZarrNode) -> float:
        # A plain array is very likely wanted as an image. A group is not an
        # array, so it is left to the OME reader.
        if isinstance(node, abczarr.ZarrArray):
            return Confidence.LIKELY
        return Confidence.NO

    @classmethod
    def _read_node(
        cls,
        node: ZarrNode,
        transformation: tx.Optional[Transformation] = None,
        **kwargs,
    ) -> tx.Self:
        if not isinstance(node, abczarr.ZarrArray):
            raise ParserContentError(
                "This Zarr store is a group, not a plain array, so it "
                "cannot be read as a single-scale image."
            )
        transformations = (
            [transformation] if transformation is not None else []
        )
        image = cls(transformations=transformations)
        image._node = node
        return image

    def _write_node(self, node: ZarrNode, **kwargs) -> None:
        data = self.data
        if data is None:
            raise WriterError(
                "This image has no data, so there is nothing to write."
            )
        if not isinstance(node, abczarr.ZarrArray):
            raise WriterError(
                "A plain Zarr image is written into an array node, not a "
                "group. Pass a store path to to_store instead."
            )
        node[...] = data

    def _create_store(
        self,
        location: str,
        chunks: tx.Any = None,
        dtype: tx.Any = None,
        **kwargs,
    ) -> None:
        data = self.data
        if data is None:
            raise WriterError(
                "This image has no data, so there is nothing to write."
            )
        options = dict(kwargs)
        if chunks is not None:
            options["chunks"] = chunks
        if dtype is not None:
            options["dtype"] = dtype
        abczarr.create(location, data=data, overwrite=True, **options)
