"""The plain Zarr image format."""

# dependencies
import numpy as np
import typing_extensions as tx
from abczarr import ArrayConfig, ZarrArray, ZarrNode
from bagof.magic import Inherit, NoRepr, replace

# core
from brainhops._core.dependencies import da
from brainhops._core.path import Path, replace_directory
from brainhops._core.properties import smartproperty
from brainhops._core.proxy import ArrayProxy
from brainhops._core.typing import ArrayProtocol

# internals
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.metadata import ConversionReport
from brainhops.datamodel.metadata._internal_api import apply_loss_policy
from brainhops.datamodel.transformations import Transformation
from brainhops.errors import (
    UnrepresentableTransformationError,
)
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    Confidence,
    ParserContentError,
    WriterError,
)
from brainhops.io.common._internal_api import (
    held_transformations,
    same_transformations,
)
from brainhops.io.common.zarr import (
    StoreLike,
    ZarrMetadata,
    ZarrRaw,
    ZarrReaderWriter,
)
from brainhops.io.common.zarr._internal_api import (
    ZarrProxy,
    absolute_location,
    as_node,
    opened_node,
    reads_store,
    store_location,
    write_array,
    write_into,
)
from brainhops.io.images.base import ImageFormat
from brainhops.io.metadata._internal_api import (
    metadata_to_write,
    requested_fields,
)

# this format
from ._views import image_to_disk, image_to_model


@register_format
class ZarrImage(ZarrReaderWriter, ImageFormat, SingleScaleImage):
    """An image stored as a plain Zarr array.

    A plain Zarr array is a store, a directory or a remote prefix of
    keys, that holds the document of one array and its chunks. The image
    holds the document as [`ZarrMetadata`][], whose record is a
    [`ZarrRaw`][], and the array as the store keeps it in `rawdata`.
    When the image is read from a store, `rawdata` is a lazy array that
    reads the chunks only when they are needed, and `data` returns that
    lazy array. An OME-Zarr pyramid, a group with a multiscale geometry,
    is read by [`OmeZarrImage`][brainhops.io.images.zarr.OmeZarrImage]
    instead.

    A plain array stores no geometry, so the image decodes no
    transformation from its record. The reader option `transformation`
    places the voxels in the world, as an assigned transformation. A
    writer cannot store a geometry either, so it refuses an image whose
    transformations are not the identity, with
    [`UnrepresentableTransformationError`][brainhops.errors.UnrepresentableTransformationError],
    rather than drop them: such an image is written as OME-Zarr, or its
    transformations are cleared first with `image.transformations = []`.

    `rawdata` and the record describe the store as it was read, or as
    the image was last consolidated. Data that are assigned are pending
    until [`consolidate`][] encodes them into a copy, or until the image
    is written, which writes such a copy and leaves the image as it was.

    An image that is read and written again to a local directory without
    changes gives the same files, byte for byte: the chunk files are
    copied, and each document is written with the bytes that were read
    when the record still describes it. A local directory is written in a
    new directory beside the target, which then replaces the target, so
    that a failed write leaves the old store as it was. A new store is
    written in Zarr version 3, and a store that was read keeps its
    version. A remote store cannot be replaced in one step, so it is
    written in place, and an image is never written over the remote store
    that its lazy array reads.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".zarr",)

    rawdata: NoRepr[tx.Optional[ArrayProtocol]] = None
    """The array as the store keeps it, or `None` without data.

    After a read, `rawdata` is a lazy array of the store, which reads
    its chunks only when it is indexed. `rawdata` describes the store as
    it was read, or as the image was last consolidated, so assigning
    `data` leaves it as it is.
    """

    _data: tx.Optional[ArrayProtocol] = Inherit("_data", default=None)
    """The data given to the image, or `None` to decode it from `rawdata`.

    The image takes over the `data` field of [`SingleScaleImage`][] to
    store it under this private name. The constructor still takes `data`
    in the same place, and the property of the same name reads it. An
    array stored here is pending: it is encoded into `rawdata` only by
    [`consolidate`][] or by a write.
    """

    _transformations: tx.Optional[tx.List[Transformation]] = Inherit(
        "transformations", default=None
    )
    """The transformations assigned to the image, or `None`.

    The image takes over the `transformations` field of
    [`SingleScaleImage`][] to store it under this private name. Only an
    assigned list, even an empty one, is stored here. `None`, the
    default, means that no list was assigned, so the property decodes
    the list of the record, which is empty.
    """

    _metadata: NoRepr[tx.Optional[ZarrMetadata]] = Inherit(
        "metadata", default=None
    )

    metadata = smartproperty("metadata", invalidates="transformations")
    """The metadata of the store, which holds its record, or `None`.

    An image built from data has no metadata. Assigning other metadata
    drops the transformations decoded from the previous record and keeps
    the transformations that were assigned.
    """

    @smartproperty(cache=True, invalidated_by="rawdata")
    def data(self) -> tx.Optional[ArrayProtocol]:
        """The image data, assigned or decoded from `rawdata`.

        An array assigned to the image is returned as it is, and stays
        pending until [`consolidate`][] or a write encodes it.
        Otherwise, the data are the lazy array of the file, which reads
        its values when a computation needs them and refuses an edit in
        place. To change the voxels, assign `data`, and assigning `None`
        gives the lazy array of the file again.
        """
        if self.rawdata is None:
            return None
        return image_to_model(self.rawdata)

    @smartproperty(cache=True)
    def transformations(self) -> tx.List[Transformation]:
        """Voxel-to-world transformations, which a plain array never stores.

        The list decoded from the record is empty, because a plain Zarr
        array has no geometry. It is cached, apart from an assigned list,
        so a change made to it in place is kept. Assigning a list, as the
        reader option `transformation` does, replaces it. The list is
        compared at [`consolidate`][], and by a write, with the empty list
        of the record, and a list that is not the identity is refused.
        """
        return _record_transformations(self.metadata)

    @property
    def shape(self) -> tx.Tuple[int, ...]:
        """The shape of the data, read without reading the chunks.

        The shape is that of the assigned data, or else of `rawdata`,
        which the lazy array knows from the record. An image without
        data has the empty shape.
        """
        if self._data is not None:
            return tuple(int(n) for n in self._data.shape)
        if self.rawdata is not None:
            return tuple(int(n) for n in self.rawdata.shape)
        return ()

    # --- sniff --------------------------------------------------------

    @classmethod
    def _score_store(cls, node: ZarrNode) -> float:
        # A plain array is very likely meant as an image. A group is not an
        # array, and is left to the OME-Zarr reader.
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
    ) -> "ZarrImage":
        """Read an image from an opened Zarr array.

        The record is read from the document of the array, and `rawdata` is a
        lazy array of the node, which reads no chunk until it is indexed.
        Without a `transformation`, which places voxels in the world, the
        image has no geometry.

        Parameters
        ----------
        node : ZarrArray or native Zarr array
            The opened array. A native zarr-python, TensorStore or
            zarrista array is wrapped in an abczarr node first.
        transformation : Transformation, optional
            The voxel-to-world transformation of the image, which is
            assigned to it.
        **kwargs : Any
            Refused, since the reader has no other option.

        Returns
        -------
        ZarrImage
            The image, whose chunks are read lazily.

        Raises
        ------
        TypeError
            If another option is given.
        ParserTypeError
            If `node` is a path, or an object that cannot be wrapped.
        ParserContentError
            If the node is a group: there is no array to read, and reading
            the group would fail later, more obscurely.
        """
        _no_options(kwargs)
        node = opened_node(node)
        if not isinstance(node, ZarrArray):
            raise ParserContentError(
                "This Zarr store is a group, not a plain array, so it cannot "
                "be read as a single-scale image."
            )
        record = ZarrRaw.from_node(node)
        image = cls(
            rawdata=ZarrProxy(node, store_location(node)),
            metadata=ZarrMetadata.from_raw(record),
        )
        if transformation is not None:
            image.transformation = transformation
        return image

    @classmethod
    def from_store(cls, location: StoreLike, **kwargs) -> "ZarrImage":
        """Read an image from a Zarr store location.

        The array is opened as by [`from_node`][], and the lazy array of
        the image knows the location that it was read from, so that the
        image is never written over the store that it still reads.

        Parameters
        ----------
        location : str or PathLike
            The location of the array.
        **kwargs : Any
            `mode` and `driver`, which are passed to `abczarr.open`, and
            the options of [`from_node`][].

        Returns
        -------
        ZarrImage
            The image, whose chunks are read lazily.

        Raises
        ------
        ParserExistsError
            If no Zarr store exists at `location`.
        ParserContentError
            If the store is a group, not a plain array.
        """
        location = absolute_location(location)
        image = super().from_store(location, **kwargs)
        image.rawdata = ZarrProxy(image.rawdata.node, location)
        return image

    # --- consolidate --------------------------------------------------

    def _consolidate(
        self,
        *,
        inplace: bool = False,
        dtype: tx.Optional[tx.Any] = None,
        chunks: tx.Optional[tx.Any] = None,
        shards: tx.Optional[tx.Any] = None,
        compressor: tx.Optional[tx.Any] = None,
        compressor_options: tx.Optional[tx.Mapping[str, tx.Any]] = None,
        filters: tx.Optional[tx.Sequence[tx.Mapping[str, tx.Any]]] = None,
        fill_value: tx.Optional[tx.Any] = None,
        dimension_separator: tx.Optional[str] = None,
        zarr_version: tx.Optional[int] = None,
        **fields: tx.Any,
    ) -> "ZarrImage":
        """Encode the pending data into `rawdata` and the record.

        Consolidating makes `rawdata` and the record describe the image, as a
        store written from it would. The array to store is the assigned
        data, converted to `dtype` when it is given, or else `rawdata`.
        The document of the record is kept as it is when the array has
        its shape and its type and no option is given, so that only the
        chunks change. Otherwise the document is described again by
        abczarr's `ArrayConfig` from the options, in the version of the
        record, or in Zarr version 3 for an image without a record, with
        the attributes of the record, its dimension names and its chunks
        when the array has as many axes, and abczarr's defaults for every
        other option. The record held before is never changed. Nothing is
        pending afterwards.

        A plain array stores no geometry. The transformations that the
        image holds, read or assigned, are compared with the empty list
        of the record, and a list of identities is written as no
        geometry. An image with a record, no pending data, and no
        transformation or only identities, consolidated without options,
        is left as it is: the copy shares `rawdata` and the metadata.

        The names and the values of the options are those of abczarr's
        `ArrayConfig` (abczarr 0.2.2, `abczarr/api/config.py`).

        The metadata is consolidated first (step 2 of the precedence on
        write), with the fields given by keyword. The record of a Zarr
        array holds no field of the vocabulary yet, so a field that is
        set is reported as lost, once, under the error policy of the
        call.

        Parameters
        ----------
        inplace : bool, default=False
            Whether to change this image rather than a copy.
        dtype : dtype, optional
            The data type to store, to which the data are converted.
        chunks : int or sequence of int, optional
            The shape of a chunk, in the order of the store.
        shards : int or sequence of int, optional
            The shape of a shard, which Zarr version 3 alone stores.
        compressor : str or mapping, optional
            The compressor, such as `"zstd"`, or `"none"` for none.
        compressor_options : mapping, optional
            The options of a named compressor.
        filters : sequence of mapping, optional
            The codecs applied before the compressor.
        fill_value : number, optional
            The value that an unwritten element reads as.
        dimension_separator : str, optional
            The separator of the indices of a chunk in its key, `"/"` or
            `"."`.
        zarr_version : int, optional
            The version of the Zarr format to write, 2 or 3.
        **fields : Any
            Fields of the metadata to override.

        Returns
        -------
        ZarrImage
            The consolidated copy, or this image with `inplace=True`.

        Raises
        ------
        TypeError
            If a keyword names no field of the metadata.
        MetadataLossError
            If metadata is lost under the `"raise"` policy.
        WriterError
            If the image has no data.
        UnrepresentableTransformationError
            If a transformation of the image is not the identity.
        """
        report = ConversionReport(source="ZarrImage", target="ZarrImage")
        requested = requested_fields(self.metadata, fields)
        metadata = metadata_to_write(
            self.metadata, ZarrMetadata, fields, report
        )
        apply_loss_policy(report, stacklevel=2)
        if not requested:
            metadata = self.metadata
        options = {
            "dtype": dtype,
            "chunks": chunks,
            "shards": shards,
            "compressor": compressor,
            "compressor_options": compressor_options,
            "filters": filters,
            "fill_value": fill_value,
            "dimension_separator": dimension_separator,
            "zarr_version": zarr_version,
        }
        options = {k: v for k, v in options.items() if v is not None}
        held = held_transformations(self)
        _check_geometry(held)
        kept = held is None or same_transformations(
            held, _record_transformations(self.metadata)
        )
        if (
            self._data is None
            and kept
            and not options
            and self.metadata is not None
            and self.metadata.raw is not None
        ):
            if not inplace:
                return replace(self, transformations=None, metadata=metadata)
            self.transformations = None
            self.metadata = metadata
            return self
        raw, record = _encoded(self, options)
        if metadata is None:
            metadata = ZarrMetadata.from_raw(record)
        else:
            metadata = replace(metadata, raw=record)
        if not inplace:
            return replace(
                self,
                data=None,
                transformations=None,
                rawdata=raw,
                metadata=metadata,
            )
        self.rawdata = raw
        self.metadata = metadata
        self.data = None
        self.transformations = None
        return self

    # --- save ---------------------------------------------------------

    def to_node(self, node: tx.Any, **kwargs) -> ZarrNode:
        """Write the data of the image into an opened Zarr array.

        The image is consolidated into a copy, whose array is written into
        the node as it is. The document of the node is left as it is, so
        the node must have the shape of the image.

        Parameters
        ----------
        node : ZarrArray or native Zarr array
            The opened array to write into.
        **kwargs : Any
            Refused, since the node already decides how it stores its
            chunks.

        Returns
        -------
        ZarrNode
            The node that was written into.

        Raises
        ------
        TypeError
            If an option is given.
        WriterError
            If the node is a group or a path, or if the image has no data.
        UnrepresentableTransformationError
            If a transformation of the image is not the identity.
        """
        _no_options(kwargs)
        wrapped = as_node(node)
        if not isinstance(wrapped, ZarrArray):
            raise WriterError(
                "A plain Zarr image is written into an array node, not a "
                "group. Pass a store path to to_store instead."
            )
        copy = self.consolidate()
        write_into(wrapped, copy.rawdata)
        return wrapped

    def to_store(
        self,
        location: StoreLike,
        driver: tx.Optional[str] = None,
        **kwargs,
    ) -> None:
        """Write the image to a store location, or into an opened array.

        The image is consolidated into a copy with the keyword arguments,
        as by [`consolidate`][], and the copy is written. The image itself
        is not changed. A local directory is filled beside the target and
        then moved into place, so a failed write leaves the old store as
        it was. A lazy array whose chunks are stored as the copy stores
        them is copied file by file, and each document keeps the bytes
        that were read when the record still describes it, so an image
        that did not change is written byte for byte. When the lazy array
        of the image reads the local directory that is written, the image
        is consolidated in place instead, so that it holds what the store
        holds (#426), and its lazy array and its record are read again
        from the new store. A remote location is written in place.

        Parameters
        ----------
        location : str, PathLike or ZarrArray
            The location of the store, or an opened array, which
            [`to_node`][] writes into.
        driver : str, optional
            The abczarr driver that creates the array, or `None` to let
            abczarr choose one.
        **kwargs : Any
            The options of [`consolidate`][].

        Raises
        ------
        TypeError
            If an unknown option is given.
        WriterError
            If the image has no data, or if the location is the remote
            store that the lazy array of the image reads, which cannot be
            written without the lazy array reading a store half written.
        UnrepresentableTransformationError
            If a transformation of the image is not the identity.
        """
        node = as_node(location)
        if node is not None:
            if driver is not None:
                kwargs["driver"] = driver
            self.to_node(node, **kwargs)
            return
        location = absolute_location(location)
        local = Path(location).is_local()
        if not reads_store(self.rawdata, location):
            copy = self.consolidate(**kwargs)
            record, raw = copy.metadata.raw, copy.rawdata
            if local:
                replace_directory(
                    location,
                    lambda name: write_array(name, record, raw, driver),
                )
            else:
                write_array(location, record, raw, driver)
            return
        if not local:
            raise WriterError(
                "This image reads the remote store that it would be written "
                f"to, {location}, and a remote store cannot be replaced in "
                "one step, so the lazy array would read a store half "
                "written. Write the image to another location."
            )
        # The image is written over the directory that its lazy array reads,
        # so it is consolidated in place, and then holds what the store holds
        # (#426). The chunk files of an unchanged array are copied from the
        # old directory while it is still in place, and then the lazy array
        # and the record are read again from the new directory.
        self.consolidate(inplace=True, **kwargs)
        record, raw = self.metadata.raw, self.rawdata
        replace_directory(
            location, lambda name: write_array(name, record, raw, driver)
        )
        options = {} if driver is None else {"driver": driver}
        fresh = type(self).from_store(location, **options)
        self.rawdata = fresh.rawdata
        self.metadata = fresh.metadata
        self.data = None


# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


def _no_options(kwargs: tx.Mapping[str, tx.Any]) -> None:
    """Refuse the options that a reader or a writer does not know.

    Raises
    ------
    TypeError
        If an option is given.
    """
    if kwargs:
        names = ", ".join(sorted(kwargs))
        raise TypeError(f"Unknown options of ZarrImage: {names}")


def _record_transformations(
    metadata: tx.Optional[ZarrMetadata],
) -> tx.List[Transformation]:
    """Decode the transformations of a record, which stores none."""
    return []


# ----------------------------------------------------------------------
#   WRITING
# ----------------------------------------------------------------------


def _check_geometry(
    transformations: tx.Optional[tx.Sequence[Transformation]],
) -> None:
    """Refuse a geometry that a plain Zarr array cannot store.

    Raises
    ------
    UnrepresentableTransformationError
        If a transformation is not the identity.
    """
    for transformation in transformations or ():
        if not transformation.is_identity(compute=True):
            raise UnrepresentableTransformationError(
                "A plain Zarr array stores no geometry, so the "
                "transformations of this image cannot be written. Write a "
                "pyramid of one level with OmeZarrImage, which stores them, "
                "or assign `image.transformations = []` to write the data "
                "alone."
            )


def _encoded(
    image: ZarrImage, options: tx.Mapping[str, tx.Any]
) -> tx.Tuple[ArrayProtocol, ZarrRaw]:
    """Encode the pending data of an image into `rawdata` and a new record.

    The new record holds the document of the held record when it still
    describes the array to store, and the bytes that a writer writes for
    each document.

    Raises
    ------
    WriterError
        If the image has no data.
    """
    if image._data is not None:
        raw = image_to_disk(image._data)
    else:
        raw = image.rawdata
    if raw is None:
        raise WriterError("A Zarr image without data cannot be written.")
    options = dict(options)
    dtype = options.pop("dtype", None)
    if dtype is not None and np.dtype(dtype) != np.dtype(raw.dtype):
        raw = _converted(raw, np.dtype(dtype))
    held = image.metadata.raw if image.metadata is not None else None
    document = _document(held, raw, options)
    base = () if held is None else held.stored
    record = ZarrRaw(document=document, stored=base)
    return raw, replace(record, stored=tuple(record.documents().items()))


def _converted(raw: ArrayProtocol, dtype: np.dtype) -> ArrayProtocol:
    """Convert an array to the type that a store keeps, as NumPy casts.

    A lazy array of a file becomes a Dask array when Dask is installed,
    so that it is converted block by block as it is written.
    """
    if da is not None and isinstance(raw, ArrayProxy):
        raw = raw.to_dask()
    if da is not None and isinstance(raw, da.Array):
        return raw.astype(dtype)
    return np.asarray(raw).astype(dtype)


def _document(
    held: tx.Optional[ZarrRaw],
    raw: ArrayProtocol,
    options: tx.Mapping[str, tx.Any],
) -> tx.Any:
    """Return the document of the array to store.

    The document of the held record is returned when the array has its
    shape and its type and no option is given. Otherwise abczarr describes
    the array from the options, in the version of the held record or in
    version 3, with the attributes, the dimension names and the chunks of
    the held record when the array has as many axes.
    """
    shape = tuple(int(n) for n in raw.shape)
    dtype = np.dtype(raw.dtype)
    if (
        held is not None
        and not options
        and held.shape == shape
        and held.dtype == dtype
    ):
        return held.document
    config = {"shape": shape, "dtype": dtype, "zarr_version": 3}
    if held is not None:
        config["zarr_version"] = held.zarr_version
        config["attributes"] = held.attributes
        names = held.dimension_names
        if names is not None and len(names) == len(shape):
            config["dimension_names"] = names
        chunks = held.chunks
        if chunks is not None and len(chunks) == len(shape):
            config["chunks"] = chunks
    config.update(options)
    return ArrayConfig(**config).to_metadata()
