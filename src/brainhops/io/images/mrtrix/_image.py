"""The MRtrix image format."""

# stdlib
import math
import os
from io import BytesIO

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import Inherit, NoRepr, replace

# internals
from brainhops._core import path
from brainhops._core.properties import smartproperty
from brainhops._core.typing import ArrayProtocol
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.orientations import Orientation
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import (
    Affine,
    Identity,
    Scaling,
    Transformation,
)
from brainhops.datamodel.units import is_physicalunit, is_timeunit
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    BinaryFileReader,
    BinaryFileWriter,
    Confidence,
    UnrepresentableTransformationError,
    WriterError,
)
from brainhops.io.common._geometry import (
    Arrangement,
    declared_axes,
    held_transformations,
    same_transformations,
)
from brainhops.io.common.mrtrix import MrtrixMetadata, MrtrixRaw
from brainhops.io.common.mrtrix._codecs import dtype_to_mrtrix
from brainhops.io.common.mrtrix._data import _MrtrixProxy, read_mrtrix
from brainhops.io.common.mrtrix._files import (
    file_entry,
    keeps_storage,
    merged_keyval,
    mrtrix_bytes,
    placed_record,
    reads_from,
    stored_voxels,
    stores_as_recorded,
    write_mrtrix,
    writer_options,
    writer_strides,
    written_files,
)
from brainhops.io.common.mrtrix._geometry import (
    split_voxel_to_scanner,
    voxel_to_ras,
)
from brainhops.io.common.mrtrix._raw import (
    _declined,
    _no_options,
    _sniffed_raw,
)
from brainhops.io.images.base import ImageFormat

# this format
from ._views import _image_to_disk, _image_to_model

_INDEX = "index"
_MM = "millimeter"
_SPATIAL = ("x", "y", "z")
_ORIENTATION = {
    "x": "left-to-right",
    "y": "posterior-to-anterior",
    "z": "inferior-to-superior",
}
_SCANNER = "scanner"
"""Name of the world space into which the MRtrix transform maps."""


def _mrtrix_axes(ndim: int) -> tx.List[Axis]:
    """Return the voxel axes of an MRtrix image.

    The first three axes are the spatial axes `x`, `y` and `z`. Further
    axes, such as diffusion volumes, time or spherical harmonic
    coefficients, have no meaning defined by MRtrix, so they are named
    `dim3`, `dim4` and so on, without a type.
    """
    axes = []
    for i in range(ndim):
        if i < 3:
            axes.append(Axis(_SPATIAL[i], "space", unit=_INDEX))
        else:
            axes.append(Axis(f"dim{i}", unit=_INDEX))
    return axes


@register_format
class MrtrixImage(
    ImageFormat, SingleScaleImage, BinaryFileReader, BinaryFileWriter
):
    """An image stored in an MRtrix file (`.mif`, `.mif.gz` or `.mih`).

    An MRtrix image holds the header of its file as [`MrtrixMetadata`][],
    whose record is an [`MrtrixRaw`][], and the voxels as stored in `rawdata`.
    When the image is read from a file, `rawdata` is a lazy array that
    reads the voxels only when they are needed, and `data` is decoded
    from it on first access. The data are indexed `[x, y, z, ...]` in
    Fortran order, whatever `layout` the file uses, and the intensity
    `scaling` of the header is applied to them.

    The transformations are decoded from the record on first access,
    unless other transformations are assigned. They are a [`Scaling`][]
    to `"physical"` by the voxel sizes and the preferred [`Affine`][] to
    scanner RAS+ in millimeters, `transform @ diag(vox)`.

    `rawdata` and the record describe the file as it was read, or as the image
    was last consolidated. Data that are assigned are pending until
    [`consolidate`][] encodes them into a copy, or until the image is
    written, which writes such a copy and leaves the image as it was. The
    transformations, read or assigned, are compared at [`consolidate`][]
    with those that the record decodes to, and the geometry of the record
    is kept when they are the same.

    When the image is consolidated, the record of its metadata is the base
    of the new header, as [`consolidate`][] describes, so header keys that
    the data model has no place for, such as `dw_scheme`, are kept. An
    image that is read and written again without changes gives the same
    bytes, after decompression for a `.mif.gz` file. A `.mih` header names
    its data file, so a copy under another name differs in that line only.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".mif", ".mif.gz", ".mih")
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("mrtrix",)

    rawdata: NoRepr[tx.Optional[ArrayProtocol]] = None
    """The voxels as the file stores them, or `None` without data.

    After a read, `rawdata` is a lazy array, which presents the voxels in the
    axis order of the header and applies its intensity scaling when it is
    read. `rawdata` describes the file as it was read, or as the image
    was last consolidated, so assigning `data` leaves it as it is.
    """

    _data: tx.Optional[ArrayProtocol] = Inherit("data", default=None)
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
    [`SingleScaleImage`][] to store it under this private name. The
    constructor still takes `transformations` in the same place, and the
    property of the same name reads it. Only an assigned list, even an
    empty one, is stored here. `None`, the default, means that no list
    was assigned, so the property decodes the list of the record and
    caches it apart, and assigning other metadata drops that cache.
    """

    _metadata: NoRepr[tx.Optional[MrtrixMetadata]] = None

    metadata = smartproperty("metadata", invalidates="transformations")
    """The metadata of the file, which holds its record, or `None`.

    An image built from data has no metadata. Assigning
    other metadata drops the transformations decoded from the previous
    record, so that they are decoded from the new one, and keeps the
    transformations that were assigned.
    """

    @smartproperty(cache=True)
    def data(self) -> tx.Optional[ArrayProtocol]:
        """The image data, assigned or decoded from `rawdata`.

        An array assigned to the image is returned as it is, and stays
        pending until [`consolidate`][] or a write encodes it. Otherwise
        the data is decoded from `rawdata` on first access and cached.
        Data decoded from a file is read-only, because changing it in
        place would not change `rawdata`, which is what the image
        writes. To change the voxels, assign `data`. Assigning `None`
        decodes `rawdata` again.
        """
        if self.rawdata is None:
            return None
        return _image_to_model(self.rawdata)

    @smartproperty(cache=True)
    def transformations(self) -> tx.List[Transformation]:
        """Voxel-to-world transformations, decoded from the record.

        The list is decoded on first access and cached, apart from an
        assigned list, so `img.transformation is img.transformation`,
        and a change made to the list in place, such as an appended
        transformation, is kept.
        Assigning other metadata drops the decoded list, so that it is
        decoded from the new record, and keeps an assigned list. A change
        made in place to a decoded list is therefore lost when other
        metadata is assigned. An image without metadata has no
        transformations.

        Assigning a list replaces the stored one. In both cases, the list
        is compared at [`consolidate`][], and by a write, with the list
        that the record decodes to, exactly. An empty list means that the
        image has no geometry, and the image is then written as an image
        built from data alone. Assigning `None` decodes the record again.
        """
        return _record_transformations(self.metadata)

    @property
    def shape(self) -> tx.Tuple[int, ...]:
        """The shape of the data, read without reading the voxels.

        The shape is that of the assigned data, or else of `rawdata`: the data
        is `rawdata` as the lazy array presents it, so the two have the
        same shape, and the lazy array knows its shape from the record.
        An image without data raises as [`SingleScaleImage`][] does.
        """
        if self._data is not None:
            return tuple(int(d) for d in self._data.shape)
        if self.rawdata is None:
            return self.data.shape
        return tuple(int(d) for d in self.rawdata.shape)

    @property
    def system(self) -> tx.Optional[CoordinateSystem]:
        """The voxel coordinate system described by the record.

        The axes are `x`, `y` and `z`, then `dim3`, `dim4` and so on, in
        Fortran order. An image without metadata has no system.
        """
        if self.metadata is None or self.metadata.raw is None:
            return None
        axes = _mrtrix_axes(self.metadata.raw.ndim)
        return CoordinateSystem(name="voxel", axes=axes, order="F")

    # --- reading ------------------------------------------------------

    @classmethod
    def _score_header(cls, header: MrtrixRaw) -> float:
        """Score a header as a plain image.

        A 4D image with three volumes may be a warp, as may a NIfTI file
        of that shape, so it is only a weak match. So is a 5D image of
        shape `(X, Y, Z, 3, 4)`, the warps that `mrregister -nl_warp_full`
        writes.
        """
        if header.ndim == 4 and header.dim[3] == 3:
            return Confidence.WEAK
        if header.ndim == 5 and header.dim[3:] == (3, 4):
            return Confidence.WEAK
        return Confidence.LIKELY

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that a stream holds an MRtrix image.

        The stream may be gzipped, and its position is restored.

        Parameters
        ----------
        file : IO
            The stream, which may be gzipped.
        error : bool or type[Exception], default=False
            Whether to raise an error when the stream holds no MRtrix
            image.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        raw, failure = _sniffed_raw(file)
        if raw is None:
            return _declined(error, failure)
        return cls._score_header(raw)

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that bytes hold an MRtrix image.

        Parameters
        ----------
        content : bytes
            The content of a file, or its first bytes.
        error : bool or type[Exception], default=False
            Whether to raise an error when the bytes hold no MRtrix
            image.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return cls.sniff_fileobj(BytesIO(bytes(content)), error=error)

    @classmethod
    def from_filename(
        cls, filename: path.FilenameLike, mmap: bool = True, **kwargs
    ) -> "MrtrixImage":
        """Read an image from the path of a `.mif`, `.mif.gz` or `.mih` file.

        The record is read from the header, and the lazy array reads the
        voxels when they are needed.

        Parameters
        ----------
        filename : path
            The file of the image, or of its header.
        mmap : bool, default=True
            Whether the lazy array memory-maps local data that is not
            compressed.
        **kwargs : Any
            Refused, since MRtrix has no other option.

        Raises
        ------
        TypeError
            If another option is given.
        ParserExistsError
            If the file or the data file that its header names does not
            exist.
        ParserContentError
            If the file is not an MRtrix image, or its data is too short.
        """
        _no_options(kwargs)
        raw, proxy = read_mrtrix(filename, mmap=mmap)
        return cls(rawdata=proxy, metadata=MrtrixMetadata.from_raw(raw))

    @classmethod
    def from_fileobj(
        cls, file: tx.BinaryIO, mmap: bool = False, **kwargs
    ) -> "MrtrixImage":
        """Read an image from an open MRtrix file, gzipped or not.

        The record is read at once, and the lazy array reads the voxels
        from the stream when they are needed, so the stream must stay open
        while the data may be read. A separate data file is found from the
        name of the stream. The position of the stream is restored.

        Parameters
        ----------
        file : file object
            The stream of the file, open in binary mode.
        mmap : bool, default=False
            Accepted for the readers of paths, and without effect: the
            voxels of a stream are never memory-mapped.
        **kwargs : Any
            Refused, since MRtrix has no other option.

        Parameters
        ----------
        file : BinaryIO
            The stream, which may be gzipped.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        MrtrixImage
            The image, whose voxels are read lazily.

        Raises
        ------
        TypeError
            If another option is given.
        ParserContentError
            If the stream is not an MRtrix image, or its header names a
            separate data file and the stream has no name.
        """
        _no_options(kwargs)
        raw, proxy = read_mrtrix(file, mmap=False)
        return cls(rawdata=proxy, metadata=MrtrixMetadata.from_raw(raw))

    # --- writing ------------------------------------------------------

    def consolidate(
        self,
        *,
        inplace: bool = False,
        layout: tx.Optional[tx.Union[str, tx.Sequence[int]]] = None,
        datatype: tx.Optional[tx.Any] = None,
        scaling: tx.Optional[tx.Sequence[float]] = None,
        keyval: tx.Optional[tx.Mapping[str, tx.Optional[str]]] = None,
    ) -> "MrtrixImage":
        """Encode the pending data and geometry into `rawdata` and the record.

        Consolidating makes `rawdata` and the record describe the image, as a
        file written from it would. The record is encoded in a fixed order.
        The base is the record of the metadata, or a new header for an
        image without metadata.

        1. The keys that describe how the voxels are stored (`dim`,
           `layout`, `datatype` and `scaling`) follow `rawdata`. A lazy array
           that the options do not change is copied as the file stores
           it, so these keys are those of its own record. Any other array
           is stored with the options, or else with the layout of the base
           when it has as many axes, the type of the data and no scaling.
           Spatial axes are stored first, and a slice with further axes
           gets a `z` axis of size one, because MRtrix reads the first
           three axes as spatial.
        2. The geometry comes from the preferred transformation. The `vox`
           and `transform` of the base are kept as they are stored when
           the transformation and the shape are those of the base,
           exactly. Otherwise, the transformation is split into unit
           direction cosines (`transform`) and voxel sizes (`vox`). The
           voxel sizes of the axes after the spatial ones come from the
           transformation, with time in seconds, or else from the base,
           from a [`Scaling`][], or are one.
        3. The other keys of the base are kept, and `keyval` is merged
           into them last.

        The `file` entry of the base is kept, and a writer sets the entry
        of the file that it writes. When this encoding gives the header of
        the base, the metadata is kept as it is, so that an image that was
        read and not changed writes the header that it read. Otherwise,
        the new record is held by new metadata, and the record held before
        is never changed. The voxels to write become `rawdata`, and nothing is
        pending afterwards, so the transformations are decoded from the
        new record, with the precision of the file. The transformations
        that the image holds, read or assigned, are compared with those
        that the record decodes to, exactly, and an image with a record
        that stores its voxels as they are stored, no pending data, and
        the transformations of its record or none read, consolidated
        without options, is left as it is: the copy shares `rawdata` and
        the metadata.

        Parameters
        ----------
        inplace : bool, default=False
            Whether to change this image rather than a copy.
        layout : str or sequence of int, optional
            Strides, in the MRtrix spelling (`"-0,-1,+2"`) or as signed
            one-based integers.
        datatype : str or dtype, optional
            MRtrix type (`"Float32LE"`, `"UInt16BE"`, `"Bit"`) or NumPy
            type.
        scaling : (offset, scale), optional
            Intensity scaling to store.
        keyval : mapping, optional
            Header keys merged into those of the base. A value of `None`
            removes a key. The keys that the other options, the data and
            the geometry decide are refused.

        Returns
        -------
        MrtrixImage
            The consolidated copy, or this image with `inplace=True`.

        Raises
        ------
        WriterError
            If there is no data, the data are zero-dimensional, the type
            cannot be stored, or `keyval` sets a key that the other
            options, the data or the geometry decide.
        UnrepresentableTransformationError
            If the transformation has no affine form, or shifts or reverses
            an axis after the spatial ones.
        """
        options = dict(
            layout=layout, datatype=datatype, scaling=scaling, keyval=keyval
        )
        # The record is kept when the transformations were never read, or
        # are the same as those that it decodes to.
        held = held_transformations(self)
        kept = held is None or same_transformations(
            held, _record_transformations(self.metadata)
        )
        if (
            self._data is None
            and kept
            and all(value is None for value in options.values())
            and self.metadata is not None
            and self.metadata.raw is not None
            and stores_as_recorded(self.rawdata, self.metadata.raw)
        ):
            if not inplace:
                return replace(self, transformations=None)
            self.transformations = None
            return self
        record, raw = _encoded(self, None, options)
        if self.metadata is not None and record is self.metadata.raw:
            metadata = self.metadata
        elif self.metadata is None:
            metadata = MrtrixMetadata.from_raw(record)
        else:
            metadata = replace(self.metadata, raw=record)
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

    def to_bytes(self, **kwargs) -> bytes:
        """Return the bytes of a single-file, uncompressed `.mif` image.

        The image is consolidated into a copy with the keyword arguments,
        as by [`consolidate`][], and the header of the copy, padded up to
        the first voxel, is written with its `rawdata`. A lazy array
        that reaches the writer unchanged is written as the file stores
        it. The image itself is not changed.

        Parameters
        ----------
        **kwargs : Any
            The options of [`consolidate`][].

        Returns
        -------
        bytes
            The content of the `.mif` file.

        Raises
        ------
        TypeError
            If an unknown option is given.
        """
        image = self.consolidate(**writer_options(kwargs))
        return mrtrix_bytes(image.metadata.raw, image.rawdata)

    def to_fileobj(self, file: tx.IO, **kwargs) -> None:
        """Write a single-file, uncompressed `.mif` image to a stream.

        Parameters
        ----------
        file : IO
            The stream.
        **kwargs : Any
            The options of [`consolidate`][].

        Raises
        ------
        TypeError
            If an unknown option is given.
        """
        file.write(self.to_bytes(**kwargs))

    def to_filename(self, filename: path.FilenameLike, **kwargs) -> None:
        """Write to a path, in the variant that its extension names.

        A `.mif` file holds the header and the data, gzipped for `.mif.gz`.
        A `.mih` header names its data, written next to it in a `.dat`
        file. The image is consolidated into a copy with the keyword
        arguments, as by [`consolidate`][], which is written, and the image
        itself is not changed. The content is encoded before any file is
        written, and each local file is replaced at once, so an image can
        be written over the files that it was read from. When the lazy
        array of the image reads a file that is written, the image is
        consolidated in place instead, so that it holds what the files
        hold (#426), and its lazy array reads the new files.

        Parameters
        ----------
        filename : path-like
            The path of the `.mif`, `.mif.gz` or `.mih` file.
        **kwargs : Any
            The options of [`consolidate`][].

        Raises
        ------
        TypeError
            If an unknown option is given.
        """
        options = writer_options(kwargs)
        reading = reads_from(self.rawdata, written_files(filename))
        if reading:
            mmap = self.rawdata.mmap
            image = self.consolidate(inplace=True, **options)
        else:
            image = self.consolidate(**options)
        write_mrtrix(filename, image.metadata.raw, image.rawdata)
        if reading:
            # The image holds what the files hold, and its lazy array reads
            # the new files, since the lazy array of the old files describes
            # how they stored the voxels.
            self.rawdata = read_mrtrix(filename, mmap=mmap)[1]
            self.data = None

    def to_file(self, file: path.FileLike, **kwargs) -> None:
        """Write to a path or a stream.

        Parameters
        ----------
        file : path-like or IO
            The path or the stream.
        **kwargs : Any
            The options of [`consolidate`][].
        """
        if isinstance(file, str):
            file = path.Path(file)
        if isinstance(file, (path.PathLike, os.PathLike)):
            return self.to_filename(file, **kwargs)
        return super().to_file(file, **kwargs)


# ----------------------------------------------------------------------
#   WRITING
# ----------------------------------------------------------------------


def _encoded(
    image: MrtrixImage,
    file: tx.Optional[str],
    options: tx.Mapping[str, tx.Any],
) -> tx.Tuple[MrtrixRaw, ArrayProtocol]:
    """Return the record and the voxels that consolidating an image stores.

    The record is encoded as [`MrtrixImage.consolidate`][] describes, and
    the held record is returned when the encoding changes nothing. The
    voxels are `rawdata` when it is a lazy array that is copied as the
    file stores it, and otherwise the array of the assigned data or of
    the data, in the axis order of the header, read back as the file
    stores them with `stored_voxels`. The `file` argument names the
    data file of the header: `"."` for a single-file image, a file name
    for a `.mih` header, or `None` to keep the entry of the base.
    """
    if image._data is None and image.rawdata is None:
        raise WriterError(
            "This image has no data, so there is nothing to write."
        )
    layout, datatype = options["layout"], options["datatype"]
    scaling = options["scaling"]
    held = None if image.metadata is None else image.metadata.raw
    merged = merged_keyval(held, options["keyval"])
    if image._data is not None:
        raw = _image_to_disk(image._data)
    else:
        raw = image.rawdata
    # The list of the image is read without being stored, so that a write
    # leaves the image as it was.
    transformations = held_transformations(image)
    if transformations is None:
        transformations = _record_transformations(image.metadata)
    xform = transformations[-1] if transformations else Identity()
    ndim = len(raw.shape)
    if ndim == 0:
        raise WriterError("MRtrix cannot store a zero-dimensional array.")
    voxel_axes = declared_axes(getattr(xform, "input", None), ndim)
    arranged = voxel_to_ras(xform, voxel_axes)
    moved = arranged.layout is not None and not arranged.layout.trivial

    if (
        isinstance(raw, _MrtrixProxy)
        and not moved
        and keeps_storage(raw.record, layout, datatype, scaling)
    ):
        source = raw
        shape, strides = raw.record.dim, raw.record.layout
        stored_type, stored_scaling = raw.record.datatype, raw.record.scaling
    else:
        # A lazy array is read here rather than through `data`, so that no
        # cached view of the old file outlives a write over that file.
        source = np.asarray(raw) if isinstance(raw, _MrtrixProxy) else raw
        if moved:
            source = arranged.layout.apply(source)
        shape = tuple(int(d) for d in np.shape(source))
        ndim = len(shape)
        strides = writer_strides(layout, held, ndim)
        if datatype is None:
            datatype = getattr(source, "dtype", np.float32)
        stored_type = dtype_to_mrtrix(datatype)
        stored_scaling = None if scaling is None else tuple(scaling)

    # The geometry of the base is kept as stored when it is unchanged.
    if (
        held is not None
        and tuple(held.dim) == tuple(shape)
        and np.array_equal(held.voxel_to_scanner(), arranged.matrix)
    ):
        vox, transform = held.vox, held.transform
    else:
        transform, spatial_vox = split_voxel_to_scanner(arranged.matrix)
        vox = list(spatial_vox[: min(3, ndim)])
        extra = _mapped_vox(arranged, ndim)
        if extra is None:
            extra = _extra_vox(transformations, held, ndim)
        vox += extra[len(vox) :]

    record = MrtrixRaw(
        dim=shape,
        vox=vox,
        layout=strides,
        datatype=stored_type,
        transform=transform,
        scaling=stored_scaling,
        file=file_entry(held, file),
        keyval=merged,
    )
    record = placed_record(record, held)
    return record, stored_voxels(record, source)


def _mapped_vox(
    arranged: Arrangement, ndim: int
) -> tx.Optional[tx.List[float]]:
    """Return the voxel sizes that the transformation gives non-spatial axes.

    The result is `None` unless every axis after the spatial ones is
    mapped. A time axis with a time unit in the world is given in seconds.

    Raises
    ------
    UnrepresentableTransformationError
        If the map shifts or reverses one of these axes, which MRtrix
        cannot store.
    """
    others = list(arranged.others)
    if ndim <= 3 or len(others) != ndim - 3:
        return None
    world = list(getattr(arranged.world, "axes", None) or [])
    vox = [math.nan] * 3
    for k, (scale, offset) in enumerate(others):
        if offset != 0:
            raise UnrepresentableTransformationError(
                f"MRtrix stores no origin for the axes after the spatial "
                f"ones, so a map that shifts axis {3 + k} ({offset}) cannot "
                f"be written."
            )
        if scale < 0:
            raise UnrepresentableTransformationError(
                f"MRtrix stores the voxel size of axis {3 + k} as a "
                f"positive number, so a map that reverses it ({scale}) "
                f"cannot be written."
            )
        unit = None
        if len(world) > 3 + k:
            unit = getattr(world[3 + k], "unit", None)
        if is_physicalunit(unit) and is_timeunit(unit):
            scale = scale * float(unit.scale)
        vox.append(float(scale))
    return vox


def _extra_vox(
    transformations: tx.Sequence[Transformation],
    source: tx.Optional[MrtrixRaw],
    ndim: int,
) -> tx.List[float]:
    """Return a voxel size for every axis, used beyond the third.

    The sizes come from the source header when it has as many axes, so
    that a NaN volume size survives a round trip. Otherwise they come from
    a [`Scaling`][] with one scale per axis, or default to one.
    """
    if source is not None and source.ndim == ndim:
        vox = list(source.vox[:ndim])
        return vox + [math.nan] * (ndim - len(vox))
    for xform in reversed(list(transformations or [])):
        if isinstance(xform, Scaling):
            scale = np.ravel(np.asarray(getattr(xform, "scale", []), float))
            if scale.size == ndim:
                return [float(v) for v in scale]
    return [1.0] * ndim


# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


def _record_transformations(
    metadata: tx.Optional[MrtrixMetadata],
) -> tx.List[Transformation]:
    """Decode the transformations of the record of some metadata.

    Metadata without a record gives no transformation.
    """
    if metadata is None or metadata.raw is None:
        return []
    return _mrtrix_to_transformations(metadata.raw)


def _mrtrix_to_transformations(
    header: MrtrixRaw,
) -> tx.List[Transformation]:
    """Convert an MRtrix header to transformations, in this order.

    1. voxel -> "physical": a `Scaling` by the voxel sizes;
    2. voxel -> "scanner": the `Affine` to scanner RAS+ millimetres,
       `transform @ diag(vox)` (or MRtrix's centred default).

    A non-finite voxel size, which is common on the volume axis, gives a
    scale of one.
    """
    ndim = header.ndim
    voxel_axes = _mrtrix_axes(ndim)
    voxel_space = CoordinateSystem(name="voxel", axes=voxel_axes, order="F")

    phys_axes = [
        replace(axis, unit=_MM if axis.type == "space" else None)
        for axis in voxel_axes
    ]
    phys_space = CoordinateSystem(name="physical", axes=phys_axes)

    spacing = header.spacing
    scale = [v if math.isfinite(v) and v > 0 else 1.0 for v in spacing]
    vox2phys = Scaling(input=voxel_space, output=phys_space, scale=scale)

    # The scanner space is always 3D, because the MRtrix transform maps
    # only the first three axes.
    ras_axes = [
        Axis(
            name,
            "space",
            unit=_MM,
            orientation=Orientation(
                type="anatomical", value=_ORIENTATION[name]
            ),
        )
        for name in _SPATIAL
    ]
    nspatial = min(3, ndim)
    world = CoordinateSystem(name=_SCANNER, axes=ras_axes)
    vox2ras = header.voxel_to_scanner()
    columns = list(range(nspatial)) + [3]
    matrix = vox2ras[:3][:, columns]
    vox2scanner = Affine(input=voxel_space, output=world, matrix=matrix)
    return [vox2phys, vox2scanner]
