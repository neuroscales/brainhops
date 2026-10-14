"""Single-scale TIFF images: one level of one series of a TIFF file."""

# stdlib
from io import BytesIO

# dependencies
import typing_extensions as tx
from bagof.magic import Inherit, NoRepr, replace

# internals
from brainhops._core import path
from brainhops._core.properties import smartproperty
from brainhops._core.streams import preserve_position
from brainhops._core.typing import ArrayProtocol
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.transformations import Transformation
from brainhops.io.base._base import register_format
from brainhops.io.base._dispatch import _to_filename
from brainhops.io.base._utils_files import same_file
from brainhops.io.base.parsers import (
    BinaryFileReader,
    BinaryFileWriter,
    Confidence,
    ParserExistsError,
    SnifferContentError,
    WriterError,
)
from brainhops.io.common._geometry import (
    held_transformations,
    same_transformations,
)
from brainhops.io.images.base import ImageFormat

# this format
from . import _utils as backend
from ._data import _TiffProxy
from ._geometry import level_geometry, record_transformations
from ._metadata import TiffMetadata
from ._raw import read_record
from ._views import _image_to_disk, _image_to_model
from ._writer import (
    Plan,
    copy_source,
    keeps_record,
    plan_file,
    planned_record,
    split_options,
    write_plan,
    write_target,
)

# ----------------------------------------------------------------------
#   READING ADAPTERS
# ----------------------------------------------------------------------


class _TiffReader:
    """The entry points through which TIFF images and pyramids are read.

    This base holds no state. It sniffs the magic number of a file and
    turns a path, a stream or bytes into a
    [`TiffSource`][brainhops.io.images.tiff._utils.TiffSource], from which
    the `from_source` method of each class reads its object.
    """

    @classmethod
    def _sniff_head(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
    ) -> bool:
        with preserve_position(file):
            head = file.read(4)
        if backend.is_tiff(head):
            return True
        if error:
            if error is True:
                error = SnifferContentError
            raise error("This content is not a TIFF file.")
        return False

    @classmethod
    def sniff_bytes(
        cls,
        content: path.BinaryContentLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that bytes hold a TIFF file.

        Parameters
        ----------
        content : bytes
            The content of a file, or its first bytes.
        error : bool or type[Exception], default=False
            Whether to raise an error when the bytes hold no TIFF file.
        **kwargs : Any
            The options of the reader.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return cls.sniff_fileobj(BytesIO(content), error=error, **kwargs)

    @classmethod
    def from_filename(cls, filename: path.FilenameLike, **kwargs) -> tx.Self:
        """Read from a file.

        A local file is opened again by name when the pixels are read, which
        allows memory-mapping, while a remote file is read into memory.

        Parameters
        ----------
        filename : str or PathLike
            The path of the file.
        **kwargs : Any
            The options of `from_source`.

        Raises
        ------
        ParserExistsError
            If the file does not exist.
        """
        if isinstance(filename, str):
            filename = path.Path(filename)
        if not path.exists(filename):
            raise ParserExistsError(f"No such file: {filename}")
        return cls.from_source(
            backend.TiffSource.from_file(filename), **kwargs
        )

    @classmethod
    def from_fileobj(cls, file: tx.IO, **kwargs) -> tx.Self:
        """Read from an open binary file.

        The file is read into memory, and its position is then restored.

        Parameters
        ----------
        file : IO
            The stream, open in binary mode.
        **kwargs : Any
            The options of `from_source`.
        """
        with preserve_position(file):
            content = file.read()
        return cls.from_source(backend.TiffSource(content=content), **kwargs)

    @classmethod
    def from_bytes(cls, content: path.BinaryContentLike, **kwargs) -> tx.Self:
        """Read from bytes.

        Parameters
        ----------
        content : bytes
            The content of the file.
        **kwargs : Any
            The options of `from_source`.
        """
        return cls.from_source(
            backend.TiffSource(content=bytes(content)), **kwargs
        )


# ----------------------------------------------------------------------
#   SINGLE-SCALE IMAGE
# ----------------------------------------------------------------------


@register_format
class TiffImage(
    _TiffReader,
    BinaryFileReader,
    BinaryFileWriter,
    ImageFormat,
    SingleScaleImage,
):
    """One level of one series of a TIFF file, read and written with tifffile.

    The file is a plain TIFF, BigTIFF, OME-TIFF or ImageJ hyperstack. The
    image holds the record of the series that it was read from as
    [`TiffMetadata`][brainhops.io.images.tiff.TiffMetadata], whose record
    is a [`TiffRaw`][brainhops.io.images.tiff.TiffRaw], and the pixels as
    a lazy array in `raw`. The reader options `series` and `level` say
    which series and which level of its pyramid the image is, and the
    level of a pyramid holds the record of the whole series, in which
    `level` finds it. What the file records, such as its dialect, its
    OME-XML, its ImageJ metadata and its tags, is read from the record,
    as `image.metadata.raw.dialect`.

    The data is F-ordered as x, y and z, then time, channels and other
    axes, and is read on first access. The image has a single
    transformation, which scales pixels to `"physical"`, with a
    translation if the file records an origin or if the level is coarser
    than the finest one. This transformation is the identity, with no
    unit, when the pixel size is unknown. It is decoded from the record
    on first access, unless the reader options `pixel_size`, `unit` or
    `origin` override it, in which case it is assigned.

    `raw` and the record describe the file as it was read, or as the image
    was last consolidated. Assigned data is pending until
    [`consolidate`][] encodes it into a copy, or until the image is
    written, which leaves the image as it was. The transformations, read
    or assigned, are compared at [`consolidate`][] with those of the
    record. An image that was read and not changed is written by copying
    the bytes of its file, because tifffile encodes again everything that
    it writes. Any other image is encoded again, with the record as the
    base of what is written: the dialect, the tags, the ImageJ metadata
    and the names of the OME image and channels. A level of a pyramid is
    a single-scale image, so it is written as a file of its level alone,
    and its consolidated copy holds a record of that level alone, at
    index 0.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = backend.EXTENSIONS
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("tiff", "tifffile")

    raw: NoRepr[tx.Optional[ArrayProtocol]] = None
    """The pixels as the file stores them, or `None` without data.

    After a read, `raw` is a lazy array of the level, which presents the
    pixels in the order of brainhops and reads them when they are needed.
    `raw` describes the file as it was read, or as the image was last
    consolidated, so assigning `data` leaves it as it is.
    """

    _data: tx.Optional[ArrayProtocol] = Inherit("data", default=None)
    """The data given to the image, or `None` to decode it from `raw`.

    The image takes over the `data` field of [`SingleScaleImage`][] to
    store it under this private name. An array stored here is pending: it
    is encoded into `raw` only by [`consolidate`][] or by a write.
    """

    _transformations: tx.Optional[tx.List[Transformation]] = Inherit(
        "transformations", default=None
    )
    """The transformations assigned to the image, or `None`.

    The image takes over the `transformations` field of
    [`SingleScaleImage`][] to store an assigned list under this private
    name. `None` means that no list was assigned, so the property decodes
    the list of the record and caches it apart.
    """

    _metadata: NoRepr[tx.Optional[TiffMetadata]] = None

    metadata = smartproperty("metadata", invalidates="transformations")
    """The metadata of the series, which holds its record, or `None`.

    An image built from data has no metadata. Assigning other metadata
    drops the transformations decoded from the previous record, so that
    they are decoded from the new one, and keeps the transformations that
    were assigned.
    """

    series: tx.Optional[int] = None
    """The index of the series that was read, a reader option."""

    level: int = 0
    """The index of the level in the record, where 0 is the finest level.

    It is a reader option, and a level of a pyramid holds the record of
    the whole series, in which it finds its level.
    """

    @smartproperty(cache=True)
    def data(self) -> tx.Optional[ArrayProtocol]:
        """The image data, assigned or decoded from `raw`.

        An array assigned to the image is returned as it is, and stays
        pending until [`consolidate`][] or a write encodes it. Otherwise
        the data is decoded from `raw` on first access and cached. A local,
        uncompressed and contiguous level is a read-only view of a memory
        map of the file, a level read lazily is a dask array, and any
        other level is read in full, read-only. To change the pixels,
        assign `data`. Assigning `None` decodes `raw` again.
        """
        if self.raw is None:
            return None
        return _image_to_model(self.raw)

    @smartproperty(cache=True)
    def transformations(self) -> tx.List[Transformation]:
        """Pixel-to-physical transformations, decoded from the record.

        The list is decoded on first access and cached, apart from an
        assigned list, so a change made to it in place is kept. Assigning
        other metadata drops the decoded list, and keeps an assigned one.
        The list is compared at [`consolidate`][], and by a write, with
        the list that the record decodes to. An image without metadata
        has no transformations. Assigning `None` decodes the record again.
        """
        return record_transformations(self.metadata, self.level)

    @property
    def shape(self) -> tx.Tuple[int, ...]:
        """The shape of the data, read without reading the pixels.

        The shape is that of the assigned data, or else of `raw`, whose
        lazy array knows its shape from the record. An image without data
        raises as [`SingleScaleImage`][] does.
        """
        if self._data is not None:
            return tuple(int(d) for d in self._data.shape)
        if self.raw is None:
            return self.data.shape
        return tuple(int(d) for d in self.raw.shape)

    # --- reading ------------------------------------------------------

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return `LIKELY` for any TIFF or BigTIFF header.

        [`TiffMultiScaleImage`][brainhops.io.images.tiff.TiffMultiScaleImage]
        claims pyramidal series with more confidence, unless a level is
        requested.

        Parameters
        ----------
        file : IO
            The stream, open in binary mode.
        error : bool or type[Exception], default=False
            Whether to raise an error when the stream holds no TIFF file.
        **kwargs : Any
            The options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        if cls._sniff_head(file, error):
            return Confidence.LIKELY
        return Confidence.NO

    @classmethod
    def from_source(
        cls,
        source: backend.TiffSource,
        series: int = 0,
        level: tx.Optional[int] = None,
        pixel_size: tx.Any = None,
        unit: tx.Any = None,
        origin: tx.Any = None,
        mmap: bool = True,
        lazy: tx.Optional[bool] = None,
        **kwargs,
    ) -> "TiffImage":
        """Read one level of one series.

        The record of the series is read at once, from the IFDs of the
        file, and the pixels are read when the data is first accessed.

        Parameters
        ----------
        source : TiffSource
            The source from which the file is read.
        series : int, default=0
            The series, the first by default. A negative index counts from
            the end.
        level : int, optional
            The pyramid level, 0 (the full resolution) by default. A
            negative index counts from the end.
        pixel_size : float or Sequence[float] or Mapping[str, float], optional
            The pixel size, which overrides the pixel size of the file.
        unit : str or Unit, optional
            The unit of `pixel_size`, or the unit to which the sizes of the
            file are converted.
        origin : bool, float, Sequence[float] or Mapping[str, float], optional
            The position of the first pixel, in the unit of the pixel size,
            which overrides the OME plane positions. `False` ignores the
            positions of the file.
        mmap : bool, default=True
            Whether to memory-map a local uncompressed series.
        lazy : bool, optional
            Whether the data is a dask array over the Zarr store of tifffile,
            which needs dask and zarr. By default, the data is a dask array
            only when dask is the array backend.
        **kwargs : Any
            Refused, since TIFF has no other option.

        Returns
        -------
        TiffImage
            The image, whose transformations are assigned when
            `pixel_size`, `unit` or `origin` is given.

        Raises
        ------
        ParserContentError
            If the file cannot be read as TIFF.
        IndexError
            If there is no such series or level.
        TypeError
            If an unknown option is given.
        """
        if kwargs:
            raise TypeError(
                f"{cls.__name__} does not take the option(s) "
                f"{', '.join(sorted(kwargs))}."
            )
        index = 0 if level is None else int(level)
        with source.open() as tif:
            backend._select(tif, int(series), index)
            record = read_record(tif, series)
        index %= len(record.levels)
        image = cls(
            raw=_TiffProxy(source, record, index, mmap=mmap, lazy=lazy),
            metadata=TiffMetadata.from_raw(record),
            series=record.series,
            level=index,
        )
        if any(value is not None for value in (pixel_size, unit, origin)):
            # The options describe the geometry, so they are assigned, and
            # the image is written with that geometry.
            image.transformations = [
                level_geometry(
                    record.storage_axes,
                    record.levels[0].shape,
                    record.levels[index].shape,
                    record,
                    pixel_size=pixel_size,
                    unit=unit,
                    origin=origin,
                )
            ]
        return image

    # --- writing ------------------------------------------------------

    def consolidate(
        self,
        *,
        inplace: bool = False,
        dialect: tx.Optional[str] = None,
        name: tx.Optional[str] = None,
        bigtiff: tx.Optional[bool] = None,
    ) -> "TiffImage":
        """Encode the pending data and geometry into `raw` and the record.

        Consolidating makes `raw` and the record describe the image, as a
        file written from it would. An image that has a record of one
        level, at index 0, no pending data, and the transformations of its
        record or none read, consolidated without options or with options
        that name what the record holds, is left as it is: the copy shares
        `raw` and the metadata. Any other image is planned as the writer
        plans its file, and the record of the copy is read from a file of
        that plan that tifffile writes without pixels, so that it holds
        what the written file holds. Its `raw` is the assigned data, or
        else the lazy array, which is not read. A level of a pyramid is
        therefore narrowed to a record of its level alone, at index 0, and
        the record held before is never changed. Nothing is pending in the
        copy, whose transformations are decoded from the new record.

        The dialect is chosen as the writer chooses it: `dialect`, else
        OME-TIFF for a `name` with an OME suffix, else ImageJ for a record
        of ImageJ that ImageJ can still store, else the dialect of the
        record while its geometry is kept, else OME-TIFF when the pixel
        size or the time interval is known, else plain TIFF. The options of
        tifffile that encode the pixels, such as `compression`, change
        nothing that the record holds, so only the writers take them.

        Parameters
        ----------
        inplace : bool, default=False
            Whether to change this image rather than a copy.
        dialect : {"ome", "imagej", "plain"}, optional
            The metadata dialect.
        name : str, optional
            The name of the file, which may ask for OME-TIFF.
        bigtiff : bool, optional
            Whether to write BigTIFF. By default, BigTIFF is written only
            when the data is too large for a classic TIFF (about 4 GiB).

        Returns
        -------
        TiffImage
            The consolidated copy, or this image with `inplace=True`.

        Raises
        ------
        WriterError
            If there is no data, the axes cannot be stored, or the dialect
            is unknown or cannot store the image.
        """
        if self._untouched(dialect, name, bigtiff):
            if not inplace:
                return replace(self, transformations=None)
            self.transformations = None
            return self
        plan = self._planned(dialect, name, bigtiff)
        record = planned_record(plan)
        if self.metadata is None:
            metadata = TiffMetadata.from_raw(record)
        else:
            metadata = replace(self.metadata, raw=record)
        raw = plan.arrays[0]
        if not inplace:
            return replace(
                self,
                data=None,
                transformations=None,
                raw=raw,
                metadata=metadata,
                series=record.series,
                level=0,
            )
        self.raw = raw
        self.metadata = metadata
        self.series = record.series
        self.level = 0
        self.data = None
        self.transformations = None
        return self

    def to_bytes(self, **kwargs) -> bytes:
        """Encode the image as TIFF.

        An image that [`consolidate`][] leaves as it is, and whose `raw`
        is the lazy array of its record, gives the bytes of its file, as
        they are. Any other image is encoded by tifffile, as the image
        consolidated with the options holds it. The image itself is not
        changed.

        Parameters
        ----------
        **kwargs : Any
            The options of [`consolidate`][] (`dialect`, `name` and
            `bigtiff`), and the options of `TiffWriter.write` of tifffile
            that encode the pixels: `compression`, `compressionargs`,
            `predictor`, `tile` and `rowsperstrip`.

        Returns
        -------
        bytes
            The encoded image.

        Raises
        ------
        TypeError
            If an unknown option is given.
        WriterError
            If the image cannot be written.
        """
        options, encoding = split_options(kwargs)
        if not encoding and self._copies_source(**options):
            return self.raw.read_bytes()
        buffer = BytesIO()
        write_plan(self._planned(**options), buffer, encoding)
        return buffer.getvalue()

    def to_fileobj(self, file: tx.IO, **kwargs) -> None:
        """Write the image to an open binary file.

        If the file has a name, the name chooses the dialect as in
        [`to_bytes`][].

        Parameters
        ----------
        file : IO
            The stream, open in binary mode.
        **kwargs : Any
            The options of [`to_bytes`][].
        """
        kwargs.setdefault("name", _to_filename(file))
        file.write(self.to_bytes(**kwargs))

    def to_filename(self, filename: path.FilenameLike, **kwargs) -> None:
        """Write the image to a file.

        A name ending with `.ome.tif`, `.ome.tiff`, `.ome.btf`, `.ome.tf2`
        or `.ome.tf8` asks for OME-TIFF. An image that was read and not
        changed is written by copying the bytes of its file. Any other
        image is encoded by tifffile. Everything that will be written is
        planned before the file is opened, so that an image that cannot be
        written leaves no file behind. A local file is written next to its
        target and then moved over it, and a remote file is first encoded
        in memory. When the lazy array of the image reads the file that is
        written, the image then holds what the file holds (#426): its
        `raw` reads the new file, and its record is read from it.

        Parameters
        ----------
        filename : str or PathLike
            The path of the file.
        **kwargs : Any
            The options of [`to_bytes`][].

        Raises
        ------
        TypeError
            If an unknown option is given.
        WriterError
            If the image cannot be written.
        """
        if isinstance(filename, str):
            filename = path.Path(filename)
        kwargs.setdefault("name", _to_filename(filename))
        options, encoding = split_options(kwargs)
        raw = self.raw
        reading = isinstance(raw, _TiffProxy) and same_file(
            raw.source.filename, filename
        )
        if not encoding and self._copies_source(**options):
            if not reading:
                write_target(filename, lambda f: copy_source(raw.source, f))
            return
        plan = self._planned(**options)
        write_target(filename, lambda f: write_plan(plan, f, encoding))
        if reading:
            # The image holds what the file holds, and its lazy array reads
            # the new file, since the lazy array of the old file describes
            # how that file stored the pixels.
            source = backend.TiffSource.from_file(filename)
            with source.open() as tif:
                record = read_record(tif, 0)
            self.raw = _TiffProxy(source, record, 0, raw.mmap, raw.lazy)
            self.metadata = TiffMetadata.from_raw(record)
            self.series = record.series
            self.level = 0
            self.data = None
            self.transformations = None

    def to_file(self, file: path.FileLike, **kwargs) -> None:
        """Write the image to a path or a stream.

        Parameters
        ----------
        file : path-like or IO
            The path or the stream.
        **kwargs : Any
            The options of [`to_bytes`][].
        """
        if isinstance(file, (str, path.PathLike)):
            return self.to_filename(file, **kwargs)
        return super().to_file(file, **kwargs)

    # --- helpers ------------------------------------------------------

    def _untouched(
        self,
        dialect: tx.Optional[str] = None,
        name: tx.Optional[str] = None,
        bigtiff: tx.Optional[bool] = None,
    ) -> bool:
        """Tell whether consolidating with the options changes nothing.

        The image has a record of one level, at index 0, no pending data,
        and the transformations of its record or none read, and the
        options name what the record holds.
        """
        metadata = self.metadata
        record = None if metadata is None else metadata.raw
        if (
            self._data is not None
            or record is None
            or len(record.levels) != 1
            or self.level != 0
            or not keeps_record(record, dialect, name, bigtiff)
        ):
            return False
        held = held_transformations(self)
        return held is None or same_transformations(
            held, record_transformations(metadata, 0)
        )

    def _copies_source(
        self,
        dialect: tx.Optional[str] = None,
        name: tx.Optional[str] = None,
        bigtiff: tx.Optional[bool] = None,
    ) -> bool:
        """Tell whether a write copies the bytes of the file of the image.

        The image is left as it is by [`consolidate`][], and its `raw` is
        the lazy array of the record that it holds, so the file holds
        exactly what the image holds.
        """
        raw = self.raw
        return (
            self._untouched(dialect, name, bigtiff)
            and isinstance(raw, _TiffProxy)
            and raw.level == 0
            and raw.record is self.metadata.raw
        )

    def _planned(
        self,
        dialect: tx.Optional[str] = None,
        name: tx.Optional[str] = None,
        bigtiff: tx.Optional[bool] = None,
    ) -> Plan:
        """Plan the file of the image, as [`plan_file`][] describes.

        The list of the image is read without being stored, so that a
        write leaves the image as it was.

        Raises
        ------
        WriterError
            If the image has no data or cannot be written.
        """
        if self._data is not None:
            raw = _image_to_disk(self._data)
        else:
            raw = self.raw
        if raw is None:
            raise WriterError(
                "This image has no data, so there is nothing to write."
            )
        metadata = self.metadata
        record = None if metadata is None else metadata.raw
        decoded = record_transformations(metadata, self.level)
        held = held_transformations(self)
        transformations = decoded if held is None else held
        kept = record is not None and (
            held is None or same_transformations(held, decoded)
        )
        return plan_file(
            [raw],
            transformations[-1] if transformations else None,
            record,
            kept,
            dialect=dialect,
            name=name,
            bigtiff=bigtiff,
        )
