"""Multiscale TIFF images: pyramidal series with one image per level."""

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
from brainhops.datamodel.images import MultiScaleImage, SingleScaleImage
from brainhops.datamodel.transformations import Transformation
from brainhops.io.base._base import register_format
from brainhops.io.base._dispatch import _to_filename
from brainhops.io.base._utils_files import same_file
from brainhops.io.base.parsers import (
    BinaryFileReader,
    BinaryFileWriter,
    Confidence,
    WriterError,
)
from brainhops.io.common._geometry import (
    held_transformations,
    same_transformations,
)
from brainhops.io.common._pyramids import held_images, own_level, own_levels
from brainhops.io.images.base import ImageFormat

# this format
from . import _utils as backend
from ._data import _TiffProxy
from ._geometry import level_geometry, record_transformations
from ._image import TiffImage, _TiffReader
from ._metadata import TiffMetadata
from ._raw import TiffRaw, read_record
from ._views import _image_to_disk
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

# A vendor whole-slide pyramid scores above a single-scale TIFF (LIKELY) and
# below the dedicated OpenSlide reader (CERTAIN).
_WHOLE_SLIDE = 0.9

# ----------------------------------------------------------------------
#   MULTISCALE IMAGE
# ----------------------------------------------------------------------


@register_format
class TiffMultiScaleImage(
    _TiffReader,
    BinaryFileReader,
    BinaryFileWriter,
    ImageFormat,
    MultiScaleImage,
):
    """A pyramidal TIFF series, read as a multiscale image.

    The series is an OME-TIFF or plain TIFF with SubIFD levels, or any other
    pyramid that tifffile recognises. The pyramid holds the record of the
    series as [`TiffMetadata`][brainhops.io.images.tiff.TiffMetadata], and
    the lazy array of each level, finest first, as a tuple in `raw`. Its
    levels are [`TiffImage`][] objects decoded from `raw` and the record on
    first access, which hold the same metadata, by identity, and the index
    `level` of their level in the record, and whose pixels are read when
    their data is accessed. An assigned list of levels is pending, and so
    is data assigned to a level.

    The pixel size of a level is the pixel size of the base level multiplied
    by the downsampling factor of the level. Along each axis, this factor is
    the base shape divided by the shape of the level. The levels are aligned by
    their extent, so pixel `i` of a level downsampled by `f` is centred on the
    base coordinate `f * i + (f - 1) / 2`. This coordinate is the centre of the
    block of base pixels that the pixel summarises, as in block-averaged and
    OME-Zarr pyramids. All levels map to the same `"physical"` system, so the
    pyramid has no transformation of its own, like an OME-Zarr pyramid that
    declares no common transformation.

    A TIFF file stores the geometry of the finest level only, and the
    levels are written as they are given, with no check of the ratios of
    their shapes. A preferred transformation assigned to the pyramid is
    therefore composed into the transformation of the finest level when
    the pyramid is written, and the transformations of the coarser levels
    are not written: a reader derives them again from the shapes of the
    levels. ImageJ cannot store a pyramid. A pyramid that was read and not
    changed is written by copying the bytes of its file, because tifffile
    encodes again everything that it writes.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = backend.EXTENSIONS
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("tiff", "tifffile")

    raw: NoRepr[tx.Optional[tx.Tuple[ArrayProtocol, ...]]] = None
    """The array of each level, finest first, as the file stores it.

    After a read, each array is a lazy array of its level. `raw` is a
    tuple, so that it cannot be changed in place, and it describes the
    file as it was read, or as the pyramid was last consolidated.
    """

    _images: tx.Optional[tx.List[SingleScaleImage]] = Inherit(
        "images", default=None
    )
    """The levels assigned to the pyramid, or `None`.

    The pyramid takes over the `images` field of [`MultiScaleImage`][] to
    store an assigned list under this private name. `None` means that no
    list was assigned, so the property decodes the levels from `raw` and
    the record and caches them apart.
    """

    _transformations: tx.Optional[tx.List[Transformation]] = Inherit(
        "transformations", default=None
    )
    """The placements assigned to the pyramid, or `None`.

    `None` means that no list was assigned, so the property decodes the
    placements of the record, which a TIFF file never stores, and caches
    the empty list apart.
    """

    _metadata: NoRepr[tx.Optional[TiffMetadata]] = None

    metadata = smartproperty(
        "metadata", invalidates=("images", "transformations")
    )
    """The metadata of the series, which holds its record, or `None`.

    Every level of the pyramid holds the same metadata. Assigning other
    metadata drops the levels and the placements decoded from the
    previous record, and keeps those that were assigned. The levels that
    are dropped keep the previous metadata.
    """

    series: tx.Optional[int] = None
    """The index of the series that was read, a reader option."""

    @smartproperty(cache=True)
    def images(self) -> tx.List[SingleScaleImage]:
        """The levels of the pyramid, finest first.

        The levels are decoded from `raw` and the record on first access
        and cached, apart from an assigned list, so `images[0] is
        images[0]` and a level changed in place, such as one whose data is
        assigned, is written. Each level is a [`TiffImage`][] that holds
        its array of `raw`, the metadata of the pyramid and its index in
        `level`. Assigning a list replaces the stored one, and assigning
        `None` decodes the levels again.
        """
        return _levels(self.raw, self.metadata)

    @smartproperty(cache=True)
    def transformations(self) -> tx.List[Transformation]:
        """The placements of the pyramid, which a TIFF file never stores.

        The decoded list is empty, and an assigned list is pending: it is
        composed into the transformation of the finest level when the
        pyramid is written.
        """
        return []

    @property
    def shape(self) -> tx.Tuple[int, ...]:
        """The shape of the finest level, read without reading the pixels.

        The shape is read from the levels that the pyramid holds, or else
        from the first array of `raw`.
        """
        images = held_images(self)
        if images is not None:
            return tuple(images[0].shape) if images else ()
        if self.raw:
            return tuple(int(n) for n in self.raw[0].shape)
        return ()

    # --- reading ------------------------------------------------------

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        series: int = 0,
        level: tx.Optional[int] = None,
        **kwargs,
    ) -> float:
        """Return the confidence that an open file holds a pyramid.

        The score is `CERTAIN` when the requested series (the first by default)
        has several levels and no level is requested, and `NO` otherwise. A
        vendor whole-slide image (Aperio SVS, Hamamatsu NDPI, Philips, Leica
        SCN or Ventana BIF) scores slightly less (0.9), still above
        [`TiffImage`][], so that the dedicated reader of
        [`brainhops.io.images.openslide`][] takes it when installed.

        Parameters
        ----------
        file : IO
            The stream, open in binary mode.
        error : bool or type[Exception], default=False
            Whether to raise an error when the stream holds no TIFF file.
        series : int, default=0
            The series whose levels are counted.
        level : int, optional
            The level that the reader is asked for, which a pyramid refuses.
        **kwargs : Any
            The other options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        if level is not None:
            return Confidence.NO
        if not cls._sniff_head(file, error):
            return Confidence.NO
        tf = backend._require_tifffile()
        try:
            with preserve_position(file):
                with tf.TiffFile(file) as tif:
                    levels = len(tif.series[int(series)].levels)
                    slide = backend.is_whole_slide(tif)
        except Exception:
            return Confidence.NO
        if levels <= 1:
            return Confidence.NO
        return _WHOLE_SLIDE if slide else Confidence.CERTAIN

    @classmethod
    def from_source(
        cls,
        source: backend.TiffSource,
        series: int = 0,
        pixel_size: tx.Any = None,
        unit: tx.Any = None,
        origin: tx.Any = None,
        mmap: bool = True,
        lazy: tx.Optional[bool] = None,
        **kwargs,
    ) -> "TiffMultiScaleImage":
        """Read every level of one series.

        The record of the series is read at once, from the IFDs of the
        file, and each level is a lazy array that reads its pixels when
        they are needed.

        Parameters
        ----------
        source : TiffSource
            The source from which the file is read.
        series : int, default=0
            The series, the first by default. A negative index counts from
            the end.
        pixel_size : float or Sequence[float] or Mapping[str, float], optional
            The pixel size of the finest level, which overrides the pixel
            size of the file.
        unit : str or Unit, optional
            The unit of `pixel_size`, or the unit to which the sizes of the
            file are converted.
        origin : bool, float, Sequence[float] or Mapping[str, float], optional
            The position of the first pixel, which overrides the OME plane
            positions. `False` ignores the positions of the file.
        mmap : bool, default=True
            Whether to memory-map the local uncompressed levels.
        lazy : bool, optional
            Whether the data of each level is a dask array.
        **kwargs : Any
            Refused, since TIFF has no other option.

        Returns
        -------
        TiffMultiScaleImage
            The pyramid. When `pixel_size`, `unit` or `origin` is given, its
            levels are assigned, with the transformations that the options
            give them.

        Raises
        ------
        TypeError
            If an unknown option is given.
        IndexError
            If there is no such series.
        """
        if kwargs:
            raise TypeError(
                f"{cls.__name__} does not take the option(s) "
                f"{', '.join(sorted(kwargs))}."
            )
        with source.open() as tif:
            record = read_record(tif, series)
        metadata = TiffMetadata.from_raw(record)
        raw = tuple(
            _TiffProxy(source, record, index, mmap=mmap, lazy=lazy)
            for index in range(len(record.levels))
        )
        pyramid = cls(raw=raw, metadata=metadata, series=record.series)
        if any(value is not None for value in (pixel_size, unit, origin)):
            # The options describe the geometry, so the levels are assigned
            # with the transformations that they give.
            levels = _levels(raw, metadata)
            for index, level in enumerate(levels):
                level.transformations = [
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
            pyramid.images = levels
        return pyramid

    @classmethod
    def from_instance(
        cls, other: tx.Any, *args, **kwargs
    ) -> "TiffMultiScaleImage":
        """Copy a pyramid into this format.

        Within the format, the stored fields are copied, so a copy of an
        untouched pyramid shares `raw` and the metadata and decodes
        nothing. From another class, the levels that `other` reports are
        copied, because its private fields mean nothing to this format.

        Parameters
        ----------
        other : MultiScaleImage
            The pyramid to copy.
        *args : Any
            Positional fields of the copy.
        **kwargs : Any
            Other fields of the copy.

        Returns
        -------
        TiffMultiScaleImage
            The copy.
        """
        if not isinstance(other, cls):
            images = getattr(other, "images", None)
            if images is not None:
                kwargs.setdefault("images", list(images))
        return super().from_instance(other, *args, **kwargs)

    # --- writing ------------------------------------------------------

    def consolidate(
        self,
        *,
        inplace: bool = False,
        dialect: tx.Optional[str] = None,
        name: tx.Optional[str] = None,
        bigtiff: tx.Optional[bool] = None,
    ) -> "TiffMultiScaleImage":
        """Encode the pending levels and placements into `raw` and the record.

        A pyramid that has a record, whose levels are those of its file,
        whose placements were never read or are empty, and which is
        consolidated without options or with options that name what the
        record holds, is left as it is: the copy shares `raw` and the
        metadata. Any other pyramid is planned as the writer plans its
        file, and the record of the copy is read from a file of that plan
        that tifffile writes without pixels, so that it holds what the
        written file holds. A level of the file keeps its array, a level
        with pending data gives that data, and a level of another class, or
        one that holds other metadata, gives its decoded data. The levels
        are written as they are given. The placements are composed into
        the transformation of the finest level, which the file stores, and
        the transformations of the coarser levels are derived again from
        their shapes. Nothing is pending in the copy, whose levels are
        decoded from the new record.

        Parameters
        ----------
        inplace : bool, default=False
            Whether to change this pyramid rather than a copy.
        dialect : {"ome", "plain"}, optional
            The metadata dialect. ImageJ cannot store a pyramid.
        name : str, optional
            The name of the file, which may ask for OME-TIFF.
        bigtiff : bool, optional
            Whether to write BigTIFF. By default, BigTIFF is written only
            when the levels are too large for a classic TIFF (about 4 GiB).

        Returns
        -------
        TiffMultiScaleImage
            The consolidated copy, or this pyramid with `inplace=True`.

        Raises
        ------
        WriterError
            If the pyramid has no level, a level has no data or another
            number of axes, or the dialect is unknown or cannot store the
            pyramid.
        """
        if self._untouched(dialect, name, bigtiff):
            if not inplace:
                return replace(self, images=None, transformations=None)
            self.images = None
            self.transformations = None
            return self
        plan = self._planned(dialect, name, bigtiff)
        record = planned_record(plan)
        if self.metadata is None:
            metadata = TiffMetadata.from_raw(record)
        else:
            metadata = replace(self.metadata, raw=record)
        raw = tuple(plan.arrays)
        if not inplace:
            return replace(
                self,
                images=None,
                transformations=None,
                raw=raw,
                metadata=metadata,
                series=record.series,
            )
        self.raw = raw
        self.metadata = metadata
        self.series = record.series
        self.images = None
        self.transformations = None
        return self

    def to_bytes(self, **kwargs) -> bytes:
        """Encode the pyramid as TIFF.

        The finest level is the main image, and the other levels are its
        SubIFDs. A pyramid that [`consolidate`][] leaves as it is, and
        whose `raw` holds the lazy arrays of its record, gives the bytes of
        its file, as they are. Any other pyramid is encoded by tifffile,
        as the pyramid consolidated with the options holds it. The pyramid
        itself is not changed.

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
            The encoded pyramid.

        Raises
        ------
        TypeError
            If an unknown option is given.
        WriterError
            If the pyramid cannot be written.
        """
        options, encoding = split_options(kwargs)
        if not encoding and self._copies_source(**options):
            return self.raw[0].read_bytes()
        buffer = BytesIO()
        write_plan(self._planned(**options), buffer, encoding)
        return buffer.getvalue()

    def to_fileobj(self, file: tx.IO, **kwargs) -> None:
        """Write the pyramid to an open binary file.

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
        """Write the pyramid to a file.

        A pyramid that was read and not changed is written by copying the
        bytes of its file, and any other pyramid is encoded by tifffile,
        as [`TiffImage.to_filename`][brainhops.io.images.tiff.TiffImage.to_filename]
        describes. When the lazy arrays of the pyramid read the file that
        is written, the pyramid then holds what the file holds (#426): its
        `raw` is a new tuple of lazy arrays of the new file, and its record
        is read from it.

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
            If the pyramid cannot be written.
        """
        if isinstance(filename, str):
            filename = path.Path(filename)
        kwargs.setdefault("name", _to_filename(filename))
        options, encoding = split_options(kwargs)
        raws = self.raw or ()
        reading = [
            raw
            for raw in raws
            if isinstance(raw, _TiffProxy)
            and same_file(raw.source.filename, filename)
        ]
        if not encoding and self._copies_source(**options):
            if not reading:
                source = raws[0].source
                write_target(filename, lambda f: copy_source(source, f))
            return
        plan = self._planned(**options)
        write_target(filename, lambda f: write_plan(plan, f, encoding))
        if reading:
            # The pyramid holds what the file holds, and its lazy arrays
            # read the new file.
            source = backend.TiffSource.from_file(filename)
            with source.open() as tif:
                record = read_record(tif, 0)
            mmap, lazy = reading[0].mmap, reading[0].lazy
            self.raw = tuple(
                _TiffProxy(source, record, index, mmap=mmap, lazy=lazy)
                for index in range(len(record.levels))
            )
            self.metadata = TiffMetadata.from_raw(record)
            self.series = record.series
            self.images = None
            self.transformations = None

    def to_file(self, file: path.FileLike, **kwargs) -> None:
        """Write the pyramid to a path or a stream.

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

        The pyramid has a record, its levels are those of its file, its
        placements were never read or are empty, and the options name what
        the record holds.
        """
        metadata = self.metadata
        record = None if metadata is None else metadata.raw
        if (
            record is None
            or self.raw is None
            or not keeps_record(record, dialect, name, bigtiff)
            or not own_levels(self, TiffImage, record_transformations)
        ):
            return False
        held = held_transformations(self)
        return held is None or same_transformations(held, [])

    def _copies_source(
        self,
        dialect: tx.Optional[str] = None,
        name: tx.Optional[str] = None,
        bigtiff: tx.Optional[bool] = None,
    ) -> bool:
        """Tell whether a write copies the bytes of the file of the pyramid.

        The pyramid is left as it is by [`consolidate`][], and its `raw`
        holds the lazy array of each level of its record, all of one file.
        """
        if not self._untouched(dialect, name, bigtiff):
            return False
        record = self.metadata.raw
        raws = self.raw
        return (
            len(raws) == len(record.levels)
            and all(
                isinstance(raw, _TiffProxy)
                and raw.record is record
                and raw.level == index
                and raw.source is raws[0].source
                for index, raw in enumerate(raws)
            )
        )

    def _planned(
        self,
        dialect: tx.Optional[str] = None,
        name: tx.Optional[str] = None,
        bigtiff: tx.Optional[bool] = None,
    ) -> Plan:
        """Plan the file of the pyramid, as [`plan_file`][] describes.

        The levels and the placements are read without being stored, so
        that a write leaves the pyramid as it was.

        Raises
        ------
        WriterError
            If the pyramid has no level, or a level has no data.
        """
        metadata = self.metadata
        record = None if metadata is None else metadata.raw
        images = held_images(self)
        if images is None:
            images = _levels(self.raw, metadata)
        arrays = [
            _level_array(self, image, index)
            for index, image in enumerate(images)
        ]
        transformation, kept = _finest_transformation(self, images)
        placements = held_transformations(self) or []
        if placements:
            # A TIFF file stores no placement, so the preferred one is
            # composed into the geometry of the finest level.
            kept = False
            if transformation is None:
                transformation = placements[-1]
            else:
                transformation = placements[-1] @ transformation
        return plan_file(
            arrays,
            transformation,
            record,
            kept and record is not None,
            dialect=dialect,
            name=name,
            bigtiff=bigtiff,
        )


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _levels(
    raw: tx.Optional[tx.Tuple[ArrayProtocol, ...]],
    metadata: tx.Optional[TiffMetadata],
) -> tx.List[TiffImage]:
    """Return a level object for each array of a pyramid, finest first."""
    if raw is None:
        return []
    record = None if metadata is None else metadata.raw
    series = record.series if isinstance(record, TiffRaw) else None
    return [
        TiffImage(raw=array, metadata=metadata, level=index, series=series)
        for index, array in enumerate(raw)
    ]


def _level_array(
    pyramid: TiffMultiScaleImage, image: tx.Any, index: int
) -> ArrayProtocol:
    """Return the array that a level of a pyramid stores.

    A level of the file gives its array of `raw`, a TIFF level gives its
    pending data or else its own `raw`, and an image of another class
    gives its decoded data.

    Raises
    ------
    WriterError
        If the level has no data.
    """
    if own_level(pyramid, image, index, TiffImage, record_transformations):
        return pyramid.raw[index]
    if type(image) is TiffImage:
        if image._data is not None:
            array = _image_to_disk(image._data)
        else:
            array = image.raw
    else:
        data = image.data
        array = None if data is None else _image_to_disk(data)
    if array is None:
        raise WriterError(
            f"Level {index} of this pyramid has no data, so it cannot be "
            f"written."
        )
    return array


def _finest_transformation(
    pyramid: TiffMultiScaleImage, images: tx.Sequence[tx.Any]
) -> tx.Tuple[tx.Optional[Transformation], bool]:
    """Return the transformation of the finest level of a pyramid to write.

    The transformation is the list that the finest level holds, read
    without being stored, or the list that its record decodes to. The
    second value tells whether it is the geometry of the record of the
    pyramid, so that the dialect of the record can be kept.
    """
    if not images:
        return None, False
    image = images[0]
    if type(image) is TiffImage:
        decoded = record_transformations(image.metadata, image.level)
        held = held_transformations(image)
        transformations = decoded if held is None else held
        kept = image.metadata is pyramid.metadata and (
            held is None or same_transformations(held, decoded)
        )
    else:
        transformations = list(image.transformations or [])
        kept = False
    if not transformations:
        return None, kept
    return transformations[-1], kept
