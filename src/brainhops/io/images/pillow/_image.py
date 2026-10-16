"""The raster image format read and written with Pillow."""

# stdlib
from io import BytesIO
from numbers import Number

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import Inherit, NoRepr, replace

# internals
from brainhops._core import path
from brainhops._core.properties import smartproperty
from brainhops._core.typing import ArrayProtocol
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.metadata import ConversionReport
from brainhops.datamodel.metadata._report import OnLoss, apply_loss_policy
from brainhops.datamodel.metadata._vocabulary import FIELDS
from brainhops.datamodel.transformations import Transformation
from brainhops.io.base._base import register_format
from brainhops.io.base._dispatch import _to_filename
from brainhops.io.base._utils_files import write_file
from brainhops.io.base.parsers import (
    BinaryFileReader,
    BinaryFileWriter,
    WriterError,
)
from brainhops.io.common._geometry import (
    held_transformations,
    same_transformations,
)
from brainhops.io.images.base import ImageFormat
from brainhops.io.images.base import _utils_raster as raster
from brainhops.io.metadata._base import metadata_to_write, requested_fields

# this format
from ._data import _PillowProxy, read_pillow_image, reads_from
from ._metadata import PillowMetadata
from ._raw import PillowRaw
from ._utils import (
    EXTENSIONS,
    array_to_pillow,
    can_write,
    encode_pillow,
    format_for_name,
)
from ._views import _image_to_model

_DpiLike = tx.Union[None, bool, float, tx.Sequence[float]]

# The output format when neither `format=` nor a file name nor the record
# names one that Pillow writes.
_DEFAULT_FORMAT = "PNG"

# The keys of `info` that the writer passes back to Pillow, when the
# format takes them, so that they survive a change of the pixels.
_ROUND_TRIP_INFO = ("icc_profile", "exif")

_JPEG_OPTIONS = frozenset(
    {
        "comment",
        "exif",
        "extra",
        "icc_profile",
        "keep_rgb",
        "optimize",
        "progression",
        "progressive",
        "qtables",
        "quality",
        "restart_marker_blocks",
        "restart_marker_rows",
        "smooth",
        "streamtype",
        "subsampling",
    }
)

_SAVE_OPTIONS: tx.Dict[str, tx.FrozenSet[str]] = {
    "BMP": frozenset(),
    "DIB": frozenset(),
    "GIF": frozenset(
        {
            "background",
            "comment",
            "disposal",
            "duration",
            "include_color_table",
            "interlace",
            "loop",
            "optimize",
            "palette",
            "transparency",
        }
    ),
    "JPEG": _JPEG_OPTIONS,
    "JPEG2000": frozenset(
        {
            "cinema_mode",
            "codeblock_size",
            "comment",
            "irreversible",
            "mct",
            "no_jp2",
            "num_resolutions",
            "offset",
            "plt",
            "precinct_size",
            "progression",
            "quality_layers",
            "quality_mode",
            "signed",
            "tile_offset",
            "tile_size",
        }
    ),
    "MPO": _JPEG_OPTIONS,
    "PCX": frozenset(),
    "PNG": frozenset(
        {
            "bits",
            "compress_level",
            "compress_type",
            "dictionary",
            "exif",
            "icc_profile",
            "optimize",
            "pnginfo",
            "transparency",
        }
    ),
    "PPM": frozenset(),
    "TGA": frozenset({"compression", "id_section", "orientation", "rle"}),
    "TIFF": frozenset(
        {
            "artist",
            "compression",
            "copyright",
            "date_time",
            "description",
            "exif",
            "icc_profile",
            "quality",
            "resolution",
            "resolution_unit",
            "software",
            "strip_size",
            "tiffinfo",
            "x_resolution",
            "y_resolution",
        }
    ),
    "WEBP": frozenset(
        {
            "alpha_quality",
            "exact",
            "exif",
            "icc_profile",
            "lossless",
            "method",
            "quality",
            "xmp",
        }
    ),
}
"""The options that Pillow reads when it writes one frame in a format.

The keys are the options that the function `_save` of the plugin of each
format reads from `im.encoderinfo`, in Pillow 10.4.0
(`src/PIL/<Format>ImagePlugin.py`). The resolution is the `dpi` option of
the image writer, so it is not listed, and the options of `_save_all`,
which writes several frames, are not either. Pillow ignores an option
that a plugin does not read, so the writer refuses it instead. A format
that is not listed takes no option.
"""


@register_format
class PillowImage(
    ImageFormat, SingleScaleImage, BinaryFileReader, BinaryFileWriter
):
    """A two-dimensional raster image, read and written with Pillow.

    A raster image holds the header of its file as [`PillowMetadata`][],
    whose record is a [`PillowRaw`][], and the pixels of a frame in `rawdata`.
    When the image is read from a file, `rawdata` is a lazy array that decodes
    the frame only when it is needed, and `data` is decoded from it on
    first access. The data are `(x, y)` for an image of one sample per
    pixel and `(x, y, c)` otherwise, in Fortran order. The format of the
    file, the mode of the frame, its `info` dictionary and the number of
    frames are in the record, as `image.metadata.raw.format` and so on.

    The image has a single transformation, which scales pixels to
    `"physical"`. It is decoded from the record on first access, unless
    other transformations are assigned, and it is the identity, with no
    unit, unless the pixel size is known (see
    [`brainhops.io.images.pillow`][]). An image read with `dpi=True`
    decodes the pixel size from the resolution of its record, and a pixel
    size given when the image is read is assigned.

    `rawdata` and the record describe the file as it was read, or as the image
    was last consolidated. Data that are assigned are pending until
    [`consolidate`][] encodes them into a copy, or until the image is
    written, which writes such a copy and leaves the image as it was. The
    transformations, read or assigned, are compared at [`consolidate`][]
    with those that the record decodes to. An image that is read and
    written again in the same format without changes is written as the
    bytes of its file, whatever the format, so it keeps them exactly.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = EXTENSIONS
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("pillow",)

    rawdata: NoRepr[tx.Optional[ArrayProtocol]] = None
    """The pixels of the frame as Pillow decodes them, `(rows, columns[,
    samples])`, or `None` without data.

    After a read, `rawdata` is a lazy array, which decodes the frame
    when it is read and gives the bytes of the file. `rawdata` describes
    the file as it was read, or as the image was last consolidated, so
    assigning `data` leaves it as it is.
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

    _metadata: NoRepr[tx.Optional[PillowMetadata]] = Inherit(
        "metadata", default=None
    )

    metadata = smartproperty("metadata", invalidates="transformations")
    """The metadata of the file, which holds its record, or `None`.

    An image built from data has no metadata. Assigning other metadata
    drops the transformations decoded from the previous record, so that
    they are decoded from the new one, and keeps the transformations that
    were assigned.
    """

    frame: int = 0
    """The index of the frame that `rawdata` decodes, a reader option.

    The record describes the same frame. A file of several frames is
    written back whole while the image is unchanged, and a changed image
    is written as one frame, so its consolidated copy reads frame 0.
    """

    palette: bool = True
    """Whether `rawdata` looks up the colours of a palette image, a reader
    option.

    Without the lookup, the data of a palette image are its indices.
    """

    dpi: bool = False
    """Whether the pixel size is decoded from the resolution of the record,
    a reader option.

    The resolution is used unless it is missing or a placeholder (72 or
    96 dpi). A pixel size or a resolution given when the image is read is
    assigned as transformations instead.
    """

    @smartproperty(cache=True)
    def data(self) -> tx.Optional[ArrayProtocol]:
        """The image data, assigned or decoded from `rawdata`.

        An array assigned to the image is returned as it is, and stays
        pending until [`consolidate`][] or a write encodes it. Otherwise
        the frame is decoded from `rawdata` on first access and cached.
        Data decoded from a file is read-only, because changing it in
        place would not change `rawdata`, which is what the image
        writes. To change the pixels, assign `data`. Assigning `None`
        decodes `rawdata` again.
        """
        if self.rawdata is None:
            return None
        return _image_to_model(self.rawdata)

    @smartproperty(cache=True)
    def transformations(self) -> tx.List[Transformation]:
        """Pixel-to-physical transformations, decoded from the record.

        The list is decoded on first access and cached, apart from an
        assigned list, so `img.transformation is img.transformation`,
        and a change made to the list in place, such as an appended
        transformation, is kept. Assigning other metadata drops the
        decoded list, so that it is decoded from the new record, and keeps
        an assigned list. A change made in place to a decoded list is
        therefore lost when other metadata is assigned. An image without
        metadata has no transformations.

        Assigning a list replaces the stored one. In both cases, the list
        is compared at [`consolidate`][], and by a write, with the list
        that the record decodes to, exactly. An empty list means that the
        image has no geometry, and the image is then written as an image
        built from data alone. Assigning `None` decodes the record again.
        """
        return _record_transformations(self.metadata, self.palette, self.dpi)

    @property
    def shape(self) -> tx.Tuple[int, ...]:
        """The shape of the data, read without decoding the pixels.

        The shape is that of the assigned data, or else that of `rawdata` with
        its rows and columns swapped, since the lazy array knows its shape
        from the record. An image without data raises as
        [`SingleScaleImage`][] does.
        """
        if self._data is not None:
            return tuple(int(d) for d in self._data.shape)
        if self.rawdata is None:
            return self.data.shape
        shape = tuple(int(d) for d in self.rawdata.shape)
        return shape[1::-1] + shape[2:]

    # --- reading ------------------------------------------------------

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that a file holds an image that Pillow reads.

        A format that is recognised by its magic number scores `LIKELY`. TIFF
        and JPEG 2000 score only `WEAK`, because the dedicated TIFF and JPEG
        2000 readers read them better. The stream is tested with
        [`PillowRaw.sniff_fileobj`][].

        Parameters
        ----------
        file : IO
            The stream, open in binary mode.
        error : bool or type[Exception], default=False
            Whether to raise an error when Pillow does not recognise the
            content.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return PillowRaw.sniff_fileobj(file, error=error)

    @classmethod
    def sniff_bytes(
        cls,
        content: path.BinaryContentLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that bytes hold an image that Pillow reads.

        Parameters
        ----------
        content : bytes
            The content of a file, or its first bytes.
        error : bool or type[Exception], default=False
            Whether to raise an error when Pillow does not recognise the
            content.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return cls.sniff_fileobj(BytesIO(content), error=error, **kwargs)

    @classmethod
    def from_filename(
        cls,
        filename: path.FilenameLike,
        frame: int = 0,
        palette: bool = True,
        dpi: _DpiLike = None,
        pixel_size: tx.Any = None,
        unit: tx.Any = None,
        **kwargs,
    ) -> tx.Self:
        """Read an image from the path of a raster file.

        The record is read from the header, and the lazy array decodes the
        frame when it is needed. The options are those of
        [`from_fileobj`][].

        Parameters
        ----------
        filename : path-like
            The path of the file.
        frame : int, default=0
            The frame of a multi-frame file. A negative index counts from
            the end.
        palette : bool, default=True
            Whether palette colours are looked up or kept as indices.
        dpi : bool or float or tuple of float, optional
            The resolution in dots per inch (see [`from_fileobj`][]).
        pixel_size : float or Sequence[float] or Mapping[str, float], optional
            The pixel size, which overrides any resolution.
        unit : str or Unit, optional
            The unit of `pixel_size`, or of a size derived from a
            resolution.
        **kwargs : Any
            Refused, since the reader has no other option.

        Returns
        -------
        PillowImage
            The image, whose frame is decoded lazily.

        Raises
        ------
        TypeError
            If an unknown option is given.
        ParserExistsError
            If the file does not exist.
        ParserContentError
            If Pillow cannot read the file.
        IndexError
            If the file has no such frame.
        PIL.Image.DecompressionBombError
            If the image exceeds the limit set by `PIL.Image.MAX_IMAGE_PIXELS`.
        """
        _no_options(cls, kwargs)
        record, proxy = read_pillow_image(filename, frame, palette)
        return _read_image(cls, record, proxy, dpi, pixel_size, unit)

    @classmethod
    def from_fileobj(
        cls,
        file: tx.IO,
        frame: int = 0,
        palette: bool = True,
        dpi: _DpiLike = None,
        pixel_size: tx.Any = None,
        unit: tx.Any = None,
        **kwargs,
    ) -> tx.Self:
        """Read an image from an open file.

        The record is read at once, and the lazy array decodes the frame
        from the stream when it is needed, so the stream must stay open
        while the data may be read. Pillow reads the file from the start
        of the stream, and the position of the stream is restored.

        Parameters
        ----------
        file : IO
            A binary file open for reading, which is not closed.
        frame : int, default=0
            The frame of a multi-frame file, the first by default. A
            negative index counts from the end.
        palette : bool, default=True
            Whether palette colours are looked up (the default) or kept as
            indices.
        dpi : bool or float or tuple of float, optional
            The resolution in dots per inch, which makes pixels `25.4 / dpi`
            millimetres wide. By default, the resolution of the file is
            ignored, since it usually describes a screen or a printer. `True`
            uses the resolution of the file unless it is absent or a
            placeholder (72 or 96 dpi), and a number or an `(x, y)` pair
            replaces it.
        pixel_size : float or Sequence[float] or Mapping[str, float], optional
            The pixel size, which overrides any resolution.
        unit : str or Unit, optional
            The unit of `pixel_size`, or the unit to which a size derived from
            a resolution is converted. Millimetres by default.
        **kwargs : Any
            Refused, since the reader has no other option.

        Returns
        -------
        PillowImage
            The image, whose frame is decoded lazily.

        Raises
        ------
        ParserContentError
            If Pillow cannot read the file.
        IndexError
            If the file has no such frame.
        PIL.Image.DecompressionBombError
            If the image exceeds the limit set by `PIL.Image.MAX_IMAGE_PIXELS`.
        TypeError
            If an unknown option is given.
        """
        _no_options(cls, kwargs)
        record, proxy = read_pillow_image(file, frame, palette)
        return _read_image(cls, record, proxy, dpi, pixel_size, unit)

    # --- writing ------------------------------------------------------

    def consolidate(
        self,
        *,
        inplace: bool = False,
        format: tx.Optional[str] = None,
        dpi: _DpiLike = None,
        on_loss: tx.Optional[OnLoss] = None,
        **options,
    ) -> "PillowImage":
        """Encode the pending data and geometry into `rawdata` and the record.

        Consolidating makes `rawdata` and the record describe the image, as a
        file written from it would. An image with a record, whose lazy
        array still reads its file with that record, with no pending data
        and the transformations of its record or none read, consolidated
        in the format of its file without other options, is left as it is:
        the copy shares `rawdata` and the metadata, and it is written as
        the bytes of its file. The transformations that the image holds,
        read or assigned, are compared with those that the record
        decodes to, exactly.

        Otherwise the image is encoded with Pillow, in a fixed order.

        1. The pixels are the assigned data, or else the frame of
           `rawdata`. The data must have two spatial axes, where any
           other spatial axis is a singleton, and at most one channel
           axis, as the axes of the input of the preferred
           transformation say. The type of the data is stored as it is
           (see [`brainhops.io.images.pillow`][]).
        2. The resolution is `dpi` when it is given. Otherwise it is
           computed from the preferred transformation, when that
           transformation scales onto axes with a unit of length, or else
           it is the resolution of the record, unless the image has no
           transformation.
        3. The ICC profile and the EXIF block of the record are written,
           in the formats that store them, unless the options give them.
           The other keys of `info` are not written.

        The bytes that Pillow writes are then read again, so the copy
        holds what the file holds: its record is the header of the new
        file and its `rawdata` is a lazy array of the new bytes, which
        reads its frame 0. Nothing is pending afterwards, so the
        transformations are decoded from the new record. The record held
        before is never changed.

        The metadata is consolidated first (step 2 of the precedence on
        write), with the fields given by keyword. The record of a raster
        image holds no field of the vocabulary yet, so a field that is
        set is reported as lost, once, by `on_loss`.

        Parameters
        ----------
        inplace : bool, default=False
            Whether to change this image rather than a copy.
        format : str, optional
            The Pillow name of the format, such as `"PNG"`. By default, the
            image is written in the format that it was read from (JPEG for
            MPO) if Pillow can write that format, and in PNG otherwise.
        dpi : bool or float or tuple of float, optional
            The resolution recorded by PNG, JPEG, BMP and TIFF. By default
            it is chosen as described above. `False` records none, and a
            number or an `(x, y)` pair is recorded as is.
        on_loss : {"ignore", "warn", "raise"} or ConversionReport, optional
            What to do with the metadata that the record cannot hold. By
            default, the policy in effect.
        **options : Any
            Options passed to `Image.save`, such as `quality=95` for JPEG.
            Only the options that Pillow reads for the format are taken.
            An option named after a field of the metadata overrides that
            field instead.

        Returns
        -------
        PillowImage
            The consolidated copy, or this image with `inplace=True`.

        Raises
        ------
        TypeError
            If a keyword names no field of the metadata.
        MetadataLossError
            If metadata is lost under the `"raise"` policy.
        TypeError
            If an option is not one that Pillow reads for the format.
        WriterError
            If there is no data, if the data has more than two spatial
            axes, or a type that Pillow or the format does not store (such
            as `float64`, or `uint16` in JPEG), or if `frame` or `palette`
            differ from those with which `rawdata` reads its file.
        """
        fields = {
            name: options.pop(name) for name in list(options) if name in FIELDS
        }
        report = ConversionReport(source="PillowImage", target="PillowImage")
        requested = requested_fields(self.metadata, fields)
        metadata = metadata_to_write(
            self.metadata, PillowMetadata, fields, report
        )
        apply_loss_policy(report, on_loss, stacklevel=2)
        if not requested:
            metadata = self.metadata
        held = None if self.metadata is None else self.metadata.raw
        target = _target_format(format, held)
        _check_options(target, options)
        _check_reading(self)
        # The record is kept when the transformations were never read, or
        # are the same as those that it decodes to.
        transformations = held_transformations(self)
        kept = transformations is None or same_transformations(
            transformations,
            _record_transformations(self.metadata, self.palette, self.dpi),
        )
        if (
            self._data is None
            and kept
            and not options
            and (dpi is None or dpi is True)
            and held is not None
            and isinstance(self.rawdata, _PillowProxy)
            and self.rawdata.record is held
            and _same_format(format, held)
        ):
            if not inplace:
                return replace(self, transformations=None, metadata=metadata)
            self.transformations = None
            self.metadata = metadata
            return self
        content = _encoded(self, target, dpi, options)
        record, raw = read_pillow_image(BytesIO(content), 0, self.palette)
        if metadata is None:
            metadata = PillowMetadata.from_raw(record)
        else:
            metadata = replace(metadata, raw=record)
        if not inplace:
            return replace(
                self,
                data=None,
                transformations=None,
                rawdata=raw,
                metadata=metadata,
                frame=0,
            )
        self.rawdata = raw
        self.metadata = metadata
        self.frame = 0
        self.data = None
        self.transformations = None
        return self

    def to_bytes(
        self,
        format: tx.Optional[str] = None,
        dpi: _DpiLike = None,
        **options,
    ) -> bytes:
        """Encode the image in a raster format.

        The image is consolidated into a copy with the options, as by
        [`consolidate`][], and the bytes of the copy are returned. An
        unchanged image read from a file gives the bytes of that file. The
        image itself is not changed.

        Parameters
        ----------
        format : str, optional
            The Pillow name of the format, such as `"PNG"`. By default, the
            image is written in the format that it was read from (JPEG for
            MPO) if Pillow can write that format, and in PNG otherwise.
        dpi : bool or float or tuple of float, optional
            The resolution recorded by PNG, JPEG, BMP and TIFF (see
            [`consolidate`][]).
        **options : Any
            Options passed to `Image.save`, such as `quality=95` for JPEG.

        Returns
        -------
        bytes
            The content of the file.

        Raises
        ------
        TypeError
            If an option is not one that Pillow reads for the format.
        WriterError
            If the image cannot be stored (see [`consolidate`][]). Nothing
            is converted silently.
        """
        if "inplace" in options:
            raise TypeError("A writer does not take the option inplace.")
        image = self.consolidate(format=format, dpi=dpi, **options)
        return image.rawdata.read_stored()

    def to_fileobj(self, file: tx.IO, **kwargs) -> None:
        """Write the image to an open binary file, in the `format` given or the
        one that the name of the file calls for (see `to_bytes`).

        Parameters
        ----------
        file : IO
            The stream, open in binary mode.
        **kwargs : Any
            The options of [`to_bytes`][].
        """
        if kwargs.get("format") is None:
            kwargs["format"] = format_for_name(_to_filename(file))
        file.write(self.to_bytes(**kwargs))

    def to_filename(self, filename: path.FilenameLike, **kwargs) -> None:
        """Write the image to a file, in the format its extension calls for.

        The image is consolidated into a copy with the options, as by
        [`consolidate`][], which is written, and the image itself is not
        changed. The image is encoded before the file is written, so an
        image that cannot be stored leaves no file behind, and a local file
        is replaced at once. When the lazy array of the image reads the
        file that is written, the image is consolidated in place instead,
        so that it holds what the file holds (#426), and its lazy array
        reads the new file.

        Parameters
        ----------
        filename : path-like
            The path of the file.
        **kwargs : Any
            The options of [`to_bytes`][].

        Raises
        ------
        WriterError
            If neither the extension nor `format` names a format, or if the
            image cannot be stored.
        """
        if isinstance(filename, str):
            filename = path.Path(filename)
        if "inplace" in kwargs:
            raise TypeError("A writer does not take the option inplace.")
        if kwargs.get("format") is None:
            fmt = format_for_name(filename)
            if fmt is None:
                raise WriterError(
                    f"Cannot tell which format to write {str(filename)!r} "
                    f"in from its extension. Pass format=, such as "
                    f"format='PNG'."
                )
            kwargs["format"] = fmt
        reading = reads_from(self.rawdata, filename)
        image = self.consolidate(inplace=reading, **kwargs)
        write_file(filename, image.rawdata.read_stored())
        if reading:
            # The image holds what the file holds, and its lazy array reads
            # the new file, since the old lazy array may read other bytes.
            self.rawdata = _PillowProxy(
                filename, self.metadata.raw, self.palette
            )
            self.data = None


# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


def _no_options(cls: type, kwargs: tx.Mapping[str, tx.Any]) -> None:
    """Refuse the options that a reader does not know."""
    if kwargs:
        raise TypeError(
            f"{cls.__name__} does not take the option(s) "
            f"{', '.join(sorted(kwargs))}."
        )


def _read_image(
    cls: type,
    record: PillowRaw,
    proxy: _PillowProxy,
    dpi: _DpiLike,
    pixel_size: tx.Any,
    unit: tx.Any,
) -> PillowImage:
    """Build an image from a record, its lazy array and the reader options.

    The resolution of the record gives the pixel size when `dpi` is
    `True`, which the image keeps as an option. A resolution, a pixel size
    or a unit given to the reader are assigned as transformations, so
    that they are compared with those of the record at consolidation.
    """
    transformations = None
    if (dpi is not None and not isinstance(dpi, bool)) or (
        pixel_size is not None or unit is not None
    ):
        transformations = _decoded_transformations(
            record, proxy.palette, dpi, pixel_size, unit
        )
    return cls(
        rawdata=proxy,
        metadata=PillowMetadata.from_raw(record),
        transformations=transformations,
        frame=record.frame,
        palette=proxy.palette,
        dpi=dpi is True,
    )


def _record_transformations(
    metadata: tx.Optional[PillowMetadata], palette: bool, dpi: bool
) -> tx.List[Transformation]:
    """Decode the transformations of the record of some metadata.

    Metadata without a record gives no transformation.
    """
    if metadata is None or metadata.raw is None:
        return []
    return _decoded_transformations(metadata.raw, palette, dpi, None, None)


def _decoded_transformations(
    record: PillowRaw,
    palette: bool,
    dpi: _DpiLike,
    pixel_size: tx.Any,
    unit: tx.Any,
) -> tx.List[Transformation]:
    """Return the pixel-to-physical scaling of a frame.

    The axes are those of the decoded frame, `x`, `y` and, when a pixel
    has several samples, `c`. The pixel size comes from `pixel_size`, or
    else from the resolution: the resolution of the record when `dpi` is
    `True` and the record has one that is not a placeholder, or `dpi`
    itself when it is a number or a pair.
    """
    shape, _ = record.layout(palette)
    stored = raster.storage_axes("YXS" if len(shape) == 3 else "YX")
    axes = [stored[i] for i in raster.canonical_permutation(stored)]
    resolution = None
    if dpi is True:
        if not raster.is_default_dpi(record.dpi):
            resolution = record.dpi
    elif dpi is not None and dpi is not False:
        resolution = _dpi_pair(dpi)
    sizes: tx.Dict[str, raster.AxisScale] = {}
    if resolution is not None:
        names = [axis.name for axis in raster.spatial_axes(axes)]
        sizes = {
            name: (raster.dpi_to_size(value), "mm")
            for name, value in zip(names, resolution)
        }
    scales = raster.resolve_pixel_size(
        axes, sizes, pixel_size=pixel_size, unit=unit
    )
    return raster.raster_transformations(axes, scales)


# ----------------------------------------------------------------------
#   WRITING
# ----------------------------------------------------------------------


def _target_format(
    format: tx.Optional[str], held: tx.Optional[PillowRaw]
) -> str:
    """Return the format in which an image is encoded.

    The format is `format` when it is given, and otherwise the format of
    the record if Pillow writes it, where MPO is written as JPEG, or else
    PNG.
    """
    if format is not None:
        return str(format).upper()
    fmt = None if held is None else held.format
    if fmt == "MPO":
        fmt = "JPEG"
    if fmt is None or not can_write(fmt):
        return _DEFAULT_FORMAT
    return str(fmt).upper()


def _same_format(format: tx.Optional[str], held: PillowRaw) -> bool:
    """Tell whether writing in `format` keeps the format of the record.

    No format keeps it. An MPO file is a JPEG file whose first picture is
    what a JPEG reader shows, so JPEG keeps the format of an MPO record.
    """
    if format is None:
        return True
    fmt = str(format).upper()
    return fmt == held.format or (fmt == "JPEG" and held.format == "MPO")


def _check_options(fmt: str, options: tx.Mapping[str, tx.Any]) -> None:
    """Refuse the options that Pillow does not read for a format.

    Raises
    ------
    TypeError
        If an option is not listed for the format in `_SAVE_OPTIONS`.
    """
    taken = _SAVE_OPTIONS.get(fmt, frozenset())
    unknown = sorted(set(options) - taken)
    if unknown:
        listed = ", ".join(sorted(taken)) or "none"
        raise TypeError(
            f"Pillow does not read the option(s) {', '.join(unknown)} when "
            f"it writes {fmt}, so the writer refuses them. The options of "
            f"{fmt} are: {listed}."
        )


def _check_reading(image: PillowImage) -> None:
    """Refuse reader options that differ from those of the lazy array.

    Raises
    ------
    WriterError
        If `frame` or `palette` differ from those with which `rawdata`
        reads its file.
    """
    raw = image.rawdata
    if not isinstance(raw, _PillowProxy):
        return
    if image.frame != raw.record.frame or image.palette != raw.palette:
        raise WriterError(
            f"The image says that it reads frame {image.frame} with "
            f"palette={image.palette}, but its lazy array reads frame "
            f"{raw.record.frame} with palette={raw.palette}. The reader "
            f"options cannot be changed after a read: read the file again "
            f"with the options."
        )


def _encoded(
    image: PillowImage,
    fmt: str,
    dpi: _DpiLike,
    options: tx.Mapping[str, tx.Any],
) -> bytes:
    """Return the bytes that Pillow writes for an image, as
    [`PillowImage.consolidate`][] describes.

    The data and the transformations are read without being stored, so
    that a write leaves the image as it was. The transformations are those
    that the image holds, assigned or decoded and maybe changed in place,
    or else those that the record decodes to.
    """
    if image._data is not None:
        data = image._data
    elif image.rawdata is not None:
        data = _image_to_model(image.rawdata)
    else:
        raise WriterError(
            "This image has no data, so there is nothing to write."
        )
    transformations = held_transformations(image)
    if transformations is None:
        transformations = _record_transformations(
            image.metadata, image.palette, image.dpi
        )
    xform = transformations[-1] if transformations else None
    storage, sizes = raster.planar_storage(data, xform)

    held = None if image.metadata is None else image.metadata.raw
    resolution = None
    if dpi is None or dpi is True:
        if sizes is not None:
            resolution = (
                raster.size_to_dpi(sizes["x"]),
                raster.size_to_dpi(sizes["y"]),
            )
        elif transformations and held is not None:
            resolution = held.dpi
    elif dpi is not False:
        resolution = _dpi_pair(dpi)

    options = dict(options)
    taken = _SAVE_OPTIONS.get(fmt, frozenset())
    info = {} if held is None else held.info
    for key in _ROUND_TRIP_INFO:
        if key in taken and info.get(key) and key not in options:
            options[key] = info[key]
    return encode_pillow(
        array_to_pillow(storage), fmt, dpi=resolution, **options
    )


def _dpi_pair(dpi: tx.Any) -> tx.Tuple[float, float]:
    """Convert a resolution, one number or an `(x, y)` pair, to a pair of
    floats.
    """
    if isinstance(dpi, (Number, np.number)):
        return float(dpi), float(dpi)
    values = [float(v) for v in np.ravel(dpi)]
    if len(values) == 1:
        values = values * 2
    if len(values) != 2:
        raise ValueError(
            f"A resolution is one number or an (x, y) pair, not {dpi!r}."
        )
    return values[0], values[1]
