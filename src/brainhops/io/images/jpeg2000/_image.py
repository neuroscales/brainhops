"""The JPEG 2000 image format: one resolution level of a file."""

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import Inherit, NoRepr, replace

# internals
from brainhops._core import path
from brainhops._core.properties import smartproperty
from brainhops._core.proxy import ArrayProxy
from brainhops._core.typing import ArrayProtocol
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.metadata import ConversionReport
from brainhops.datamodel.metadata._report import (
    apply_loss_policy,
    takes_error_policy,
)
from brainhops.datamodel.metadata._vocabulary import FIELDS
from brainhops.datamodel.transformations import Affine, Transformation
from brainhops.io.base._base import register_format
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
from ._boxes import rewrite
from ._codec import default_resolutions, encode_jpeg2000, encoder_options
from ._data import _Jpeg2000Proxy, _Jpeg2000Source, read_jpeg2000
from ._metadata import Jpeg2000Metadata
from ._raw import (
    Jpeg2000Raw,
    check_layout,
    read_record,
    with_container,
)
from ._views import _image_to_model

_EXTENSIONS = (".jp2", ".jpx", ".jpf", ".j2k", ".j2c", ".jpc")
"""The extensions of JP2 files, then of raw codestreams."""

_CODESTREAM_EXTENSIONS = (".j2k", ".j2c", ".jpc")
"""The extensions of raw codestreams."""

_HINTS = ("jpeg2000", "jp2", "j2k", "j2c")
"""The format hints of JPEG 2000 images."""


@register_format
class Jpeg2000Image(
    BinaryFileReader, BinaryFileWriter, ImageFormat, SingleScaleImage
):
    """One resolution level of a JPEG 2000 image.

    A JPEG 2000 image holds everything that its file says beside the
    coded image as [`Jpeg2000Metadata`][], whose record is a
    [`Jpeg2000Raw`][], and the samples of one resolution level in
    `rawdata`. When the image is read from a file, `rawdata` is a lazy
    array, which Pillow decodes only when the samples are needed, and
    `data` returns that lazy array. The data is indexed `(x, y)`, or
    `(x, y, c)` for several samples per pixel, in Fortran order. Level 0
    is the full resolution, and level `r` is reduced by `2**r`.

    The transformations are decoded from the record on first access,
    unless other transformations are assigned. The only transformation is
    a [`Scaling`][brainhops.datamodel.transformations.Scaling] from the
    pixels to a `"physical"` system, by the pixel size of the level, with
    a translation for a reduced level whose first pixel is not centred on
    the first pixel of the full resolution.

    `rawdata` and the record describe the file as it was read, or as the image
    was last consolidated. Data that are assigned are pending until
    [`consolidate`][] encodes them into a copy, or until the image is
    written, which writes such a copy and leaves the image as it was. The
    transformations, read or assigned, are compared at [`consolidate`][]
    with those that the record decodes to, and the resolutions of the
    record are kept when they are the same. A full-resolution image that
    is read and written again without changes gives the same bytes, and a
    file whose comments, resolutions or boxes change is written without
    encoding its image again.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = _EXTENSIONS
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = _HINTS

    rawdata: NoRepr[tx.Optional[ArrayProtocol]] = None
    """The samples as Pillow decodes them, or `None` without data.

    After a read, `rawdata` is a lazy array of one resolution level,
    indexed `(rows, columns)` or `(rows, columns, samples)`. `rawdata`
    describes the file as it was read, or as the image was last
    consolidated, so assigning `data` leaves it as it is.
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
    [`SingleScaleImage`][] to store it under this private name. The
    constructor still takes `transformations` in the same place, and the
    property of the same name reads it. Only an assigned list, even an
    empty one, is stored here. `None`, the default, means that no list
    was assigned, so the property decodes the list of the record and
    caches it apart, and assigning other metadata drops that cache.
    """

    _metadata: NoRepr[tx.Optional[Jpeg2000Metadata]] = Inherit(
        "metadata", default=None
    )

    metadata = smartproperty("metadata", invalidates="transformations")
    """The metadata of the file, which holds its record, or `None`.

    An image built from data has no metadata. Assigning other metadata
    drops the transformations decoded from the previous record, so that
    they are decoded from the new one, and keeps the transformations that
    were assigned.
    """

    @smartproperty(cache=True, invalidated_by="rawdata")
    def data(self) -> tx.Optional[ArrayProtocol]:
        """The image data, assigned or decoded from `rawdata`.

        An array assigned to the image is returned as it is, and stays
        pending until [`consolidate`][] or a write encodes it.
        Otherwise, the data are the lazy array of the file, which reads
        its values when a computation needs them and refuses an edit in
        place. To change the pixels, assign `data`, and assigning `None`
        gives the lazy array of the file again.
        """
        if self.rawdata is None:
            return None
        return _image_to_model(self.rawdata)

    @smartproperty(cache=True)
    def transformations(self) -> tx.List[Transformation]:
        """Pixel-to-world transformations, decoded from the record.

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
        image has no pixel size, which is then not written. Assigning
        `None` decodes the record again.
        """
        return _record_transformations(self.metadata, self.level)

    @property
    def level(self) -> int:
        """The resolution level that `rawdata` reads, where 0 is the full one.

        An image whose `rawdata` is not a lazy array of a file holds the full
        resolution of the file that it would write.
        """
        raw = self.rawdata
        stored = raw.root if isinstance(raw, ArrayProxy) else raw
        if isinstance(stored, _Jpeg2000Proxy):
            return stored.level
        return 0

    @property
    def shape(self) -> tx.Tuple[int, ...]:
        """The shape of the data, read without decoding the samples.

        The shape is that of the assigned data, or else of `rawdata` with its
        first two axes swapped: a lazy array knows its shape from the
        record. An image without data raises as [`SingleScaleImage`][]
        does.
        """
        if self._data is not None:
            return tuple(int(d) for d in self._data.shape)
        if self.rawdata is None:
            return self.data.shape
        shape = tuple(int(d) for d in self.rawdata.shape)
        return (shape[1], shape[0]) + shape[2:]

    # --- reading ------------------------------------------------------

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that a stream holds a JPEG 2000 image.

        The stream is tested with [`Jpeg2000Raw.sniff_fileobj`][], which
        scores a JP2 file or a codestream `LIKELY`. A file with several
        resolution levels is claimed more confidently by
        [`Jpeg2000MultiScaleImage`][brainhops.io.images.jpeg2000.Jpeg2000MultiScaleImage],
        unless a `level` is asked for.

        Parameters
        ----------
        file : IO
            The stream, whose position is restored.
        error : bool or type[Exception], default=False
            Whether to raise an error when the stream holds no JPEG 2000
            header.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return Jpeg2000Raw.sniff_fileobj(file, error=error)

    @classmethod
    def from_filename(
        cls,
        filename: path.FilenameLike,
        level: tx.Optional[int] = None,
        pixel_size: tx.Any = None,
        unit: tx.Any = None,
        **kwargs,
    ) -> "Jpeg2000Image":
        """Read one resolution level from the path of a JPEG 2000 file.

        The record is read from the header. A local file is decoded by
        name when the samples are needed, and any other file is read into
        memory.

        Parameters
        ----------
        filename : path-like
            The path of the file.
        level : int, optional
            The resolution level to read. Level 0, the default, is the full
            resolution, level `r` is reduced by `2**r`, and negative values
            count from the coarsest level.
        pixel_size : float or sequence of float or mapping, optional
            The pixel size of the full resolution, which overrides the one
            of the file: one size, `(x, y)`, or a mapping by axis name. The
            transformations that it gives are assigned to the image.
        unit : str or Unit, optional
            The unit of `pixel_size`, or the unit to which the pixel size
            of the file, in millimetres, is converted.
        **kwargs : Any
            Refused, since JPEG 2000 has no other option.

        Returns
        -------
        Jpeg2000Image
            The image, whose samples are decoded lazily.

        Raises
        ------
        TypeError
            If another option is given.
        ParserExistsError
            If the path does not exist.
        ParserContentError
            If the file is not a JPEG 2000 file.
        IndexError
            If the file has no such level.
        """
        _no_options(cls, kwargs)
        raw, source = read_jpeg2000(filename)
        return _read_image(cls, raw, source, level, pixel_size, unit)

    @classmethod
    def from_fileobj(
        cls,
        file: tx.BinaryIO,
        level: tx.Optional[int] = None,
        pixel_size: tx.Any = None,
        unit: tx.Any = None,
        **kwargs,
    ) -> "Jpeg2000Image":
        """Read one resolution level from an open JPEG 2000 stream.

        The stream is read into memory, and its position is restored. The
        options are those of [`from_filename`][].

        Parameters
        ----------
        file : BinaryIO
            The stream, open in binary mode.
        level : int, optional
            The resolution level to read, as in [`from_filename`][].
        pixel_size : float or sequence of float or mapping, optional
            The pixel size of the full resolution, as in
            [`from_filename`][].
        unit : str or Unit, optional
            The unit of the pixel size, as in [`from_filename`][].
        **kwargs : Any
            Refused, since JPEG 2000 has no other option.

        Returns
        -------
        Jpeg2000Image
            The image, whose samples are decoded lazily.

        Raises
        ------
        TypeError
            If another option is given.
        ParserContentError
            If the stream is not a JPEG 2000 file.
        IndexError
            If the file has no such level.
        """
        _no_options(cls, kwargs)
        raw, source = read_jpeg2000(file)
        return _read_image(cls, raw, source, level, pixel_size, unit)

    # --- writing ------------------------------------------------------

    def _consolidate(
        self,
        *,
        inplace: bool = False,
        container: tx.Optional[str] = None,
        num_resolutions: tx.Optional[int] = None,
        comment: tx.Optional[str] = None,
        **options,
    ) -> "Jpeg2000Image":
        """Encode the pending data and geometry into `rawdata` and the record.

        Consolidating makes `rawdata` and the record describe the image, as a
        file written from it would. The record of the metadata is the base
        of the new file, which is built in one of two ways.

        1. When the image holds the full resolution of the file that it
           read, its data are not pending, and no option of the encoder is
           given, the coded image of the file is copied as it is. Only the
           comments, the resolution box, the XML and UUID boxes and the
           container are written again, from the record. A record whose
           other fields differ from those of the file is refused.
        2. Otherwise the pixels are encoded again with Pillow, losslessly
           unless the options ask otherwise, with as many resolution levels
           as the file had below the level that the image reads, or six,
           as long as the coarsest level has a pixel. The comments and the
           boxes of the record are written into the new file.

        In both cases, the capture and display resolutions of the record
        are kept when the transformations are those that the record
        decodes to. Otherwise the pixel size of the preferred
        transformation becomes the capture resolution, and the display
        resolution of the record is kept only when it does not decide the
        pixel size. A raw codestream stores no resolution and no box, so
        it stores no pixel size.

        When the image holds the full resolution of the file that it read,
        nothing is pending, the transformations are those of the record or
        none were read, and no option is given, the image is left as it
        is: the copy shares `rawdata` and the metadata. Otherwise the new file
        is held in memory, `rawdata` decodes it, and new metadata holds
        its record. The record held before is never changed. Nothing is
        pending afterwards, so the transformations are decoded from the
        new record, with the precision of the file.

        The metadata is consolidated first (step 2 of the precedence on
        write), with the fields given by keyword. The record of a JPEG
        2000 file holds no field of the vocabulary yet, so a field that
        is set is reported as lost, once, under the error policy of the
        call.

        Parameters
        ----------
        inplace : bool, default=False
            Whether to change this image rather than a copy.
        container : {"jp2", "j2k"}, optional
            The container of the file: a JP2 file, or a raw codestream. By
            default, the container of the record, or JP2 for an image
            without metadata.
        num_resolutions : int, optional
            The number of resolution levels to encode, which encodes the
            pixels again.
        comment : str, optional
            The comment of the file, in place of those of the record.
        **options : Any
            Options of the JPEG 2000 writer of Pillow, which encode the
            pixels again, such as `irreversible=True,
            quality_mode="rates", quality_layers=[20]` for a lossy file, or
            `tile_size=(256, 256)`. The options `signed`, `no_jp2` and
            `dpi` are refused, because the data, `container` and the
            transformations decide them.
            An option named after a field of the metadata overrides that
            field instead.

        Returns
        -------
        Jpeg2000Image
            The consolidated copy, or this image with `inplace=True`.

        Raises
        ------
        TypeError
            If a keyword names no field of the metadata.
        MetadataLossError
            If metadata is lost under the `"raise"` policy.
        TypeError
            If an unknown option is given.
        WriterError
            If there is no data, the data are not a picture of two spatial
            axes and at most one channel axis, Pillow cannot store their
            type, the container is unknown, or the record changes a field
            that the coded image decides.
        """
        fields = {
            name: options.pop(name) for name in list(options) if name in FIELDS
        }
        report = ConversionReport(
            source="Jpeg2000Image", target="Jpeg2000Image"
        )
        requested = requested_fields(self.metadata, fields)
        metadata = metadata_to_write(
            self.metadata, Jpeg2000Metadata, fields, report
        )
        apply_loss_policy(report, stacklevel=2)
        if not requested:
            metadata = self.metadata
        options = encoder_options(options)
        held = None if self.metadata is None else self.metadata.raw
        if container is None:
            container = "jp2" if held is None else held.container
        if container not in ("jp2", "j2k"):
            raise WriterError(
                f"A JPEG 2000 file is a JP2 file ('jp2') or a raw codestream "
                f"('j2k'), not {container!r}."
            )
        # The record is kept when the transformations were never read, or
        # are the same as those that it decodes to.
        transformations = held_transformations(self)
        geometry = self.level == 0 and (
            transformations is None
            or same_transformations(
                transformations,
                _record_transformations(self.metadata, self.level),
            )
        )
        copied = (
            self._data is None
            and isinstance(self.rawdata, _Jpeg2000Proxy)
            and self.rawdata.level == 0
            and num_resolutions is None
            and not options
            and not (container == "j2k" and self.rawdata.record.palette)
        )
        if (
            copied
            and geometry
            and held is not None
            and self.rawdata.record is held
            and container == held.container
            and comment is None
        ):
            if not inplace:
                return replace(self, transformations=None, metadata=metadata)
            self.transformations = None
            self.metadata = metadata
            return self
        if copied:
            content = _rewritten(self, held, container, comment, geometry)
        else:
            content = _encoded(
                self,
                held,
                container,
                num_resolutions,
                comment,
                geometry,
                options,
            )
        record = read_record(content)
        raw = _Jpeg2000Proxy(_Jpeg2000Source(content=content), record, 0)
        if metadata is None:
            metadata = Jpeg2000Metadata.from_raw(record)
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

    @takes_error_policy
    def to_bytes(self, **kwargs) -> bytes:
        """Return the JPEG 2000 file of the image.

        The image is consolidated into a copy with the keyword arguments,
        as by [`consolidate`][], and the file that the copy holds is
        returned. An image that is untouched gives the bytes of its file.
        The image itself is not changed.

        Parameters
        ----------
        **kwargs : Any
            The options of [`consolidate`][].

        Returns
        -------
        bytes
            The content of the file.

        Raises
        ------
        WriterError
            If the image cannot be encoded, as in [`consolidate`][].
        """
        return _written(self.consolidate(**kwargs))

    @takes_error_policy
    def to_fileobj(self, file: tx.IO, **kwargs) -> None:
        """Write the JPEG 2000 file of the image to a stream.

        The container follows the extension of the name of the stream,
        when it has one and `container` is not given.

        Parameters
        ----------
        file : IO
            The stream, open in binary mode.
        **kwargs : Any
            The options of [`consolidate`][].
        """
        if kwargs.get("container") is None:
            kwargs["container"] = _container_for_name(path.base_name(file))
        file.write(self.to_bytes(**kwargs))

    @takes_error_policy
    def to_filename(self, filename: path.FilenameLike, **kwargs) -> None:
        """Write the image to a path.

        The file is a raw codestream when its name ends with `.j2k`,
        `.j2c` or `.jpc`, and a JP2 file when it ends with `.jp2`, `.jpx`
        or `.jpf`, unless `container` says otherwise. The other keyword
        arguments are those of [`consolidate`][]. The content is encoded
        before the file is written, and a local file is replaced at once,
        so an image can be written over the file that it was read from.
        The image is then consolidated in place, so that it holds what the
        file holds (#426), and its lazy array reads the new file.

        Parameters
        ----------
        filename : path-like
            The path of the file.
        **kwargs : Any
            The options of [`consolidate`][].
        """
        filename = path.Path(filename)
        if kwargs.get("container") is None:
            kwargs["container"] = _container_for_name(filename)
        if not _reads_file(self.rawdata, filename):
            filename.write_bytes(
                self.to_bytes(**kwargs), atomic=filename.is_local()
            )
            return
        self.consolidate(inplace=True, **kwargs)
        filename.write_bytes(_written(self), atomic=filename.is_local())
        # The image holds what the file holds, and its lazy array reads the
        # new file, since the lazy array of the old file may be another
        # level, or decode another coded image.
        self.rawdata = _Jpeg2000Proxy(
            _Jpeg2000Source(filename=filename),
            self.metadata.raw,
            0,
        )
        self.data = None


# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


def _no_options(cls: type, kwargs: tx.Mapping[str, tx.Any]) -> None:
    """Refuse the options that a reader does not take.

    Raises
    ------
    TypeError
        If an option is given.
    """
    if kwargs:
        raise TypeError(
            f"{cls.__name__} does not take the option(s) "
            f"{', '.join(sorted(kwargs))}."
        )


def _read_image(
    cls: type,
    record: Jpeg2000Raw,
    source: _Jpeg2000Source,
    level: tx.Optional[int],
    pixel_size: tx.Any,
    unit: tx.Any,
) -> Jpeg2000Image:
    """Build the image of one level of a file that was read.

    Raises
    ------
    IndexError
        If the file has no such level.
    """
    n = record.n_levels
    index = 0 if level is None else int(level)
    if not -n <= index < n:
        raise IndexError(
            f"This file has {n} resolution level(s), so it has no level "
            f"{level}."
        )
    index %= n
    raw = _Jpeg2000Proxy(source, record, index)
    metadata = Jpeg2000Metadata.from_raw(record)
    if pixel_size is None and unit is None:
        return cls(rawdata=raw, metadata=metadata)
    xform = level_transformation(record, index, pixel_size, unit)
    return cls(rawdata=raw, metadata=metadata, transformations=[xform])


def _record_transformations(
    metadata: tx.Optional[Jpeg2000Metadata], level: int
) -> tx.List[Transformation]:
    """Decode the transformations of one level from some metadata.

    Metadata without a record gives no transformation.
    """
    if metadata is None or metadata.raw is None:
        return []
    return [level_transformation(metadata.raw, level)]


def level_transformation(
    record: Jpeg2000Raw,
    level: int,
    pixel_size: tx.Any = None,
    unit: tx.Any = None,
) -> Transformation:
    """Return the pixel-to-physical transformation of a resolution level.

    The pixel size of the full resolution comes from the record, unless
    `pixel_size` or `unit` override it, as for every raster image. Level
    `r` is reduced by `f = 2**r`, and its pixel `i` is centred on the
    full-resolution pixel `f * i + s`, where `s` is the start of the level
    ([`Jpeg2000Raw.level_start`][]). The pixel size of the level is `f`
    times that of the full resolution.

    Parameters
    ----------
    record : Jpeg2000Raw
        The record of the file.
    level : int
        The resolution level.
    pixel_size : float or sequence of float or mapping, optional
        The pixel size of the full resolution, which overrides the one of
        the record.
    unit : str or Unit, optional
        The unit of `pixel_size`, or the unit to which the pixel size of
        the record is converted.

    Returns
    -------
    Transformation
        A [`Scaling`][brainhops.datamodel.transformations.Scaling], or an
        [`Affine`][] when the level has a translation.
    """
    codes = "YXS" if record.samples > 1 else "YX"
    axes = raster.storage_axes(codes)
    axes = [axes[i] for i in raster.canonical_permutation(axes)]
    sizes = _record_pixel_size(record)
    metadata = {}
    if sizes is not None:
        metadata = {"x": (sizes[0], "mm"), "y": (sizes[1], "mm")}
    scales = raster.resolve_pixel_size(
        axes, metadata, pixel_size=pixel_size, unit=unit
    )
    if level == 0:
        return raster.raster_transformations(axes, scales)[0]
    factor = 2**level
    offsets = dict(zip("xy", record.level_start(level)))
    level_scales = dict(scales)
    translation = []
    for axis in axes:
        size, size_unit = scales.get(axis.name, (1.0, None))
        if axis.name in offsets:
            level_scales[axis.name] = (size * factor, size_unit)
        translation.append(size * offsets.get(axis.name, 0))
    scaling = raster.raster_transformations(axes, level_scales)[0]
    if not any(translation):
        return scaling
    n = len(axes)
    matrix = np.zeros((n, n + 1))
    matrix[:, :-1] = np.diag(np.asarray(scaling.scale, dtype=float))
    matrix[:, -1] = translation
    return Affine(matrix=matrix, input=scaling.input, output=scaling.output)


def _record_pixel_size(
    record: Jpeg2000Raw,
) -> tx.Optional[tx.Tuple[float, float]]:
    """Return the pixel size `(x, y)` in millimetres that a record gives.

    The size comes from the capture resolution, or else from the display
    resolution. A placeholder, 72 or 96 dots per inch or one per inch, is
    not a size.
    """
    for resolution in (record.capture_resolution, record.display_resolution):
        if resolution is None or _placeholder(resolution):
            continue
        return 1000 / resolution[0], 1000 / resolution[1]
    return None


def _placeholder(resolution: tx.Sequence[float]) -> bool:
    """Tell whether a resolution in grid points per metre is a placeholder."""
    dpi = [value * raster.MM_PER_INCH / 1000 for value in resolution]
    return raster.is_default_dpi(dpi)


# ----------------------------------------------------------------------
#   WRITING
# ----------------------------------------------------------------------


def _resolutions(
    held: tx.Optional[Jpeg2000Raw],
    sizes: tx.Optional[tx.Tuple[float, float]],
) -> tx.Dict[str, tx.Any]:
    """Return the resolutions that encode a pixel size over a record.

    The pixel size, in millimetres, becomes the capture resolution, and an
    unknown size gives none. The display resolution of the record is kept
    when it does not decide the pixel size, which is when the capture
    resolution is known or the display resolution is a placeholder.
    """
    capture = None
    if sizes is not None:
        capture = (1000 / sizes[0], 1000 / sizes[1])
    display = None if held is None else held.display_resolution
    if capture is None and display is not None and not _placeholder(display):
        display = None
    return {"capture_resolution": capture, "display_resolution": display}


def _pixel_size(
    image: Jpeg2000Image, ndim: int
) -> tx.Optional[tx.Tuple[float, float]]:
    """Return the pixel size `(x, y)` in millimetres of an image, if known.

    The size is that of the preferred transformation, whose input names
    the axes, and the read list is used without being stored.
    """
    transformations = held_transformations(image)
    if transformations is None:
        transformations = _record_transformations(image.metadata, image.level)
    xform = transformations[-1] if transformations else None
    axes = raster.image_axes(ndim, getattr(xform, "input", None))
    if axes is None:
        axes = raster.default_axes(ndim, channel=ndim == 3)
    sizes = raster.physical_pixel_size(xform, axes, "mm")
    space = [axis.name for axis in raster.spatial_axes(axes)]
    if sizes is None or len(space) != 2:
        return None
    return sizes[space[0]], sizes[space[1]]


def _rewritten(
    image: Jpeg2000Image,
    held: tx.Optional[Jpeg2000Raw],
    container: str,
    comment: tx.Optional[str],
    geometry: bool,
) -> bytes:
    """Return the file of an image whose coded image is copied.

    The comments, the resolutions, the boxes and the container of the
    record are written over the file that `rawdata` reads.

    Raises
    ------
    WriterError
        If the record changes a field that the coded image decides.
    """
    record = image.rawdata.record if held is None else held
    changes: tx.Dict[str, tx.Any] = {}
    if comment is not None:
        changes["comments"] = (comment,)
    if not geometry:
        changes.update(
            _resolutions(record, _pixel_size(image, image.rawdata.ndim))
        )
    target = with_container(replace(record, **changes), container)
    content = rewrite(image.rawdata.source.read(), target)
    check_layout(target, read_record(content))
    return content


def _encoded(
    image: Jpeg2000Image,
    held: tx.Optional[Jpeg2000Raw],
    container: str,
    num_resolutions: tx.Optional[int],
    comment: tx.Optional[str],
    geometry: bool,
    options: tx.Mapping[str, tx.Any],
) -> bytes:
    """Return the file of an image whose pixels are encoded again.

    The pixels are the assigned data, or else the level that `rawdata`
    reads. The transformations are read without being stored, so that a
    write leaves the image as it was.

    Raises
    ------
    WriterError
        If there is no data, or the data cannot be stored.
    """
    transformations = held_transformations(image)
    if transformations is None:
        transformations = _record_transformations(image.metadata, image.level)
    xform = transformations[-1] if transformations else None
    data = image._data if image._data is not None else image.data
    storage, sizes = raster.planar_storage(data, xform)
    if sizes is not None:
        sizes = (sizes["x"], sizes["y"])
    if num_resolutions is None and held is not None:
        num_resolutions = min(
            held.n_levels - image.level,
            default_resolutions(storage.shape[:2]),
        )
    content = encode_jpeg2000(storage, container, num_resolutions, options)
    record = read_record(content)
    changes: tx.Dict[str, tx.Any] = {}
    if held is not None:
        changes["comments"] = held.comments
        changes["boxes"] = held.boxes
    if comment is not None:
        changes["comments"] = (comment,)
    if geometry and held is not None:
        changes["capture_resolution"] = held.capture_resolution
        changes["display_resolution"] = held.display_resolution
    else:
        changes.update(_resolutions(held, sizes))
    target = with_container(replace(record, **changes), container)
    if target == record:
        return content
    return rewrite(content, target)


def _written(image: Jpeg2000Image) -> bytes:
    """Return the file that a consolidated image holds.

    Raises
    ------
    WriterError
        If the image has no data.
    """
    if not isinstance(image.rawdata, _Jpeg2000Proxy):
        raise WriterError(
            "This image has no data, so there is nothing to write."
        )
    return image.rawdata.source.read()


def _reads_file(raw: tx.Any, file: path.Path) -> bool:
    """Tell whether a lazy array of samples reads a local file by name."""
    raw = raw.root if isinstance(raw, ArrayProxy) else raw
    return isinstance(raw, _Jpeg2000Proxy) and raw.source.is_file(file)


def _container_for_name(name: tx.Any) -> tx.Optional[str]:
    """Return the container that the extension of a file name asks for."""
    if name is None:
        return None
    lower = str(name).lower()
    if lower.endswith(_CODESTREAM_EXTENSIONS):
        return "j2k"
    if lower.endswith(_EXTENSIONS):
        return "jp2"
    return None
