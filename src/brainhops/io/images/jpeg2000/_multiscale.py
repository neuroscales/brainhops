"""The JPEG 2000 image format: every resolution level of a file."""

# stdlib
import os

# dependencies
import typing_extensions as tx
from bagof.magic import Inherit, NoRepr, replace

# internals
from brainhops._core import path
from brainhops._core.path import write_file
from brainhops._core.properties import cached_value, smartproperty
from brainhops._core.proxy import ArrayProxy
from brainhops._core.typing import ArrayProtocol
from brainhops.datamodel.images import MultiScaleImage, SingleScaleImage
from brainhops.datamodel.metadata._report import OnLoss
from brainhops.datamodel.transformations import Transformation
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    BinaryFileReader,
    BinaryFileWriter,
    Confidence,
    WriterError,
)
from brainhops.io.common._geometry import held_transformations
from brainhops.io.images.base import ImageFormat

# this format
from ._data import _Jpeg2000Proxy, _Jpeg2000Source, read_jpeg2000
from ._image import (
    _EXTENSIONS,
    _HINTS,
    Jpeg2000Image,
    _container_for_name,
    _no_options,
    _reads_file,
    _record_transformations,
    _written,
    level_transformation,
)
from ._metadata import Jpeg2000Metadata
from ._raw import Jpeg2000Raw


@register_format
class Jpeg2000MultiScaleImage(
    BinaryFileReader, BinaryFileWriter, ImageFormat, MultiScaleImage
):
    """Every resolution level of a JPEG 2000 image, finest first.

    A JPEG 2000 codestream is a wavelet decomposition, so an image coded
    with `N` decomposition levels can be decoded at `N + 1` resolutions:
    the full one, and reductions by 2, 4, up to `2**N`. A multiscale image
    holds everything that its file says beside the coded image as
    [`Jpeg2000Metadata`][], and the samples of the full resolution in
    `rawdata`, a lazy array that also gives the reduced levels. Its
    levels are [`Jpeg2000Image`][] objects that read the same file,
    decoded from `rawdata` and the record on first access, and each
    level is decoded by Pillow only when its data is first needed.

    Level `r` is reduced by `2**r`, and its pixel `i` is centred on the
    full-resolution pixel `2**r * i`, because the wavelet keeps the even
    samples of each level. Every level maps onto the same `"physical"`
    system, so the transformations of the pyramid itself are empty.

    A pyramid is written from its full-resolution level alone, with as
    many resolution levels as it has. The coarser levels are taken to be
    downsampled versions of the first one, and the codec derives them
    again, so whatever was assigned to them is not written, and their
    shapes are not checked. A pyramid that is read and written again
    without changes gives the same bytes.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = _EXTENSIONS
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = _HINTS

    rawdata: NoRepr[tx.Optional[ArrayProtocol]] = None
    """The full-resolution samples as Pillow decodes them, or `None`.

    After a read, `rawdata` is a lazy array of level 0 of the file, from which
    the levels are decoded. `rawdata` describes the file as it was read, or as
    the pyramid was last consolidated.
    """

    _images: tx.Optional[tx.List[SingleScaleImage]] = Inherit(
        "images", default=None
    )
    """The levels assigned to the pyramid, or `None`.

    The pyramid takes over the `images` field of [`MultiScaleImage`][] to
    store it under this private name. The constructor still takes
    `images` in the same place, and the property of the same name reads
    it. Only an assigned list, even an empty one, is stored here. `None`,
    the default, means that no list was assigned, so the property decodes
    the levels from `rawdata` and the record and caches them apart, and
    assigning other metadata drops that cache.
    """

    _transformations: tx.Optional[tx.List[Transformation]] = Inherit(
        "transformations", default=None
    )
    """The placements assigned to the pyramid, or `None`.

    The pyramid takes over the `transformations` field of
    [`MultiScaleImage`][] in the same way as `images`. Only an assigned
    list is stored here. A file stores no placement of its own, so the
    decoded list, which is cached apart, is empty.
    """

    _metadata: NoRepr[tx.Optional[Jpeg2000Metadata]] = Inherit(
        "metadata", default=None
    )

    metadata = smartproperty(
        "metadata", invalidates=("images", "transformations")
    )
    """The metadata of the file, which holds its record, or `None`.

    A pyramid built from levels has no metadata. Assigning other metadata
    drops the levels and the placements decoded from the previous record,
    so that they are decoded again, and keeps the levels and the
    placements that were assigned.
    """

    @smartproperty(cache=True)
    def images(self) -> tx.List[SingleScaleImage]:
        """The levels of the pyramid, decoded from `rawdata` on first access.

        The levels are [`Jpeg2000Image`][] objects whose `rawdata` reads
        one level of the file and whose metadata is that of the pyramid,
        so decoding them reads nothing. The list is cached, apart from
        an assigned list, so a level changed in place is kept. Assigning
        other metadata drops the decoded list and keeps an assigned one.
        Assigning a list replaces it, and assigning `None` decodes the
        levels again.
        """
        return _record_levels(self.rawdata, self.metadata)

    @smartproperty(cache=True)
    def transformations(self) -> tx.List[Transformation]:
        """The placements of the pyramid, which a file never stores.

        The decoded list is empty, because every level maps onto the
        `"physical"` system, and it is cached apart from an assigned list.
        An assigned list is composed with the transformation of the
        full-resolution level when the pyramid is written. Assigning other
        metadata drops the decoded list and keeps an assigned one.
        Assigning `None` decodes the empty list again.
        """
        return []

    @property
    def shape(self) -> tx.Tuple[int, ...]:
        """The shape of the full-resolution level, read without decoding.

        The levels are read without being stored.

        Raises
        ------
        IndexError
            If the pyramid has no level.
        """
        images = self._images
        if images is None:
            images = cached_value(self, "images")
        if images is None:
            images = _record_levels(self.rawdata, self.metadata)
        return tuple(int(n) for n in images[0].shape)

    # --- reading ------------------------------------------------------

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        level: tx.Optional[int] = None,
        **kwargs,
    ) -> float:
        """Return the confidence that a stream holds a JPEG 2000 pyramid.

        The score is `CERTAIN` when the file has several resolution levels
        and no `level` is asked for, and `NO` otherwise. An image whose
        area has an offset on the reference grid scores `NO` too, because
        Pillow cannot decode its reduced levels, and it is read at full
        resolution by [`Jpeg2000Image`][].

        Parameters
        ----------
        file : IO
            The stream, whose position is restored.
        error : bool or type[Exception], default=False
            Whether to raise an error when the stream holds no JPEG 2000
            header.
        level : int, optional
            The level that the reader is asked for, which a pyramid does
            not take.
        **kwargs : Any
            Other options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        if level is not None:
            return Confidence.NO
        if not Jpeg2000Raw.sniff_fileobj(file, error=error):
            return Confidence.NO
        record = Jpeg2000Raw.from_fileobj(file)
        if record.n_levels > 1 and not any(record.image_offset):
            return Confidence.CERTAIN
        return Confidence.NO

    @classmethod
    def from_filename(
        cls,
        filename: path.FilenameLike,
        pixel_size: tx.Any = None,
        unit: tx.Any = None,
        **kwargs,
    ) -> "Jpeg2000MultiScaleImage":
        """Read every resolution level from the path of a JPEG 2000 file.

        The record is read from the header. A local file is decoded by
        name when the samples of a level are needed, and any other file is
        read into memory.

        Parameters
        ----------
        filename : path-like
            The path of the file.
        pixel_size : float or sequence of float or mapping, optional
            The pixel size of the full resolution, which overrides the one
            of the file: one size, `(x, y)`, or a mapping by axis name. The
            levels are then built with the transformations that it gives.
        unit : str or Unit, optional
            The unit of `pixel_size`, or the unit to which the pixel size
            of the file, in millimetres, is converted.
        **kwargs : Any
            Refused, since a pyramid has no other option.

        Returns
        -------
        Jpeg2000MultiScaleImage
            The pyramid, whose levels are decoded lazily.

        Raises
        ------
        TypeError
            If another option is given.
        ParserExistsError
            If the path does not exist.
        ParserContentError
            If the file is not a JPEG 2000 file.
        """
        _no_options(cls, kwargs)
        raw, source = read_jpeg2000(filename)
        return _read_pyramid(cls, raw, source, pixel_size, unit)

    @classmethod
    def from_fileobj(
        cls,
        file: tx.BinaryIO,
        pixel_size: tx.Any = None,
        unit: tx.Any = None,
        **kwargs,
    ) -> "Jpeg2000MultiScaleImage":
        """Read every resolution level from an open JPEG 2000 stream.

        The stream is read into memory, and its position is restored.

        Parameters
        ----------
        file : BinaryIO
            The stream, open in binary mode.
        pixel_size : float or sequence of float or mapping, optional
            The pixel size of the full resolution, as in
            [`from_filename`][].
        unit : str or Unit, optional
            The unit of the pixel size, as in [`from_filename`][].
        **kwargs : Any
            Refused, since a pyramid has no other option.

        Returns
        -------
        Jpeg2000MultiScaleImage
            The pyramid, whose levels are decoded lazily.

        Raises
        ------
        TypeError
            If another option is given.
        ParserContentError
            If the stream is not a JPEG 2000 file.
        """
        _no_options(cls, kwargs)
        raw, source = read_jpeg2000(file)
        return _read_pyramid(cls, raw, source, pixel_size, unit)

    # --- writing ------------------------------------------------------

    def consolidate(
        self,
        *,
        inplace: bool = False,
        container: tx.Optional[str] = None,
        num_resolutions: tx.Optional[int] = None,
        comment: tx.Optional[str] = None,
        on_loss: tx.Optional[OnLoss] = None,
        **options,
    ) -> "Jpeg2000MultiScaleImage":
        """Encode the full-resolution level and the placements of the pyramid.

        The pyramid is consolidated through its full-resolution level, as
        [`Jpeg2000Image.consolidate`][] describes, with the record of the
        pyramid as the base of the new file. The transformation of that
        level is composed with the preferred placement of the pyramid, if
        it has one. A pyramid of another number of levels than its file is
        encoded again with as many resolution levels as it has. Only the
        first level is written: the coarser levels are taken to be
        downsampled versions of it, and the codec derives them again, so
        their data, transformations and shapes, assigned or not, are
        neither written nor checked.

        A pyramid whose full-resolution level is left as it is by its own
        consolidation, and whose levels and placements are those of its
        file, is left as it is: the copy shares `rawdata` and the
        metadata. Otherwise the new file is held in memory, and
        `rawdata` decodes it. Nothing is pending afterwards, so the
        levels are decoded from the new record.

        The metadata of the pyramid, which its levels share, is
        consolidated by that of the full-resolution level, with the fields
        given by keyword, and a field that the record cannot hold is
        reported as lost, once, by `on_loss`.

        Parameters
        ----------
        inplace : bool, default=False
            Whether to change this pyramid rather than a copy.
        container : {"jp2", "j2k"}, optional
            The container of the file, as for
            [`Jpeg2000Image.consolidate`][].
        num_resolutions : int, optional
            The number of resolution levels to encode, by default the
            number of levels of the pyramid.
        comment : str, optional
            The comment of the file, in place of those of the record.
        on_loss : {"ignore", "warn", "raise"} or ConversionReport, optional
            What to do with the metadata that the record cannot hold. By
            default, the policy in effect.
        **options : Any
            Options of the JPEG 2000 writer of Pillow, and fields of the
            metadata, as for [`Jpeg2000Image.consolidate`][].

        Returns
        -------
        Jpeg2000MultiScaleImage
            The consolidated copy, or this pyramid with `inplace=True`.

        Raises
        ------
        TypeError
            If an unknown option is given.
        MetadataLossError
            If metadata is lost under the `"raise"` policy.
        WriterError
            If the pyramid has no level, or the full-resolution level
            cannot be encoded.
        """
        images = self._images
        if images is None:
            images = cached_value(self, "images")
        if images is None:
            images = _record_levels(self.rawdata, self.metadata)
        base = _base_level(self, images)
        if num_resolutions is None and len(images) != _file_levels(base):
            num_resolutions = len(images)
        level = base.consolidate(
            container=container,
            num_resolutions=num_resolutions,
            comment=comment,
            on_loss=on_loss,
            **options,
        )
        if level.rawdata is self.rawdata and level.metadata is self.metadata:
            if not inplace:
                return replace(self, images=None, transformations=None)
            self.images = None
            self.transformations = None
            return self
        if not inplace:
            return replace(
                self,
                images=None,
                transformations=None,
                rawdata=level.rawdata,
                metadata=level.metadata,
            )
        self.rawdata = level.rawdata
        self.metadata = level.metadata
        self.images = None
        self.transformations = None
        return self

    def to_bytes(self, **kwargs) -> bytes:
        """Return the JPEG 2000 file of the pyramid.

        The pyramid is consolidated into a copy with the keyword
        arguments, as by [`consolidate`][], and the file that the copy
        holds is returned. The pyramid itself is not changed.

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
            If the pyramid cannot be encoded, as in [`consolidate`][].
        """
        return _written(self.consolidate(**kwargs))

    def to_fileobj(self, file: tx.IO, **kwargs) -> None:
        """Write the JPEG 2000 file of the pyramid to a stream.

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

    def to_filename(self, filename: path.FilenameLike, **kwargs) -> None:
        """Write the pyramid to a path.

        The file is a raw codestream when its name ends with one of
        `.j2k`, `.j2c` and `.jpc`, and a JP2 file otherwise, unless
        `container` says otherwise, as for
        [`Jpeg2000Image.to_filename`][]. The content is encoded before
        the file is written, and a local file is replaced at once. A
        pyramid written over the file that it was read from is
        consolidated in place, so that it holds what the file holds
        (#426), and its lazy array reads the new file.

        Parameters
        ----------
        filename : path-like
            The path of the file.
        **kwargs : Any
            The options of [`consolidate`][].
        """
        if kwargs.get("container") is None:
            kwargs["container"] = _container_for_name(filename)
        if not _reads_file(self.rawdata, filename):
            write_file(filename, self.to_bytes(**kwargs))
            return
        self.consolidate(inplace=True, **kwargs)
        write_file(filename, _written(self))
        self.rawdata = _Jpeg2000Proxy(
            _Jpeg2000Source(filename=os.fspath(filename)),
            self.metadata.raw,
            0,
        )
        self.images = None


# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


def _read_pyramid(
    cls: type,
    record: Jpeg2000Raw,
    source: _Jpeg2000Source,
    pixel_size: tx.Any,
    unit: tx.Any,
) -> Jpeg2000MultiScaleImage:
    """Build the pyramid of a file that was read.

    The levels are decoded on first access, unless a pixel size or a unit
    overrides the one of the file, in which case they are built with the
    transformations that it gives.
    """
    raw = _Jpeg2000Proxy(source, record, 0)
    metadata = Jpeg2000Metadata.from_raw(record)
    if pixel_size is None and unit is None:
        return cls(rawdata=raw, metadata=metadata)
    images = [
        Jpeg2000Image(
            rawdata=raw
            if level == 0
            else _Jpeg2000Proxy(source, record, level),
            metadata=metadata,
            transformations=[
                level_transformation(record, level, pixel_size, unit)
            ],
        )
        for level in range(record.n_levels)
    ]
    return cls(rawdata=raw, metadata=metadata, images=images)


def _record_levels(
    raw: tx.Any, metadata: tx.Optional[Jpeg2000Metadata]
) -> tx.List[SingleScaleImage]:
    """Decode the levels of a pyramid from its lazy array and metadata.

    A pyramid without a lazy array of a file has no level.
    """
    stored = raw.root if isinstance(raw, ArrayProxy) else raw
    if not isinstance(stored, _Jpeg2000Proxy):
        return []
    # The full-resolution level reads the lazy array of the pyramid itself,
    # so that consolidating an untouched pyramid through that level shares
    # `rawdata`.
    return [
        Jpeg2000Image(
            rawdata=raw
            if level == 0
            else _Jpeg2000Proxy(stored.source, stored.record, level),
            metadata=metadata,
        )
        for level in range(stored.record.n_levels)
    ]


# ----------------------------------------------------------------------
#   WRITING
# ----------------------------------------------------------------------


def _base_level(
    pyramid: Jpeg2000MultiScaleImage, images: tx.Sequence[SingleScaleImage]
) -> Jpeg2000Image:
    """Return the full-resolution level of a pyramid, as it is written.

    The metadata of the pyramid is the metadata that is written, as for
    the other pyramid formats. A level of another class is copied with
    the metadata of the pyramid, whose record is then the base of the
    new file. A JPEG 2000 level that holds other metadata is given the
    metadata of the pyramid in the same way, and it keeps its pixels and
    the transformation that it had. The preferred placement of the
    pyramid is composed with the transformation of the level.

    Only the first level is written. The coarser levels are taken to be
    downsampled versions of it, which the codec derives again, so
    whatever was assigned to them is ignored, and their shapes are not
    checked.

    Raises
    ------
    WriterError
        If the pyramid has no level.
    """
    if not images:
        raise WriterError(
            "This multiscale image has no levels, so there is nothing to "
            "write."
        )
    base = images[0]
    if not isinstance(base, Jpeg2000Image):
        base = Jpeg2000Image.from_instance(base, metadata=pyramid.metadata)
    elif base.metadata is not pyramid.metadata:
        # The transformation is read from the record of the level before
        # that record is replaced by the record of the pyramid.
        held = _preferred(base)
        base = replace(
            base,
            metadata=pyramid.metadata,
            transformations=None if held is None else [held],
        )
    first = _preferred(base)
    if held_transformations(pyramid):
        xform = pyramid.transformation
        if first is not None:
            xform = xform @ first
        base = replace(base, transformations=[xform])
    return base


def _preferred(level: Jpeg2000Image) -> tx.Optional[Transformation]:
    """Return the preferred transformation of a level, if it has one.

    The transformations are read without being stored, so that a write
    leaves the level as it was.
    """
    transformations = level._transformations
    if transformations is None:
        transformations = _record_transformations(level.metadata, level.level)
    return transformations[-1] if transformations else None


def _file_levels(level: Jpeg2000Image) -> tx.Optional[int]:
    """Return the number of levels of the file that a level reads.

    A level that reads no file gives `None`.
    """
    raw = level.rawdata
    stored = raw.root if isinstance(raw, ArrayProxy) else raw
    if not isinstance(stored, _Jpeg2000Proxy):
        return None
    return stored.record.n_levels - stored.level
