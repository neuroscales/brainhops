"""The images and pyramids that every OpenSlide vendor shares.

The vendor classes of `_formats` derive from the two bases of this module,
[`OpenSlideImage`][] and [`OpenSlideMultiScaleImage`][], and add only their
extensions, their hints and their sniffing, which names their vendor.
"""

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import Inherit, NoRepr, replace

# internals
from brainhops._core import path
from brainhops._core._internal_api import (
    ArrayProtocol,
    ArrayProxy,
    preserve_position,
    smartproperty,
)
from brainhops.datamodel.images import MultiScaleImage, SingleScaleImage
from brainhops.datamodel.metadata import ConversionReport
from brainhops.datamodel.metadata._internal_api import apply_loss_policy
from brainhops.datamodel.transformations import Transformation
from brainhops.io.base.parsers import (
    BinaryFileReader,
    Confidence,
    ParserContentError,
    ParserExistsError,
    SnifferContentError,
    SnifferExistsError,
    WriterError,
)
from brainhops.io.common._internal_api import (
    held_images,
    held_transformations,
    same_transformations,
)
from brainhops.io.images.base import ImageFormat
from brainhops.io.metadata._internal_api import (
    metadata_to_write,
    requested_fields,
)

# this format
from ._data import Slide, SlideArray, detect, local_path, read_associated
from ._geometry import level_geometry, record_transformations
from ._metadata import OpenSlideMetadata
from ._raw import read_record
from ._views import image_to_disk, image_to_model

# ----------------------------------------------------------------------
#   READING ADAPTERS
# ----------------------------------------------------------------------


class OpenSlideReader(BinaryFileReader):
    """Reader of every image read with OpenSlide.

    This reader holds no state. It gives the hint `"openslide"`, which selects
    every vendor, while `"openslide.<vendor>"` selects one vendor (see
    [OpenSlide](https://openslide.org/)). It also turns a path, a stream or
    bytes into the slide that OpenSlide opens, from which the `from_source`
    method of each class reads its object. OpenSlide opens slides by the
    name of a local file only, so a remote file, a stream without a local
    file and bytes are first copied to a temporary file, which is deleted
    with the last object that reads it. A stream is sniffed through the
    name of its local file, with the `sniff_filename` method of each class.
    """

    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("openslide",)

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that an open file is a slide of this class.

        OpenSlide sniffs the file through the name of the local file that it
        was opened from. A stream without such a name scores `NO`, so it is
        read only by hint.

        Parameters
        ----------
        file : IO
            The stream, open in binary mode.
        error : bool or type[Exception], default=False
            Whether to raise an error when the stream is not a slide of this
            class.
        **kwargs : Any
            The options of the reader, such as `level`.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        local = local_path(file)
        if local is None:
            if error:
                if error is True:
                    error = SnifferContentError
                raise error("OpenSlide only sniffs local files.")
            return Confidence.NO
        return cls.sniff_filename(local, error=error, **kwargs)

    @classmethod
    def sniff_bytes(
        cls,
        content: path.BinaryContentLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return `NO`, because content in memory is read only by hint.

        Parameters
        ----------
        content : bytes
            The content of a file.
        error : bool or type[Exception], default=False
            Whether to raise an error, which is always raised when set.
        **kwargs : Any
            The options of the reader, which are ignored.

        Returns
        -------
        float
            `NO`.
        """
        if error:
            if error is True:
                error = SnifferContentError
            raise error("OpenSlide only sniffs local files.")
        return Confidence.NO

    @classmethod
    def from_filename(cls, filename: path.FilenameLike, **kwargs) -> tx.Self:
        """Read a slide from a file.

        A local file is opened in place, and a remote file is first copied
        to a temporary file.

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
        local = local_path(filename)
        if local is not None:
            return cls.from_source(Slide(local), **kwargs)
        if isinstance(filename, str):
            filename = path.Path(filename)
        if not path.exists(filename):
            raise ParserExistsError(f"No such file: {filename}")
        with filename.open("rb") as f:
            content = f.read()
        slide = Slide.from_content(content, path.base_name(filename))
        return cls.from_source(slide, **kwargs)

    @classmethod
    def from_fileobj(cls, file: tx.IO, **kwargs) -> tx.Self:
        """Read a slide from an open file.

        A local file is opened by name, and any other stream is copied to a
        temporary file. The position of the stream is restored.

        Parameters
        ----------
        file : IO
            The stream, open in binary mode.
        **kwargs : Any
            The options of `from_source`.
        """
        local = local_path(file)
        if local is not None:
            return cls.from_source(Slide(local), **kwargs)
        with preserve_position(file):
            content = file.read()
        slide = Slide.from_content(content, path.base_name(file))
        return cls.from_source(slide, **kwargs)

    @classmethod
    def from_bytes(cls, content: path.BinaryContentLike, **kwargs) -> tx.Self:
        """Read a slide from bytes, through a temporary copy.

        Parameters
        ----------
        content : bytes
            The content of the file.
        **kwargs : Any
            The options of `from_source`.
        """
        return cls.from_source(Slide.from_content(bytes(content)), **kwargs)


# ----------------------------------------------------------------------
#   SINGLE-SCALE IMAGE
# ----------------------------------------------------------------------


class OpenSlideImage(OpenSlideReader, ImageFormat, SingleScaleImage):
    """A single level of a whole-slide image, read with OpenSlide.

    This class is the base of the single-scale image of every vendor, and
    it reads one level of a slide of any vendor. The image holds the record
    of the whole slide as
    [`OpenSlideMetadata`][brainhops.io.images.openslide.OpenSlideMetadata],
    whose record is an
    [`OpenSlideRaw`][brainhops.io.images.openslide.OpenSlideRaw], the index
    `level` of its level in that record, and the pixels of the level as a
    lazy array in `rawdata`. What OpenSlide reports about the slide,
    such as its vendor, its properties, the downsampling factors of its
    levels, its background colour, its bounds and the names of its
    associated images, is read from the record, as
    `image.metadata.raw.properties`.

    The data is an F-ordered `(x, y, c)` RGB `uint8` array that reads only
    the region it is indexed with, or a dask array of whole tiles. The
    transformation scales pixels to micrometres when the slide records
    `openslide.mpp-x` and `openslide.mpp-y`, and is otherwise the identity,
    with no unit. It is decoded from the record on first access, unless the
    reader options `pixel_size` or `unit` override it, in which case it is
    assigned.

    OpenSlide cannot write slides, so the class has no writer: an image is
    saved to another format, such as TIFF. Assigned data is pending until
    [`consolidate`][] moves it into `rawdata`, and the transformations that the
    image holds, read or assigned, are kept, since no record can be
    encoded.
    """

    rawdata: NoRepr[tx.Optional[ArrayProtocol]] = None
    """The pixels of the level, or `None` without data.

    After a read, `rawdata` is a lazy array of the level, which reads
    the region that it is indexed with. Assigning `data` leaves it as it
    is, and [`consolidate`][] stores the assigned array here.
    """

    _data: tx.Optional[ArrayProtocol] = Inherit("_data", default=None)
    """The data given to the image, or `None` to decode it from `rawdata`.

    The image takes over the `data` field of [`SingleScaleImage`][] to
    store it under this private name. An array stored here is pending until
    [`consolidate`][] moves it into `rawdata`.
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

    _metadata: NoRepr[tx.Optional[OpenSlideMetadata]] = Inherit(
        "metadata", default=None
    )

    metadata = smartproperty("metadata", invalidates="transformations")
    """The metadata of the slide, which holds its record, or `None`.

    An image built from data has no metadata. Assigning other metadata
    drops the transformations decoded from the previous record, so that
    they are decoded from the new one, and keeps the transformations that
    were assigned.
    """

    level: int = 0
    """The index of the level in the record, where 0 is the finest level.

    It is a reader option, and a level of a pyramid holds the record of the
    whole slide, in which it finds its level.
    """

    @smartproperty(cache=True, invalidated_by="rawdata")
    def data(self) -> tx.Optional[ArrayProtocol]:
        """The pixels of the level, assigned or decoded from `rawdata`.

        An array assigned to the image is returned as it is, and stays
        pending until [`consolidate`][] moves it into `rawdata`.
        Otherwise, the data are the lazy array of the file, which reads
        its values when a computation needs them and refuses an edit in
        place. With the reader option `lazy`, the data are instead a
        Dask array of whole tiles. To change the pixels, assign `data`,
        and assigning `None` gives the lazy array of the file again.
        """
        if self.rawdata is None:
            return None
        return image_to_model(self.rawdata)

    @smartproperty(cache=True)
    def transformations(self) -> tx.List[Transformation]:
        """Pixel-to-physical transformations, decoded from the record.

        The list is decoded on first access and cached, apart from an
        assigned list, so a change made to it in place is kept. Assigning
        other metadata drops the decoded list, and keeps an assigned one.
        An image without metadata has no transformations. Assigning `None`
        decodes the record again.
        """
        return record_transformations(self.metadata, self.level)

    @property
    def shape(self) -> tx.Tuple[int, ...]:
        """The shape of the data, read without reading the pixels.

        The shape is that of the assigned data, or else of `rawdata`,
        whose lazy array knows its shape from the record. An image
        without data raises as [`SingleScaleImage`][] does.
        """
        if self._data is not None:
            return tuple(int(d) for d in self._data.shape)
        if self.rawdata is None:
            return self.data.shape
        return tuple(int(d) for d in self.rawdata.shape)

    # --- reading ------------------------------------------------------

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        level: tx.Optional[int] = None,
        **kwargs,
    ) -> float:
        """Return the confidence that a local file is a level of a slide.

        The base class reads the slides of every vendor. A slide scores
        `CERTAIN` when a level is requested or when it has a single level,
        and 0.8 times that otherwise, so that the multiscale class reads a
        pyramid.

        Parameters
        ----------
        filename : str or PathLike
            The path of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when OpenSlide does not read the
            file.
        level : int, optional
            The level that the reader is asked for.
        **kwargs : Any
            The other options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return sniff_level(filename, None, Confidence.CERTAIN, error, level)

    @classmethod
    def from_source(
        cls,
        slide: Slide,
        level: tx.Optional[int] = None,
        pixel_size: tx.Any = None,
        unit: tx.Any = None,
        lazy: tx.Optional[bool] = None,
        **kwargs,
    ) -> tx.Self:
        """Read one level of a slide.

        The class checks with its `sniff_filename` method that OpenSlide
        attributes the slide to its vendor. The record of the slide is read
        at once, and the pixels are read when the data is indexed.

        Parameters
        ----------
        slide : Slide
            The slide.
        level : int, optional
            The level, 0 (the full resolution) by default. A negative level
            counts from the end.
        pixel_size : float or Sequence[float] or Mapping[str, float], optional
            The full-resolution pixel size, which overrides the one of the
            slide.
        unit : str or Unit, optional
            The unit of `pixel_size`, or the unit to which micrometres are
            converted.
        lazy : bool, optional
            Whether the data is a dask array of whole tiles. By default, it
            is one only when dask is the array backend.
        **kwargs : Any
            Refused, since OpenSlide has no other option.

        Returns
        -------
        OpenSlideImage
            The image, whose transformations are assigned when `pixel_size`
            or `unit` is given.

        Raises
        ------
        ParserContentError
            If OpenSlide cannot read the file, or the slide is of another
            vendor.
        IndexError
            If the slide has no such level.
        TypeError
            If an unknown option is given.
        """
        _no_options(cls, kwargs)
        cls.sniff_filename(slide.filename, error=ParserContentError)
        record = read_record(slide.handle)
        count = len(record.levels)
        index = 0 if level is None else int(level)
        if not -count <= index < count:
            raise IndexError(
                f"This slide has {count} level(s), so it has no level {index}."
            )
        index %= count
        image = cls(
            rawdata=SlideArray(slide, record, index, lazy=lazy),
            metadata=OpenSlideMetadata.from_raw(record),
            level=index,
        )
        if pixel_size is not None or unit is not None:
            # The options describe the geometry, so they are assigned.
            image.transformations = [
                level_geometry(record, index, pixel_size, unit)
            ]
        return image

    # --- consolidating ------------------------------------------------

    def _consolidate(
        self,
        *,
        inplace: bool = False,
        **fields: tx.Any,
    ) -> tx.Self:
        """Move the pending data into `rawdata`.

        OpenSlide cannot write slides, so no record can be encoded, and
        consolidating an image is limited to its data: an assigned array
        becomes `rawdata`, and is no longer pending. The metadata and the level
        are kept, and so are the transformations that the image holds, read
        or assigned. An image without pending data is left as it is: the
        copy shares `rawdata` and the metadata.

        The metadata is consolidated first (step 2 of the precedence on
        write), with the fields given by keyword. The record of a slide
        holds no field of the vocabulary yet, so a field that is set is
        reported as lost, once, under the error policy of the call.

        Parameters
        ----------
        inplace : bool, default=False
            Whether to change this image rather than a copy.
        **fields : Any
            Fields of the metadata to override.

        Returns
        -------
        OpenSlideImage
            The consolidated copy, or this image with `inplace=True`.

        Raises
        ------
        TypeError
            If a keyword names no field of the metadata.
        MetadataLossError
            If metadata is lost under the `"raise"` policy.
        """
        report = ConversionReport(
            source="OpenSlideImage", target="OpenSlideImage"
        )
        requested = requested_fields(self.metadata, fields)
        metadata = metadata_to_write(
            self.metadata, OpenSlideMetadata, fields, report
        )
        apply_loss_policy(report, stacklevel=2)
        if not requested:
            metadata = self.metadata
        # The record cannot be encoded, so a copy is given the list that
        # the image holds when it is not the list that the record decodes
        # to, as a list decoded and then changed in place would otherwise
        # be decoded again and lost.
        held = held_transformations(self)
        if held is None or same_transformations(
            held, record_transformations(self.metadata, self.level)
        ):
            held = self._transformations
        if self._data is None:
            if not inplace:
                return replace(self, transformations=held, metadata=metadata)
            if requested:
                self.transformations = held
                self.metadata = metadata
            return self
        raw = image_to_disk(self._data)
        if not inplace:
            return replace(
                self,
                data=None,
                rawdata=raw,
                transformations=held,
                metadata=metadata,
            )
        self.rawdata = raw
        self.data = None
        if requested:
            self.transformations = held
            self.metadata = metadata
        return self

    def associated_image(self, name: str) -> np.ndarray:
        """Read an associated image of the slide.

        Associated images are the label, the macro, the thumbnail and the
        other images that `metadata.raw.associated_images` lists. They are
        read from the slide that `rawdata` reads, and returned as
        F-ordered `(x, y, c)` RGB `uint8` arrays, composited onto white.

        Parameters
        ----------
        name : str
            The name of the associated image.

        Returns
        -------
        np.ndarray
            The associated image.

        Raises
        ------
        KeyError
            If the slide has no associated image of that name, or if the
            image reads no slide.
        """
        return _associated(self.metadata, (self.rawdata,), name)


# ----------------------------------------------------------------------
#   MULTISCALE IMAGE
# ----------------------------------------------------------------------


class OpenSlideMultiScaleImage(OpenSlideReader, ImageFormat, MultiScaleImage):
    """A whole-slide image read with OpenSlide, as a pyramid.

    This class is the base of the multiscale image of every vendor, and it
    reads a slide of any vendor. The pyramid holds the record of the slide
    as
    [`OpenSlideMetadata`][brainhops.io.images.openslide.OpenSlideMetadata],
    and the lazy array of each level, finest first, as a tuple in
    `rawdata`. Its levels are single-scale images decoded from `rawdata`
    and the record on first access, which hold the same metadata, by
    identity, and the index `level` of their level in the record. The
    levels of the base class are [`OpenSlideImage`][] objects, and those
    of a vendor class are objects of the single-scale class of the
    vendor.

    Every level covers the whole slide and maps to the same `"physical"`
    system, as in TIFF and OME-Zarr pyramids, so the pyramid has no
    transformation of its own, and OpenSlide stores none.

    OpenSlide cannot write slides, so the class has no writer, and
    [`consolidate`][] moves only the pending data of the levels into `rawdata`.
    """

    rawdata: NoRepr[tx.Optional[tx.Tuple[ArrayProtocol, ...]]] = None
    """The array of each level, finest first.

    After a read, each array is a lazy array of its level. `rawdata` is
    a tuple, so that it cannot be changed in place, and it describes the
    slide as it was read, or as the pyramid was last consolidated.
    """

    _images: tx.Optional[tx.List[SingleScaleImage]] = Inherit(
        "images", default=None
    )
    """The levels assigned to the pyramid, or `None`.

    The pyramid takes over the `images` field of [`MultiScaleImage`][] to
    store an assigned list under this private name. `None` means that no
    list was assigned, so the property decodes the levels from `rawdata`
    and the record and caches them apart.
    """

    _transformations: tx.Optional[tx.List[Transformation]] = Inherit(
        "transformations", default=None
    )
    """The placements assigned to the pyramid, or `None`.

    `None` means that no list was assigned, so the property decodes the
    placements of the record, which a slide never stores, and caches the
    empty list apart.
    """

    _metadata: NoRepr[tx.Optional[OpenSlideMetadata]] = Inherit(
        "metadata", default=None
    )

    metadata = smartproperty(
        "metadata", invalidates=("images", "transformations")
    )
    """The metadata of the slide, which holds its record, or `None`.

    Every level of the pyramid holds the same metadata. Assigning other
    metadata drops the levels and the placements decoded from the previous
    record, and keeps those that were assigned. The levels that are
    dropped keep the previous metadata.
    """

    PRIORITY: tx.ClassVar[int] = ImageFormat.PRIORITY + 1
    """Content in memory is not sniffed, so when it is read by hint, the
    multiscale and single-scale classes of a vendor would tie. The higher
    priority makes the multiscale class be tried first. That class declines a
    `level` option, which leaves such a request to the single-scale class.
    """

    @smartproperty(cache=True)
    def images(self) -> tx.List[SingleScaleImage]:
        """The levels of the slide, finest first.

        The levels are decoded from `rawdata` and the record on first
        access and cached, apart from an assigned list, so
        `images[0] is images[0]` and a level changed in place, such as
        one whose data is assigned, is seen by [`consolidate`][]. Each
        level is an [`OpenSlideImage`][] that holds its array of
        `rawdata`, the metadata of the pyramid and its index in `level`.
        Assigning a list replaces the stored one, and assigning `None`
        decodes the levels again.
        """
        return decode_levels(OpenSlideImage, self.rawdata, self.metadata)

    @smartproperty(cache=True)
    def transformations(self) -> tx.List[Transformation]:
        """The placements of the pyramid, which a slide never stores.

        The decoded list is empty. An assigned list is kept, since no record
        can be encoded.
        """
        return []

    @property
    def shape(self) -> tx.Tuple[int, ...]:
        """The shape of the finest level, read without reading the pixels.

        The shape is read from the levels that the pyramid holds, or else
        from the first array of `rawdata`.
        """
        images = held_images(self)
        if images is not None:
            return tuple(images[0].shape) if images else ()
        if self.rawdata:
            return tuple(int(n) for n in self.rawdata[0].shape)
        return ()

    # --- reading ------------------------------------------------------

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        level: tx.Optional[int] = None,
        **kwargs,
    ) -> float:
        """Return the confidence that a local file is a pyramidal slide.

        The base class reads the slides of every vendor. A slide of several
        levels scores `CERTAIN` when no level is requested, and any other
        file scores `NO`.

        Parameters
        ----------
        filename : str or PathLike
            The path of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when OpenSlide does not read the
            file.
        level : int, optional
            The level that the reader is asked for, which a pyramid
            declines.
        **kwargs : Any
            The other options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return sniff_pyramid(filename, None, Confidence.CERTAIN, error, level)

    @classmethod
    def from_source(
        cls,
        slide: Slide,
        pixel_size: tx.Any = None,
        unit: tx.Any = None,
        lazy: tx.Optional[bool] = None,
        **kwargs,
    ) -> tx.Self:
        """Read every level of a slide.

        The class checks with its `sniff_filename` method that OpenSlide
        attributes the slide to its vendor. The record of the slide is read
        at once, and each level is a lazy array that reads its pixels when
        its data is indexed.

        Parameters
        ----------
        slide : Slide
            The slide.
        pixel_size : float or Sequence[float] or Mapping[str, float], optional
            The full-resolution pixel size, which overrides the one of the
            slide.
        unit : str or Unit, optional
            The unit of `pixel_size`, or the unit to which micrometres are
            converted.
        lazy : bool, optional
            Whether the data of each level is a dask array of whole tiles.
        **kwargs : Any
            Refused, since OpenSlide has no other option. A pyramid refuses
            `level`, which the single-scale class takes.

        Returns
        -------
        OpenSlideMultiScaleImage
            The pyramid. When `pixel_size` or `unit` is given, its levels
            are assigned, with the transformations that the options give
            them.

        Raises
        ------
        ParserContentError
            If OpenSlide cannot read the file, or the slide is of another
            vendor.
        TypeError
            If an unknown option is given.
        """
        _no_options(cls, kwargs)
        cls.sniff_filename(slide.filename, error=ParserContentError)
        record = read_record(slide.handle)
        raw = tuple(
            SlideArray(slide, record, index, lazy=lazy)
            for index in range(len(record.levels))
        )
        pyramid = cls(rawdata=raw, metadata=OpenSlideMetadata.from_raw(record))
        if pixel_size is not None or unit is not None:
            # The options describe the geometry, so the levels are assigned
            # with the transformations that they give.
            levels = list(pyramid.images)
            for index, image in enumerate(levels):
                image.transformations = [
                    level_geometry(record, index, pixel_size, unit)
                ]
            pyramid.images = levels
        return pyramid

    # --- consolidating ------------------------------------------------

    def _consolidate(
        self,
        *,
        inplace: bool = False,
        **fields: tx.Any,
    ) -> tx.Self:
        """Move the pending data of the levels into `rawdata`.

        OpenSlide cannot write slides, so no record can be encoded, and
        consolidating a pyramid is limited to the data of its levels. A
        pyramid whose levels were neither read nor assigned is left as it
        is: the copy shares `rawdata` and the metadata. Otherwise each
        level that the pyramid holds must be a level of its slide: an
        [`OpenSlideImage`][] that holds the metadata of the pyramid, by
        identity, at its own index. Its pending data becomes its array in
        `rawdata`, and the array of a level without pending data is kept, so a
        pyramid whose levels were only read keeps `rawdata`. The
        metadata is kept, and so are the placements that the pyramid
        holds. The levels of the copy are decoded from `rawdata` and the
        record, unless a level holds transformations that differ from
        those of the record: the copy then holds a list of its levels,
        each with its array of `rawdata` and its transformations, since
        no record can be encoded.

        The metadata is consolidated first (step 2 of the precedence on
        write), with the fields given by keyword. The record of a slide
        holds no field of the vocabulary yet, so a field that is set is
        reported as lost, once, under the error policy of the call.

        Parameters
        ----------
        inplace : bool, default=False
            Whether to change this pyramid rather than a copy.
        **fields : Any
            Fields of the metadata to override.

        Returns
        -------
        OpenSlideMultiScaleImage
            The consolidated copy, or this pyramid with `inplace=True`.

        Raises
        ------
        TypeError
            If a keyword names no field of the metadata.
        MetadataLossError
            If metadata is lost under the `"raise"` policy.
        WriterError
            If the pyramid holds another number of levels than `rawdata`, or a
            level that is not a level of its slide, such as an image of
            another class, since no record can describe it.
        """
        report = ConversionReport(
            source="OpenSlideMultiScaleImage",
            target="OpenSlideMultiScaleImage",
        )
        requested = requested_fields(self.metadata, fields)
        metadata = metadata_to_write(
            self.metadata, OpenSlideMetadata, fields, report
        )
        apply_loss_policy(report, stacklevel=2)
        if not requested:
            metadata = self.metadata
        # The record stores no placement, so a copy is given the list that
        # the pyramid holds when it is not empty, as a list decoded and then
        # changed in place would otherwise be decoded again and lost.
        placements = held_transformations(self)
        if not placements:
            placements = self._transformations
        images = held_images(self)
        if images is None:
            if not inplace:
                return replace(
                    self, transformations=placements, metadata=metadata
                )
            if requested:
                self.transformations = placements
                self.metadata = metadata
            return self
        arrays, levels = _consolidated_levels(self, images, metadata)
        raw = self.rawdata if _same(arrays, self.rawdata) else arrays
        if not inplace:
            return replace(
                self,
                rawdata=raw,
                images=levels,
                transformations=placements,
                metadata=metadata,
            )
        self.rawdata = raw
        self.images = levels
        if requested:
            self.transformations = placements
            self.metadata = metadata
        return self

    def associated_image(self, name: str) -> np.ndarray:
        """Read an associated image of the slide.

        Associated images are the label, the macro, the thumbnail and the
        other images that `metadata.raw.associated_images` lists. They are
        read from the slide that the arrays of `rawdata` read, and returned as
        F-ordered `(x, y, c)` RGB `uint8` arrays, composited onto white.

        Parameters
        ----------
        name : str
            The name of the associated image.

        Returns
        -------
        np.ndarray
            The associated image.

        Raises
        ------
        KeyError
            If the slide has no associated image of that name, or if no
            array of the pyramid reads a slide.
        """
        return _associated(self.metadata, self.rawdata or (), name)


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def sniff_level(
    filename: path.FilenameLike,
    vendor: tx.Optional[str],
    score: float,
    error: tx.Union[bool, tx.Type[Exception]],
    level: tx.Optional[int],
) -> float:
    """Score a local file for the single-scale class of a vendor.

    OpenSlide detects the vendor of the file. A slide of the vendor scores
    `score` when a level is requested or when the slide has a single
    level, and 0.8 times `score` otherwise, so that the multiscale class of
    the vendor reads a pyramid.

    Parameters
    ----------
    filename : str or PathLike
        The path of the file.
    vendor : str or None
        The vendor, as `OpenSlide.detect_format` names it, or `None` for a
        slide of any vendor.
    score : float
        The confidence for a slide of the vendor.
    error : bool or type[Exception]
        Whether to raise an error when the file is not a slide of the
        vendor, and which.
    level : int or None
        The level that the reader is asked for.

    Returns
    -------
    float
        The confidence, between 0 and 1.
    """
    levels = _vendor_levels(filename, vendor, error)
    if levels is None:
        return Confidence.NO
    if level is not None or levels < 2:
        return score
    return 0.8 * score


def sniff_pyramid(
    filename: path.FilenameLike,
    vendor: tx.Optional[str],
    score: float,
    error: tx.Union[bool, tx.Type[Exception]],
    level: tx.Optional[int],
) -> float:
    """Score a local file for the multiscale class of a vendor.

    OpenSlide detects the vendor of the file. A slide of the vendor that
    has several levels scores `score` when no level is requested, and any
    other file scores `NO`.

    Parameters
    ----------
    filename : str or PathLike
        The path of the file.
    vendor : str or None
        The vendor, as `OpenSlide.detect_format` names it, or `None` for a
        slide of any vendor.
    score : float
        The confidence for a pyramidal slide of the vendor.
    error : bool or type[Exception]
        Whether to raise an error when the file is not a slide of the
        vendor, and which. A slide of the vendor with a single level, or a
        request for a level, scores `NO` without an error.
    level : int or None
        The level that the reader is asked for.

    Returns
    -------
    float
        The confidence, between 0 and 1.
    """
    levels = _vendor_levels(filename, vendor, error)
    if levels is None or level is not None or levels < 2:
        return Confidence.NO
    return score


def decode_levels(
    level_type: type,
    raw: tx.Optional[tx.Tuple[ArrayProtocol, ...]],
    metadata: tx.Optional[OpenSlideMetadata],
) -> tx.List[SingleScaleImage]:
    """Return a level object for each array of a pyramid, finest first.

    Parameters
    ----------
    level_type : type
        The single-scale class of the levels.
    raw : tuple of ArrayProtocol or None
        The arrays of the levels.
    metadata : OpenSlideMetadata or None
        The metadata of the pyramid, which every level holds.

    Returns
    -------
    list of SingleScaleImage
        The levels, each with its index in `level`.
    """
    if raw is None:
        return []
    return [
        level_type(rawdata=array, metadata=metadata, level=index)
        for index, array in enumerate(raw)
    ]


def _vendor_levels(
    filename: path.FilenameLike,
    vendor: tx.Optional[str],
    error: tx.Union[bool, tx.Type[Exception]],
) -> tx.Optional[int]:
    """Return the number of levels of a slide of a vendor, or `None`.

    Raises
    ------
    Exception
        The class `error`, when it is set and the file is not a local
        slide of the vendor.
    """
    local = local_path(filename)
    if local is None:
        if error:
            if error is True:
                error = SnifferExistsError
            raise error(f"No such local file: {filename}")
        return None
    found, levels = detect(local)
    if found and (vendor is None or found == vendor):
        return levels
    if error:
        if error is True:
            error = SnifferContentError
        if not found:
            raise error("OpenSlide does not read this file.")
        raise error(
            f"OpenSlide reads this file as a slide of the vendor {found!r}, "
            f"not {vendor!r}."
        )
    return None


def _no_options(cls: type, kwargs: tx.Mapping[str, tx.Any]) -> None:
    """Refuse the options that a reader does not take.

    Raises
    ------
    TypeError
        If any option is given.
    """
    if kwargs:
        raise TypeError(
            f"{cls.__name__} does not take the option(s) "
            f"{', '.join(sorted(kwargs))}."
        )


def _associated(
    metadata: tx.Optional[OpenSlideMetadata],
    arrays: tx.Sequence[tx.Any],
    name: str,
) -> np.ndarray:
    """Read an associated image through the first array that reads a slide.

    Raises
    ------
    KeyError
        If the record lists no associated image of that name, or if no
        array reads a slide.
    """
    record = None if metadata is None else metadata.raw
    if record is None or name not in record.associated_images:
        raise KeyError(name)
    for array in arrays:
        stored = array.root if isinstance(array, ArrayProxy) else array
        if isinstance(stored, SlideArray):
            return read_associated(stored.slide, name)
    raise KeyError(name)


def _consolidated_levels(
    pyramid: OpenSlideMultiScaleImage,
    images: tx.Sequence[tx.Any],
    consolidated: tx.Optional[OpenSlideMetadata],
) -> tx.Tuple[
    tx.Tuple[ArrayProtocol, ...], tx.Optional[tx.List[SingleScaleImage]]
]:
    """Return the arrays of the levels of a pyramid, and the levels to keep.

    Each level must be a level of the slide of the pyramid. Its array is
    its pending data, or else its own `rawdata`. The levels are kept, each with
    its array and the transformations that it holds, only when a level
    holds transformations that differ from those of the record. The
    levels that are kept hold `consolidated`, the metadata of the
    consolidated pyramid, so that each level stays a level of the slide
    that the consolidated pyramid describes.

    Raises
    ------
    WriterError
        If a level is not a level of the slide of the pyramid, or has no
        data.
    """
    raw = pyramid.rawdata or ()
    metadata = pyramid.metadata
    if len(images) != len(raw):
        raise WriterError(
            f"This pyramid holds {len(images)} level(s) where its slide has "
            f"{len(raw)}, and OpenSlide cannot write a slide that would "
            f"describe them."
        )
    arrays = []
    held = []
    changed = False
    for index, image in enumerate(images):
        if (
            not isinstance(image, OpenSlideImage)
            or metadata is None
            or image.metadata is not metadata
            or image.level != index
        ):
            raise WriterError(
                f"Level {index} of this pyramid is not level {index} of its "
                f"slide, and OpenSlide cannot write a slide that would "
                f"describe it."
            )
        if image._data is not None:
            array = image_to_disk(image._data)
        else:
            array = image.rawdata
        if array is None:
            raise WriterError(f"Level {index} of this pyramid has no data.")
        arrays.append(array)
        transformations = held_transformations(image)
        held.append(transformations)
        if transformations is not None and not same_transformations(
            transformations, record_transformations(metadata, index)
        ):
            changed = True
    if not changed:
        return tuple(arrays), None
    levels = [
        replace(
            image,
            data=None,
            rawdata=array,
            transformations=None if kept is None else list(kept),
            metadata=consolidated,
        )
        for image, array, kept in zip(images, arrays, held)
    ]
    return tuple(arrays), levels


def _same(
    first: tx.Sequence[tx.Any], second: tx.Optional[tx.Sequence[tx.Any]]
) -> bool:
    """Tell whether two sequences hold the same objects, in the same order."""
    return second is not None and (
        len(first) == len(second)
        and all(a is b for a, b in zip(first, second))
    )
