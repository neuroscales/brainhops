"""The Imaris image format, which is read only, one level at a time."""

# stdlib
import os.path as op
from io import BytesIO

# dependencies
import h5py
import typing_extensions as tx
from bagof.magic import Inherit, NoRepr, replace

# internals
from brainhops._core import path
from brainhops._core.properties import smartproperty
from brainhops._core.streams import preserve_position
from brainhops._core.typing import ArrayProtocol
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.metadata import ConversionReport
from brainhops.datamodel.metadata._report import (
    apply_loss_policy,
    takes_error_policy,
)
from brainhops.datamodel.transformations import Transformation
from brainhops.errors import UNSET, ErrorPolicy
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import Confidence
from brainhops.io.common._geometry import (
    held_transformations,
    same_transformations,
)
from brainhops.io.common.hdf5 import Hdf5Reader
from brainhops.io.images.base import ImageFormat
from brainhops.io.images.base import _utils_raster as raster
from brainhops.io.metadata._base import metadata_to_write, requested_fields

# this format
from ._data import read_imaris_level
from ._metadata import ImarisMetadata
from ._raw import ImarisRaw, no_options
from ._views import _image_to_disk, _image_to_model

_AXES = "XYZTC"
"""The axes of the data of an Imaris image, fastest first."""

_DEFAULT_UNIT = "um"
"""The unit of a file without a `Unit` parameter.

ImarisWriter writes `"um"` when it is not given (`UpdateImageMetadata` in
`writer/bpWriterHDF5.cxx`), and Bio-Formats reads the extent in
micrometers.
"""


@register_format
class ImarisImage(Hdf5Reader, ImageFormat, SingleScaleImage):
    """One resolution level of an Imaris file, read as a single image.

    An Imaris file holds the same image at several resolutions. This class
    reads one of them, the finest by default or the one that the option
    `level` selects, and
    [`ImarisMultiScaleImage`][brainhops.io.images.imaris.ImarisMultiScaleImage]
    reads them all. A file with a single level is read by this class. An
    Imaris image holds the groups and parameters of its file as
    [`ImarisMetadata`][], whose record is an [`ImarisRaw`][], and the
    voxels of its level in `rawdata`, a lazy array that reads them only
    when `data` is first needed.

    The data of an image always have five axes, `(x, y, z, t, c)`: the
    three spatial axes, then the time points and the channels of the
    file, in the order of the other brainhops images. A file stores each
    channel at each time point as a separate dataset, and the lazy array
    presents them as one array, which reads only the datasets and the
    voxels that a selection covers. When the data are a Dask array, its
    blocks hold whole chunks of the file. The voxels beyond the size of
    the image, which pad the datasets to whole chunks, are left out.

    The transformations are decoded from the record on first access,
    unless other transformations are assigned. There is one, an
    [`Affine`][brainhops.datamodel.transformations.Affine] from the voxel
    system to the `"physical"` system, which scales x, y and z by the
    voxel size of the level and moves the first voxel to the center of
    the first voxel of the extent. The extent of an Imaris file lies on
    the outer border of the voxels, so the voxel size is the size of the
    extent divided by the size of the level, and the first voxel is
    centered half a voxel inside the extent. The unit is the `Unit`
    parameter of the file, or micrometers when it has none. The time and
    channel axes are left as indices, and no orientation is attached to
    the axes. The layout is described in
    [`brainhops.io.images.imaris`][brainhops.io.images.imaris].

    Imaris files cannot be written by brainhops, so the class has no
    writer: an image is saved to another format, such as NIfTI or
    OME-Zarr. Assigned data is pending until [`consolidate`][] moves it
    into `rawdata`, and the transformations that the image holds, read
    or assigned, are kept, since no record can be encoded.
    """

    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".ims",)
    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("imaris", "ims")

    rawdata: NoRepr[tx.Optional[ArrayProtocol]] = None
    """The voxels of the level, slowest axis first, or `None` without data.

    After a read, `rawdata` is a lazy array of shape
    `(channels, time points, z, y, x)`, which reads the voxels when it is
    read. Assigning `data` leaves `rawdata` as it is, and
    [`consolidate`][] stores the assigned array here, with its axes
    reversed.
    """

    _data: tx.Optional[ArrayProtocol] = Inherit("_data", default=None)
    """The data given to the image, or `None` to decode it from `rawdata`.

    The image takes over the `data` field of [`SingleScaleImage`][] to
    store it under this private name. The constructor still takes `data`
    in the same place, and the property of the same name reads it. An
    array stored here is pending until [`consolidate`][] moves it into
    `rawdata`.
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

    _metadata: NoRepr[tx.Optional[ImarisMetadata]] = Inherit(
        "metadata", default=None
    )

    level: int = 0
    """The resolution level that the image holds, from 0 for the finest.

    It is the reader option `level`, and it tells which level of the
    record places the image in space.
    """

    metadata = smartproperty("metadata", invalidates="transformations")
    """The metadata of the file, which holds its record, or `None`.

    An image built from data has no metadata. Assigning other metadata
    drops the transformations decoded from the previous record, so that
    they are decoded from the new one, and keeps the transformations that
    were assigned.
    """

    @smartproperty(cache=True, invalidated_by="rawdata")
    def data(self) -> tx.Optional[ArrayProtocol]:
        """The voxels, indexed `(x, y, z, t, c)`, assigned or decoded from
        `rawdata`.

        An array assigned to the image is returned as it is, and stays
        pending until [`consolidate`][] moves it into `rawdata`.
        Otherwise, the data are the lazy array of the file, which reads
        its values when a computation needs them and refuses an edit in
        place. To change the voxels, assign `data`, and assigning `None`
        gives the lazy array of the file again.
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
        transformations. Assigning a list replaces the stored one, and
        assigning `None` decodes the record again.
        """
        return _record_transformations(self.metadata, self.level)

    @property
    def shape(self) -> tx.Tuple[int, ...]:
        """The shape of the data, read without reading the voxels.

        The shape is that of the assigned data, or else the reverse of the
        shape of `rawdata`: the data reverse the axes of `rawdata`, and
        the lazy array knows its shape from the record. An image without
        data raises as [`SingleScaleImage`][] does.
        """
        if self._data is not None:
            return tuple(int(d) for d in self._data.shape)
        if self.rawdata is None:
            return self.data.shape
        return tuple(int(d) for d in reversed(self.rawdata.shape))

    @takes_error_policy
    def consolidate(
        self,
        *,
        inplace: bool = False,
        on_error: ErrorPolicy = UNSET,
        **fields: tx.Any,
    ) -> "ImarisImage":
        """Move the pending data into `rawdata`.

        Imaris files cannot be written by brainhops, so no record can be
        encoded, and consolidating an Imaris image is limited to its data:
        an assigned array becomes `rawdata`, with its axes reversed, and is no
        longer pending. The metadata is kept, and so are the
        transformations that the image holds, read or assigned. An image
        without pending data is left as it is: the copy shares `rawdata`
        and the metadata.

        The metadata is consolidated first (step 2 of the precedence on
        write), with the fields given by keyword. The record of an
        Imaris file holds no field of the vocabulary yet, so a field
        that is set is reported as lost, once, by `on_error`.

        Parameters
        ----------
        inplace : bool, default=False
            Whether to change this image rather than a copy.
        on_error : ErrorPolicy, optional
            What to do with the metadata that the record cannot hold,
            and with any other conversion error
            (see [`ErrorPolicy`][brainhops.errors.ErrorPolicy]). By
            default, an error is raised, and a loss follows the policy
            in effect.
        **fields : Any
            Fields of the metadata to override.

        Returns
        -------
        ImarisImage
            The consolidated copy, or this image with `inplace=True`.

        Raises
        ------
        TypeError
            If a keyword names no field of the metadata.
        MetadataLossError
            If metadata is lost under the `"raise"` policy.
        """
        report = ConversionReport(source="ImarisImage", target="ImarisImage")
        requested = requested_fields(self.metadata, fields)
        metadata = metadata_to_write(
            self.metadata, ImarisMetadata, fields, report
        )
        apply_loss_policy(report, on_error, stacklevel=2)
        if not requested:
            metadata = self.metadata
        # The record cannot be encoded, so a copy is given the list that
        # the image holds when it is not the list that the record decodes
        # to, as a list decoded and then changed in place would otherwise
        # be decoded again and lost.
        held = held_transformations(self)
        if held is None or same_transformations(
            held, _record_transformations(self.metadata, self.level)
        ):
            held = self._transformations
        if self._data is None:
            if not inplace:
                return replace(self, transformations=held, metadata=metadata)
            if requested:
                self.transformations = held
                self.metadata = metadata
            return self
        raw = _image_to_disk(self._data)
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

    # --- reading ------------------------------------------------------

    @classmethod
    def sniff_h5(
        cls,
        h5file: h5py.File,
        error: tx.Union[bool, tx.Type[Exception]] = False,
    ) -> float:
        """Return the confidence that an open HDF5 file is an Imaris file
        that this class reads.

        The score is `CERTAIN` for an Imaris file with one resolution
        level and `LIKELY` for one with several, so that
        [`ImarisMultiScaleImage`][brainhops.io.images.imaris.ImarisMultiScaleImage]
        reads a pyramid unless the option `level` is given.

        Parameters
        ----------
        h5file : h5py.File
            The open file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the file is not an Imaris file.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        score = ImarisRaw.sniff_h5(h5file, error=error)
        if score and _level_count(h5file) > 1:
            return Confidence.LIKELY
        return score

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        level: tx.Optional[int] = None,
        **kwargs,
    ) -> float:
        """Score the Imaris file at a path.

        Parameters
        ----------
        filename : path-like
            The path of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the file is not an Imaris file.
        level : int, optional
            The level that the caller asks for. Any Imaris file is then
            `CERTAIN`.
        **kwargs : Any
            Other options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        score = super().sniff_filename(filename, error=error)
        return Confidence.CERTAIN if score and level is not None else score

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        level: tx.Optional[int] = None,
        **kwargs,
    ) -> float:
        """Score an open, seekable binary stream.

        Parameters
        ----------
        file : file object
            The stream, open in binary mode, whose position is restored.
        error : bool or type[Exception], default=False
            Whether to raise an error when the stream is not an Imaris
            file.
        level : int, optional
            The level that the caller asks for. Any Imaris file is then
            `CERTAIN`.
        **kwargs : Any
            Other options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        score = super().sniff_fileobj(file, error=error)
        return Confidence.CERTAIN if score and level is not None else score

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        level: tx.Optional[int] = None,
        **kwargs,
    ) -> float:
        """Score the bytes of an Imaris file.

        Parameters
        ----------
        content : bytes
            The content of a file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the bytes are not an Imaris
            file.
        level : int, optional
            The level that the caller asks for. Any Imaris file is then
            `CERTAIN`.
        **kwargs : Any
            Other options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return cls.sniff_fileobj(BytesIO(content), error=error, level=level)

    @classmethod
    def from_h5(cls, h5file: h5py.File, level: int = 0, **kwargs) -> tx.Self:
        """Read one level of an open Imaris file, without its voxels.

        The lazy array reads the voxels again from the file by its name,
        when they are needed. A file that h5py opened over a stream has no
        name, so the lazy array reads that open file, which the caller
        must keep open while the image is used.

        Parameters
        ----------
        h5file : h5py.File
            The open file.
        level : int, default=0
            The resolution level to read, from 0 for the finest.
        **kwargs : Any
            Other options of the reader, which are refused.

        Returns
        -------
        ImarisImage
            The image, whose voxels are read lazily.

        Raises
        ------
        TypeError
            If an unknown option is given.
        IndexError
            If the file has no such level.
        ParserContentError
            If the file is not an Imaris file that brainhops reads.
        """
        no_options(kwargs)
        return _read_level(cls, h5file, _source(h5file), level)

    @classmethod
    def from_fileobj(
        cls, file: tx.BinaryIO, level: int = 0, **kwargs
    ) -> tx.Self:
        """Read one level of an Imaris file from an open stream.

        A stream has no name to open again, so its content is read into
        memory, where the lazy array reads the voxels when they are
        needed. The position of the stream is restored.

        Parameters
        ----------
        file : BinaryIO
            The stream, open in binary mode.
        level : int, default=0
            The resolution level to read, from 0 for the finest.
        **kwargs : Any
            Other options of the reader, which are refused.

        Returns
        -------
        ImarisImage
            The image, whose voxels are read lazily.

        Raises
        ------
        TypeError
            If an unknown option is given.
        IndexError
            If the file has no such level.
        ParserContentError
            If the stream is not an Imaris file that brainhops reads.
        """
        no_options(kwargs)
        with preserve_position(file):
            content = bytes(file.read())
        with h5py.File(BytesIO(content), "r") as f:
            return _read_level(cls, f, content, level)


# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


def _source(h5file: h5py.File) -> tx.Union[str, h5py.File]:
    """Return what the lazy array of an open file reads from.

    A file opened by name is read again by its absolute name, and a file
    that h5py opened over a stream is kept open.
    """
    if h5file.driver == "fileobj":
        return h5file
    return op.abspath(h5file.filename)


def _level_count(h5file: h5py.File) -> int:
    """Return the number of resolution levels of an open Imaris file."""
    group = h5file["DataSet"]
    count = 0
    while f"ResolutionLevel {count}" in group:
        count += 1
    return count


def _read_level(
    cls: type, h5file: h5py.File, source: tx.Any, level: int
) -> tx.Any:
    """Read the record of an open file and build the image of one level."""
    record = ImarisRaw.from_h5(h5file)
    return cls(
        rawdata=read_imaris_level(source, record, int(level)),
        metadata=ImarisMetadata.from_raw(record),
        level=int(level),
    )


def _record_transformations(
    metadata: tx.Optional[ImarisMetadata], level: int
) -> tx.List[Transformation]:
    """Decode the transformations of one level from the record of some
    metadata.

    Metadata without a record gives no transformation.
    """
    if metadata is None or metadata.raw is None:
        return []
    return [_level_transformation(metadata.raw, level)]


def _level_transformation(record: ImarisRaw, level: int) -> Transformation:
    """Return the voxel-to-physical transformation of one level.

    Every level spans the extent of the file. The voxel size of level 0
    is the size of the extent divided by the size of that level, and its
    first voxel is centered half a voxel inside the extent. A coarser
    level is placed as a level of every other brainhops pyramid, by the
    ratio of the size of level 0 to its own size, which gives the size of
    the extent divided by the size of the level.

    Parameters
    ----------
    record : ImarisRaw
        The record of the file.
    level : int
        The index of the level, from 0 for the finest.

    Returns
    -------
    Transformation
        The transformation from the voxel system, with axes x, y, z, t and
        c, to the `"physical"` system.
    """
    axes = raster.storage_axes(_AXES)
    unit = _unit(record.unit)
    low, high = record.extent_min, record.extent_max
    base = record.levels[0].size
    sizes = record.levels[level].size
    scales = {}
    origin = {}
    for name, a, b, n in zip("xyz", low, high, base):
        size = (b - a) / n
        scales[name] = (size, unit)
        origin[name] = a + size / 2
    factors = [n / m for n, m in zip(base, sizes)] + [1.0, 1.0]
    return raster.level_transformation(axes, scales, factors, origin)


def _unit(text: tx.Optional[str]) -> tx.Optional[str]:
    """Return the unit of the extent, or `None` when it is not a length.

    A file without a `Unit` parameter, or with an empty one, is in
    micrometers.
    """
    if not text:
        return _DEFAULT_UNIT
    try:
        raster.convert_length(1.0, text, "m")
    except ValueError:
        return None
    return text
