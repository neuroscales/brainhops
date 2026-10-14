"""One scale of a Neuroglancer precomputed volume, read as an image."""

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import Inherit, NoRepr, replace

# internals
from brainhops._core import path
from brainhops._core.dependencies import HAS_TENSORSTORE
from brainhops._core.properties import smartproperty
from brainhops._core.typing import ArrayProtocol
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.transformations import Affine, Transformation
from brainhops.io.base._base import register_format
from brainhops.io.base.parsers import (
    BinaryFileReader,
    Confidence,
    ParserContentError,
    SnifferTypeError,
)
from brainhops.io.common._geometry import (
    held_transformations,
    same_transformations,
)
from brainhops.io.images.base import ImageFormat
from brainhops.io.images.base._utils_raster import (
    default_axes,
    physical_system,
    pixel_system,
)

# this format
from ._data import (
    READABLE_ENCODINGS,
    _PrecomputedProxy,
    read_precomputed,
    stream_location,
)
from ._metadata import PrecomputedMetadata
from ._raw import PrecomputedRaw, _no_options
from ._views import _image_to_disk, _image_to_model

_UNIT = "nm"
"""The unit of the resolution of a scale, and of the physical space."""


class PrecomputedImage(ImageFormat, SingleScaleImage, BinaryFileReader):
    """One scale of a Neuroglancer precomputed volume.

    A precomputed volume is a directory, local or remote, that holds an
    `info` file and the chunks of each scale (see
    [`brainhops.io.images.precomputed`][brainhops.io.images.precomputed]).
    This class reads one scale, the finest by default or the one that
    `level` selects, from the directory or from its `info` file. A
    volume of several scales is read whole by
    [`PrecomputedMultiScaleImage`][brainhops.io.images.precomputed.PrecomputedMultiScaleImage]
    unless `level` is given.

    The image holds the `info` file as [`PrecomputedMetadata`][], whose
    record is a [`PrecomputedRaw`][], and the voxels in `rawdata`, a
    lazy array that TensorStore reads chunk by chunk when the data are
    needed. The data are indexed `[x, y, z]`, with a channel axis `c`
    last when the volume has several channels. The first voxel of the
    data is the first voxel of the scale, whatever its `voxel_offset`.

    The transformation is an [`Affine`][] from the voxels to the
    `"physical"` space of Neuroglancer, in nanometres. It scales each
    spatial axis by the `resolution` of the scale and moves it by
    `(voxel_offset + 0.5) * resolution`, because an integer index of
    brainhops is the centre of a voxel while Neuroglancer puts the
    corner of the voxel `voxel_offset` at `voxel_offset * resolution`.
    The physical space has no anatomical orientation.

    Precomputed volumes are not written, so the class has no writer, and
    [`save`][brainhops.io.save] refuses to write the image as a
    precomputed volume. Assigned data are pending until
    [`consolidate`][] moves them into `rawdata`, and the transformations
    that the image holds, read or assigned, are kept, since no record can
    be encoded.
    """

    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("precomputed", "neuroglancer")

    rawdata: NoRepr[tx.Optional[ArrayProtocol]] = None
    """The voxels as the volume stores them, or `None` without data.

    After a read, `rawdata` is a lazy array of the scale. Assigning
    `data` leaves `rawdata` as it is, and [`consolidate`][] stores the
    assigned array here.
    """

    _data: tx.Optional[ArrayProtocol] = Inherit("data", default=None)
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

    _metadata: NoRepr[tx.Optional[PrecomputedMetadata]] = None

    metadata = smartproperty("metadata", invalidates="transformations")
    """The metadata of the volume, which holds its record, or `None`.

    An image built from data has no metadata. Assigning other metadata
    drops the transformations decoded from the previous record, so that
    they are decoded from the new one, and keeps the transformations that
    were assigned.
    """

    level: tx.Optional[int] = None
    """
    The index of the scale in the `info` file, where 0 is the finest.
    `None`, the default, reads the finest scale.
    """

    @smartproperty(cache=True)
    def data(self) -> tx.Optional[ArrayProtocol]:
        """The voxels, assigned or decoded from `rawdata`.

        An array assigned to the image is returned as it is, and stays
        pending until [`consolidate`][] moves it into `rawdata`. Otherwise the
        data is decoded from `rawdata` on first access and cached. Data
        decoded from a volume is read-only. To change the voxels, assign
        `data`. Assigning `None` decodes `rawdata` again.
        """
        if self.rawdata is None:
            return None
        return _image_to_model(self.rawdata)

    @smartproperty(cache=True)
    def transformations(self) -> tx.List[Transformation]:
        """Voxel-to-physical transformations, decoded from the record.

        The list is decoded on first access and cached, apart from an
        assigned list, so `img.transformation is img.transformation`,
        and a change made to the list in place is kept. Assigning other
        metadata drops the decoded list, so that it is decoded from the
        new record, and keeps an assigned list. A change made in place to
        a decoded list is therefore lost when other metadata is assigned.
        An image without metadata has no transformations. Assigning a list
        replaces the stored one, and assigning `None` decodes the record
        again.
        """
        return _record_transformations(self.metadata, self.level)

    @property
    def shape(self) -> tx.Tuple[int, ...]:
        """The shape of the data, read without reading the voxels.

        The shape is that of the assigned data, or else the shape of
        `rawdata`, which the lazy array knows from the record. An image
        without data raises as [`SingleScaleImage`][] does.
        """
        if self._data is not None:
            return tuple(int(d) for d in self._data.shape)
        if self.rawdata is None:
            return self.data.shape
        return tuple(int(d) for d in self.rawdata.shape)

    def consolidate(self, *, inplace: bool = False) -> "PrecomputedImage":
        """Move the pending data into `rawdata`.

        Precomputed volumes are not written, so no record can be encoded,
        and consolidating an image is limited to its data: an assigned
        array becomes `rawdata` and is no longer pending. The metadata
        is kept, and so are the transformations that the image holds,
        read or assigned. An image without pending data is left as it
        is: the copy shares `rawdata` and the metadata.

        Parameters
        ----------
        inplace : bool, default=False
            Whether to change this image rather than a copy.

        Returns
        -------
        PrecomputedImage
            The consolidated copy, or this image with `inplace=True`.
        """
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
            return self if inplace else replace(self, transformations=held)
        raw = _image_to_disk(self._data)
        if not inplace:
            return replace(self, data=None, rawdata=raw, transformations=held)
        self.rawdata = raw
        self.data = None
        return self

    # --- reading ------------------------------------------------------

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        level: tx.Optional[int] = None,
        **kwargs,
    ) -> float:
        """Return the confidence that a name designates a volume.

        The name is the directory of the volume or its `info` file, local
        or remote. A volume is certainly read by this class when `level`
        is given. Otherwise the score is `Confidence.LIKELY`, below
        [`PrecomputedMultiScaleImage`][brainhops.io.images.precomputed.PrecomputedMultiScaleImage],
        which reads a volume of several scales whole.

        Parameters
        ----------
        filename : path-like
            The directory or the `info` file of a volume.
        error : bool or type[Exception], default=False
            Whether to raise an error when the name designates no volume.
        level : int, optional
            The scale to read.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        score = PrecomputedMetadata.sniff_filename(filename, error=error)
        if not score:
            return score
        return Confidence.CERTAIN if level is not None else Confidence.LIKELY

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        level: tx.Optional[int] = None,
        **kwargs,
    ) -> float:
        """Return the confidence that a stream reads the `info` of a volume.

        Parameters
        ----------
        file : IO
            The stream of the `info` file, open in binary mode.
        error : bool or type[Exception], default=False
            Whether to raise an error when the stream holds no `info`
            file of a volume.
        level : int, optional
            The scale to read.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        score = PrecomputedMetadata.sniff_fileobj(file, error=error)
        if not score:
            return score
        return Confidence.CERTAIN if level is not None else Confidence.LIKELY

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return `Confidence.NO`, since bytes cannot hold a whole volume.

        Parameters
        ----------
        content : bytes
            The bytes, which are not read.
        error : bool or type[Exception], default=False
            Whether to raise an error, which is then always raised.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            `Confidence.NO`.
        """
        if error:
            if error is True:
                error = SnifferTypeError
            raise error(
                "A precomputed volume is a directory of files, which bytes "
                "alone cannot hold."
            )
        return Confidence.NO

    @classmethod
    def from_filename(
        cls,
        filename: path.FilenameLike,
        level: tx.Optional[int] = None,
        **kwargs,
    ) -> "PrecomputedImage":
        """Read one scale of a volume from its directory or `info` file.

        Only the `info` file is read. The lazy array reads the chunks when
        the voxels are needed. A remote volume, such as
        `gs://bucket/volume`, is read with TensorStore.

        Parameters
        ----------
        filename : path-like
            The directory or the `info` file of a volume, local or remote.
        level : int, optional
            The index of the scale, 0 (the finest) by default. A negative
            index counts from the coarsest scale.
        **kwargs : Any
            Refused, since the format has no other option.

        Returns
        -------
        PrecomputedImage
            The image, whose voxels are read lazily.

        Raises
        ------
        TypeError
            If another option is given.
        ParserExistsError
            If there is no volume at that name.
        ParserContentError
            If the `info` file breaks the specification, if there is no
            such scale, or if the encoding of the scale cannot be read.
        """
        _no_options(kwargs)
        location, record = read_precomputed(filename)
        metadata = PrecomputedMetadata.from_raw(record)
        return _read_scale(cls, location, metadata, level)

    @classmethod
    def from_fileobj(
        cls,
        file: tx.BinaryIO,
        level: tx.Optional[int] = None,
        **kwargs,
    ) -> "PrecomputedImage":
        """Read one scale of a volume from an open `info` file.

        The volume is found from the name of the stream, which must have
        one, and the `info` file is read again from there.

        Parameters
        ----------
        file : BinaryIO
            The open `info` file, which must have a name.
        level : int, optional
            The index of the scale, 0 (the finest) by default.
        **kwargs : Any
            Refused, since the format has no other option.

        Returns
        -------
        PrecomputedImage
            The image, whose voxels are read lazily.

        Raises
        ------
        TypeError
            If another option is given.
        ParserTypeError
            If the stream has no file name.
        """
        _no_options(kwargs)
        location, record = read_precomputed(stream_location(file))
        metadata = PrecomputedMetadata.from_raw(record)
        return _read_scale(cls, location, metadata, level)

    @classmethod
    def from_bytes(cls, content: bytes, **kwargs) -> "PrecomputedImage":
        """Refuse to read bytes, which cannot hold a volume of many files.

        Parameters
        ----------
        content : bytes
            The bytes, which are not read.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        PrecomputedImage
            Never, since an error is always raised.

        Raises
        ------
        ParserContentError
            Always.
        """
        raise ParserContentError(
            "A precomputed volume is a directory of files, which bytes "
            "alone cannot hold. Read it from the path of its directory or "
            "of its info file instead."
        )


# Without TensorStore, the voxels cannot be read, so the format is not
# registered, and a request for it by hint reports which package to
# install (see `brainhops.io.images`).
if HAS_TENSORSTORE:
    register_format(PrecomputedImage)


# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


def _read_scale(
    cls: type,
    location: str,
    metadata: PrecomputedMetadata,
    level: tx.Optional[int],
) -> PrecomputedImage:
    """Build the image of one scale of a volume, which holds `metadata`.

    Raises
    ------
    ParserContentError
        If there is no such scale, or its encoding cannot be read.
    """
    record = metadata.raw
    index = _level_index(record, level)
    scale = record.scales[index]
    if scale.encoding not in READABLE_ENCODINGS:
        raise ParserContentError(
            f"The scale {scale.key!r} of the precomputed volume at "
            f"{location} is encoded as {scale.encoding!r}, which brainhops "
            f"cannot decode. The encodings that it reads are "
            f"{', '.join(READABLE_ENCODINGS)}."
        )
    proxy = _PrecomputedProxy(location, record, scale)
    return cls(rawdata=proxy, metadata=metadata, level=index)


def _level_index(record: PrecomputedRaw, level: tx.Optional[int]) -> int:
    """Return the index of a scale, 0 for `None`, counting from 0.

    Raises
    ------
    ParserContentError
        If the volume has no such scale.
    """
    count = len(record.scales)
    index = 0 if level is None else int(level)
    if not -count <= index < count:
        raise ParserContentError(
            f"The precomputed volume has {count} scale(s), so it has no "
            f"scale {index}."
        )
    return index % count


def _record_transformations(
    metadata: tx.Optional[PrecomputedMetadata], level: tx.Optional[int]
) -> tx.List[Transformation]:
    """Decode the transformation of a scale from the record of metadata.

    Metadata without a record gives no transformation.
    """
    if metadata is None or metadata.raw is None:
        return []
    record = metadata.raw
    return [_scale_transformation(record, _level_index(record, level))]


def _scale_transformation(record: PrecomputedRaw, index: int) -> Affine:
    """Return the voxel-to-physical transformation of a scale.

    The spatial axes are scaled by the resolution, in nanometres, and
    moved by `(voxel_offset + 0.5) * resolution`, so that the centre of
    the voxel `i` lies where Neuroglancer puts it. A channel axis is
    neither scaled nor moved.
    """
    scale = record.scales[index]
    channels = record.num_channels != 1
    axes = default_axes(4 if channels else 3, channel=channels)
    voxel = pixel_system(axes)
    physical = physical_system(axes, {name: _UNIT for name in "xyz"})
    n = len(axes)
    matrix = np.zeros((n, n + 1))
    for i in range(n):
        if i < 3:
            matrix[i, i] = scale.resolution[i]
            matrix[i, n] = (scale.voxel_offset[i] + 0.5) * scale.resolution[i]
        else:
            matrix[i, i] = 1.0
    return Affine(matrix=matrix, input=voxel, output=physical)
