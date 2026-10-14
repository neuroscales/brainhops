"""The MINC image formats, which are read only."""

# stdlib
from io import BytesIO

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import Inherit, NoRepr, replace

# internals
from brainhops._core import path
from brainhops._core.dependencies import HAS_H5PY
from brainhops._core.properties import smartproperty
from brainhops._core.typing import ArrayProtocol
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.images import SingleScaleImage
from brainhops.datamodel.metadata import ConversionReport
from brainhops.datamodel.metadata._report import OnLoss, apply_loss_policy
from brainhops.datamodel.orientations import Orientation
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.datamodel.transformations import Affine, Scaling, Transformation
from brainhops.datamodel.units import (
    SpaceUnit,
    TimeUnit,
    Unit,
    is_physicalunit,
)
from brainhops.io.base._base import register_format
from brainhops.io.base._utils_files import is_local
from brainhops.io.base.parsers import BinaryFileReader, ParserContentError
from brainhops.io.common.minc import MincDimension, MincMetadata, MincRaw
from brainhops.io.common.minc._data import read_minc
from brainhops.io.common.minc._raw import _no_options, sniff_minc
from brainhops.io.common.minc._utils import (
    _axis,
    _score,
    _sniff_minc2,
    minc_version,
)
from brainhops.io.images.base import ImageFormat
from brainhops.io.metadata._base import (
    metadata_to_write,
    requested_fields,
)

# this format
from ._views import _image_to_disk, _image_to_model

_PHYSICAL = "physical"
"""Name of the scaled voxel space."""

_WORLD = "world"
"""Name of the MINC world (RAS) space, the preferred world space."""

_RAS_ORIENTATION = {
    "x": "left-to-right",
    "y": "posterior-to-anterior",
    "z": "inferior-to-superior",
}

_UNIT_TYPES = {"space": SpaceUnit, "time": TimeUnit}


class MincImage(ImageFormat, SingleScaleImage, BinaryFileReader):
    """An image stored in a MINC file, of version 1 or 2.

    This class answers to the format hint `"minc"` and reads both
    versions. It hands the file to [`Minc1Image`][] or [`Minc2Image`][],
    which are the classes registered as formats. A MINC image holds the
    header of its file as [`MincMetadata`][], whose record is a
    [`MincRaw`][], and the voxels in `rawdata`, a lazy array that reads
    them with nibabel, scaled to real values, only when `data` is first
    needed. The lazy array presents the voxels in the order of the file,
    and the data reverse it into Fortran order, so a file with
    dimensions `zspace, yspace, xspace` gives data indexed `(x, y, z)`.

    The transformations are decoded from the record on first access,
    unless other transformations are assigned. They are, in order, a
    [`Scaling`][] to `"physical"` by the absolute step of each dimension,
    in its unit (millimeters by default for spatial dimensions), and the
    preferred [`Affine`][] to the RAS space `"world"`, which moves along
    the direction cosines of each spatial dimension by its signed step
    from `start`. The affine is recorded only for volumes with three
    spatial dimensions.

    MINC files cannot be written, so the class has no writer: an image is
    saved to another format, such as NIfTI, and [`save`][brainhops.io.save]
    refuses to write it as MINC. The record is `metadata.raw`, and it
    describes the file as it was read. Assigned data is pending until
    [`consolidate`][] moves it into `rawdata`, and the transformations
    that the image holds, read or assigned, are kept, since no record
    can be encoded.
    """

    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("minc",)
    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".mnc",)

    VERSION: tx.ClassVar[tx.Optional[int]] = None
    """The MINC version this class reads, or `None` for both."""

    VARIANTS: tx.ClassVar[tx.Tuple[type, ...]] = ()
    """The classes to which a versionless class hands files."""

    rawdata: NoRepr[tx.Optional[ArrayProtocol]] = None
    """The voxels in the order of the file, or `None` without data.

    After a read, `rawdata` is a lazy array, which reads the voxels and
    scales them when it is read. Assigning `data` leaves `rawdata` as it
    is, and [`consolidate`][] stores the assigned array here, with its
    axes reversed.
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

    _metadata: NoRepr[tx.Optional[MincMetadata]] = Inherit(
        "metadata", default=None
    )

    metadata = smartproperty("metadata", invalidates="transformations")
    """The metadata of the file, which holds its record, or `None`.

    An image built from data has no metadata. Assigning
    other metadata drops the transformations decoded from the previous
    record, so that they are decoded from the new one, and keeps the
    transformations that were assigned.
    """

    @smartproperty(cache=True)
    def data(self) -> tx.Optional[ArrayProtocol]:
        """The voxels in Fortran order, assigned or decoded from `rawdata`.

        An array assigned to the image is returned as it is, and stays
        pending until [`consolidate`][] moves it into `rawdata`. Otherwise the
        data is decoded from `rawdata` on first access and cached. Data
        decoded from a file is read-only. To change the voxels, assign
        `data`. Assigning `None` decodes `rawdata` again.
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

    @property
    def system(self) -> tx.Optional[CoordinateSystem]:
        """The voxel coordinate system described by the record.

        The axes are named after the dimensions (`xspace` gives `x`,
        `time` gives `t`) and are in Fortran order. An image without
        metadata has no system.
        """
        if self.metadata is None or self.metadata.raw is None:
            return None
        return _minc_system(self.metadata.raw)

    def consolidate(
        self,
        *,
        inplace: bool = False,
        on_loss: tx.Optional[OnLoss] = None,
        **fields: tx.Any,
    ) -> "MincImage":
        """Move the pending data into `rawdata`.

        MINC files cannot be written, so no record can be encoded, and
        consolidating a MINC image is limited to its data: an assigned
        array becomes `rawdata`, with its axes reversed, and is no
        longer pending. The metadata is kept, and so are the
        transformations that the image holds, read or assigned. An image
        without pending data is left as it is: the copy shares `rawdata`
        and the metadata.

        The metadata is consolidated first (step 2 of the precedence on
        write), with the fields given by keyword. The record of a MINC
        file holds no field of the vocabulary yet, so a field that is set
        is reported as lost, once, by `on_loss`, and the copy holds the
        consolidated metadata.

        Parameters
        ----------
        inplace : bool, default=False
            Whether to change this image rather than a copy.
        on_loss : {"ignore", "warn", "raise"} or ConversionReport, optional
            What to do with the metadata that the record cannot hold. By
            default, the policy in effect.
        **fields : Any
            Fields of the metadata to override.

        Returns
        -------
        MincImage
            The consolidated copy, or this image with `inplace=True`.

        Raises
        ------
        TypeError
            If a keyword names no field of the metadata.
        MetadataLossError
            If metadata is lost under the `"raise"` policy.
        """
        report = ConversionReport(source="MincImage", target="MincImage")
        requested = requested_fields(self.metadata, fields)
        metadata = metadata_to_write(
            self.metadata, MincMetadata, fields, report
        )
        apply_loss_policy(report, on_loss, stacklevel=2)
        if not requested:
            metadata = self.metadata
        if self._data is None and not requested:
            return self if inplace else replace(self)
        raw = (
            self.rawdata if self._data is None else _image_to_disk(self._data)
        )
        if not inplace:
            return replace(self, data=None, rawdata=raw, metadata=metadata)
        self.rawdata = raw
        self.data = None
        self.metadata = metadata
        return self

    # --- reading ------------------------------------------------------

    @classmethod
    def _variant(cls, version: tx.Optional[int]) -> type:
        """Return the class that reads a file of a given version.

        Raises
        ------
        ParserContentError
            If neither this class nor its variants read that version.
        """
        if cls.VERSION is not None:
            if cls.VERSION == version:
                return cls
        else:
            for variant in cls.VARIANTS:
                if variant.VERSION == version:
                    return variant
        raise ParserContentError(
            f"{cls.__name__} does not read MINC{version or ''} files."
        )

    @classmethod
    def _versions(cls) -> tx.Tuple[int, ...]:
        """Return the versions that this class or its variants read."""
        if cls.VERSION is not None:
            return (cls.VERSION,)
        return tuple(v.VERSION for v in cls.VARIANTS)

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score a file from its content.

        A local MINC2 file is opened by name, so that only its header is
        read.

        Parameters
        ----------
        filename : path-like
            The path of the file.
        error : bool or type[Exception], default=False
            Whether to raise an error when the file is not a MINC file
            that this class reads.
        **kwargs : Any
            Options of the reader, which are passed on when the
            content is scored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        if isinstance(filename, str):
            filename = path.Path(filename)
        if is_local(filename) and path.exists(filename):
            with filename.open("rb") as f:
                version = minc_version(f.read(8))
            if version == 2 and 2 in cls._versions():
                return _score(_sniff_minc2(str(filename)), error)
        return super().sniff_filename(filename, error=error, **kwargs)

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that a stream holds a MINC file that this
        class reads.

        A NetCDF file, gzipped or not, is certainly MINC1 when its header
        names a spatial dimension and an `image` variable, and weakly
        accepted otherwise. An HDF5 file is MINC2 when it has a
        `/minc-2.0` group.

        Parameters
        ----------
        file : IO
            The stream, which may be gzipped.
        error : bool or type[Exception], default=False
            Whether to raise an error when the stream is not a MINC file
            that this class reads.
        **kwargs : Any
            Options of the reader, which are ignored.

        Returns
        -------
        float
            The confidence, between 0 and 1.
        """
        return _score(sniff_minc(file, cls._versions()), error)

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that bytes hold a MINC file.

        Parameters
        ----------
        content : bytes
            The content of a file, or its first bytes.
        error : bool or type[Exception], default=False
            Whether to raise an error when the bytes are not a MINC file
            that this class reads.
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
        cls, filename: path.FilenameLike, **kwargs
    ) -> "MincImage":
        """Read an image from a MINC file, reading only its header.

        The lazy array reads the voxels from the file, by name, when they
        are needed. A file that is not local, or that is gzipped, is then
        read into memory. MINC has no reader option.

        Parameters
        ----------
        filename : path-like
            The path of the file.
        **kwargs : Any
            Options of the reader, which are refused.

        Returns
        -------
        MincImage
            The image, of the class of its MINC version, whose voxels are read
            lazily.

        Raises
        ------
        TypeError
            If an option is given.
        ParserExistsError
            If the file does not exist.
        ParserContentError
            If the file is not a MINC file of a version that this class
            reads, or nibabel cannot read its voxels.
        """
        _no_options(kwargs)
        raw, proxy = read_minc(filename)
        klass = cls._variant(raw.version)
        return klass(rawdata=proxy, metadata=MincMetadata.from_raw(raw))

    @classmethod
    def from_fileobj(cls, file: tx.BinaryIO, **kwargs) -> "MincImage":
        """Read an image from an open MINC file, gzipped or not.

        A stream has no name to open again, so its content is read into
        memory, where the lazy array decodes the voxels when they are
        needed. The position of the stream is restored. MINC has no reader
        option.

        Parameters
        ----------
        file : BinaryIO
            The stream, which may be gzipped.
        **kwargs : Any
            Options of the reader, which are refused.

        Returns
        -------
        MincImage
            The image, of the class of its MINC version, whose voxels are read
            lazily.

        Raises
        ------
        TypeError
            If an option is given.
        ParserContentError
            If the stream is not a MINC file of a version that this class
            reads, or nibabel cannot read its voxels.
        """
        _no_options(kwargs)
        raw, proxy = read_minc(file)
        klass = cls._variant(raw.version)
        return klass(rawdata=proxy, metadata=MincMetadata.from_raw(raw))


@register_format
class Minc1Image(MincImage):
    """An image read from a MINC1 (NetCDF classic) file.

    The file may be gzipped. Besides `"minc"`, this class answers to the
    hints `"minc1"` and `"minc.1"`. See [`MincImage`][].
    """

    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("minc1", "1")
    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".mnc", ".mnc.gz")
    VERSION: tx.ClassVar[int] = 1


class Minc2Image(MincImage):
    """An image read from a MINC2 (HDF5) file, which needs `h5py`.

    Besides `"minc"`, this class answers to the hints `"minc2"` and
    `"minc.2"`. See [`MincImage`][].
    """

    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("minc2", "2")
    VERSION: tx.ClassVar[int] = 2


MincImage.VARIANTS = (Minc1Image, Minc2Image)

# Without h5py, MINC2 is not registered, and a request for it by hint
# reports which package to install instead (see the package `__init__`).
if HAS_H5PY:
    register_format(Minc2Image)


# ----------------------------------------------------------------------
#   READING
# ----------------------------------------------------------------------


def _unit(dimension: MincDimension, axis_type: tx.Optional[str]) -> tx.Any:
    """Return the unit of the step of a dimension.

    The unit is the `units` of the dimension if it is a physical unit of
    the axis type, and otherwise millimeters for a spatial dimension and
    `None` for any other.
    """
    kind = _UNIT_TYPES.get(axis_type)
    if kind is None:
        return None
    unit = None
    if dimension.units:
        try:
            unit = Unit(dimension.units)
        except ValueError:
            unit = None
        if unit is not None and (
            not isinstance(unit, kind) or not is_physicalunit(unit)
        ):
            unit = None
    if unit is None and axis_type == "space":
        unit = "mm"
    return unit


def _minc_system(record: MincRaw) -> CoordinateSystem:
    """Return the voxel system of a record, in Fortran order."""
    axes = [_axis(d.name) for d in reversed(record.dimensions)]
    return CoordinateSystem(name="voxel", axes=axes, order="F")


def _record_transformations(
    metadata: tx.Optional[MincMetadata],
) -> tx.List[Transformation]:
    """Decode the transformations of the record of some metadata.

    Metadata without a record gives no transformation.
    """
    if metadata is None or metadata.raw is None:
        return []
    return _minc_to_transformations(metadata.raw)


def _minc_to_transformations(record: MincRaw) -> tx.List[Transformation]:
    """Return the voxel-to-physical and voxel-to-world transformations."""
    voxel_space = _minc_system(record)
    axes = list(voxel_space.axes)
    dimensions = list(reversed(record.dimensions))

    phys_axes = [
        replace(axis, unit=_unit(dim, axis.type))
        for axis, dim in zip(axes, dimensions)
    ]
    phys_space = CoordinateSystem(name=_PHYSICAL, axes=phys_axes)
    scale = [abs(dim.spacing) for dim in dimensions]
    xforms = [Scaling(input=voxel_space, output=phys_space, scale=scale)]

    vox2world = record.vox2world
    if vox2world is None:
        return xforms
    keep = [i for i, dim in enumerate(dimensions) if dim.is_spatial]
    unit = phys_axes[keep[0]].unit
    world_axes = [
        Axis(
            name,
            "space",
            unit=unit,
            orientation=Orientation(type="anatomical", value=value),
        )
        for name, value in _RAS_ORIENTATION.items()
    ]
    world_space = CoordinateSystem(name=_WORLD, axes=world_axes)
    # Non-spatial voxel axes, such as time, have zero columns in the
    # matrix, so they do not move a world point.
    matrix = np.zeros((3, len(axes) + 1))
    matrix[:, keep] = vox2world[:3, :3]
    matrix[:, -1] = vox2world[:3, 3]
    xforms.append(Affine(input=voxel_space, output=world_space, matrix=matrix))
    return xforms
