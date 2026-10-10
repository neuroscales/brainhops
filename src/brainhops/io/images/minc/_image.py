"""The MINC image formats, which are read only."""

# stdlib
from io import BytesIO

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import KwOnly, NoRepr, replace

# internals
from brainhops._core import path
from brainhops._core.dependencies import HAS_H5PY
from brainhops._core.properties import smartproperty
from brainhops._core.typing import ArrayProtocol
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.images import SingleScaleImage
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
from brainhops.io.base.parsers import BinaryFileReader, ParserContentError
from brainhops.io.common.minc import MincDimension, MincMetadata, MincRaw
from brainhops.io.common.minc._data import read_minc
from brainhops.io.common.minc._raw import sniff_minc
from brainhops.io.common.minc._utils import (
    _axis,
    _score,
    _sniff_minc2,
    minc_version,
)
from brainhops.io.common.minc._views import _image_to_disk, _image_to_model
from brainhops.io.common.nifti._files import _is_local
from brainhops.io.images.base import ImageFormat

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
    [`MincRaw`][], and the voxels in `raw`, a lazy array that reads them
    with nibabel, scaled to real values, only when `data` is first
    needed. The lazy array presents the voxels in the order of the file,
    and the data reverse it into Fortran order, so a file with
    dimensions `zspace, yspace, xspace` gives data indexed `(x, y, z)`.

    The transformations are decoded from the record, unless other
    transformations are assigned. They are, in order, a [`Scaling`][] to
    `"physical"` by the absolute step of each dimension, in its unit
    (millimeters by default for spatial dimensions), and the preferred
    [`Affine`][] to the RAS space `"world"`, which moves along the
    direction cosines of each spatial dimension by its signed step from
    `start`. The affine is recorded only for volumes with three spatial
    dimensions.

    MINC files cannot be written, so the class has no writer and no
    `to_raw`: an image is saved to another format, such as NIfTI, and
    [`brainhops.io.save`][] refuses to write it as MINC. The record is
    `metadata.raw`.
    """

    HINTS: tx.ClassVar[tx.Tuple[str, ...]] = ("minc",)
    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".mnc",)

    VERSION: tx.ClassVar[tx.Optional[int]] = None
    """The MINC version this class reads, or `None` for both."""

    VARIANTS: tx.ClassVar[tx.Tuple[type, ...]] = ()
    """The classes to which a versionless class hands files."""

    raw: KwOnly[NoRepr[tx.Optional[ArrayProtocol]]] = None
    """The voxels in the order of the file, or `None` without data.

    After a read, `raw` is a lazy array, which reads the voxels and
    scales them when it is read. Setting `data` stores the new array
    here, with its axes reversed.
    """

    _metadata: KwOnly[NoRepr[tx.Optional[MincMetadata]]] = None

    metadata = smartproperty("metadata", invalidates=("transformations",))
    """The metadata of the file, which holds its record, or `None`.

    An image built from data has no metadata. Assigning other metadata
    drops the transformations decoded from the previous record.
    """

    def __post_init__(self, arguments: tx.Any) -> None:
        # The constructor assigns the fields in order, so the default of
        # `raw`, assigned after `data`, erases the array stored by `data=`.
        # The array is therefore stored again. When both are given, `data`
        # takes precedence over `raw`.
        if arguments.get("data") is not None:
            self.data = arguments["data"]

    @smartproperty(cache=True, invalidates=("data",))
    def data(self) -> tx.Optional[ArrayProtocol]:
        """The voxels in Fortran order, decoded from `raw` and cached.

        Data decoded from a file is read-only. To change the voxels,
        assign `data`: setting it stores the new array in `raw` and drops
        the cached value.
        """
        if self.raw is None:
            return None
        return _image_to_model(self.raw)

    @data.setter
    def data(self, value: tx.Optional[ArrayProtocol]) -> None:
        self.raw = None if value is None else _image_to_disk(value)

    @smartproperty(cache=True, unset=(None, "empty"))
    def transformations(self) -> tx.List[Transformation]:
        """Voxel-to-world transformations, decoded from the record.

        The decoded list is cached, so `img.transformation is
        img.transformation`. Assigning a list replaces the decoded one, and
        assigning other metadata drops it. An image without metadata has
        no transformations.
        """
        if self.metadata is None or self.metadata.raw is None:
            return []
        return _minc_to_transformations(self.metadata.raw)

    @property
    def shape(self) -> tx.Tuple[int, ...]:
        """The shape of the data, read from `raw` without reading the voxels.

        The data reverse the axes of `raw`, and the lazy array knows its
        shape from the record. An image without data raises as
        [`SingleScaleImage`][] does.
        """
        if self.raw is None:
            return self.data.shape
        return tuple(int(d) for d in reversed(self.raw.shape))

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
        """
        if isinstance(filename, str):
            filename = path.Path(filename)
        if _is_local(filename) and path.exists(filename):
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
        """
        return _score(sniff_minc(file, cls._versions()), error)

    @classmethod
    def sniff_bytes(
        cls,
        content: bytes,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that bytes hold a MINC file."""
        return cls.sniff_fileobj(BytesIO(bytes(content)), error=error)

    @classmethod
    def from_filename(
        cls, filename: path.FilenameLike, **kwargs
    ) -> "MincImage":
        """Read an image from a MINC file, reading only its header.

        The lazy array reads the voxels from the file, by name, when they
        are needed. A file that is not local, or that is gzipped, is then
        read into memory. The keyword arguments are ignored.

        Raises
        ------
        ParserExistsError
            If the file does not exist.
        ParserContentError
            If the file is not a MINC file of a version that this class
            reads, or nibabel cannot read its voxels.
        """
        raw, proxy = read_minc(filename)
        klass = cls._variant(raw.version)
        return klass(raw=proxy, metadata=MincMetadata.from_raw(raw))

    @classmethod
    def from_fileobj(cls, file: tx.BinaryIO, **kwargs) -> "MincImage":
        """Read an image from an open MINC file, gzipped or not.

        A stream has no name to open again, so its content is read into
        memory, where the lazy array decodes the voxels when they are
        needed. The position of the stream is restored, and the keyword
        arguments are ignored.

        Raises
        ------
        ParserContentError
            If the stream is not a MINC file of a version that this class
            reads, or nibabel cannot read its voxels.
        """
        raw, proxy = read_minc(file)
        klass = cls._variant(raw.version)
        return klass(raw=proxy, metadata=MincMetadata.from_raw(raw))


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
