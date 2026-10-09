"""
Reading of MINC files.

MINC, the volume format of the MINC toolkit, comes in two containers
that share the `.mnc` extension. A MINC1 file is a NetCDF classic file
whose `image` variable holds the voxels. A MINC2 file is an HDF5 file
whose voxels are the dataset `/minc-2.0/image/0/image`. In both, each
dimension is described by a variable of its own name, and the
dimensions are listed slowest first, in any order.

A dimension has a `start`, a `step`, which may be negative, and
`direction_cosines`. World coordinates, in RAS millimetres by default,
are

```
world = sum_d (start_d + index_d * step_d) * direction_cosines_d
```

over the spatial dimensions `xspace`, `yspace` and `zspace`. Integer
voxels are scaled to real values through `image-min` and `image-max`.
Files are read with nibabel (and h5py for MINC2), which cannot write
MINC.

!!! warning "Limitations, inherited from nibabel"
    Irregularly spaced dimensions are not supported, nor are dimensions
    without a describing variable (such as `vector_dimension`), nor
    volumes without `image-min` and `image-max`.
"""

__all__ = ["MincParser", "MincDimension", "minc_version"]

from io import BytesIO

import numpy as np
import typing_extensions as tx
from bagof.magic import Magic
from nibabel import minc1 as _minc1
from nibabel import minc2 as _minc2
from nibabel.externals.netcdf import netcdf_file as _netcdf_file

from brainhops._core import path
from brainhops._core.streams import open_compressed, preserve_position
from brainhops._core.typing import ArrayProtocol
from brainhops.backends import get_array_backend
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.base import DataModelBase
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.io.base.parsers import (
    BinaryFileParser,
    Confidence,
    ParserContentError,
    ParserExistsError,
    SnifferContentError,
)
from brainhops.io.common.nifti import _is_local

_NETCDF_MAGICS = (b"CDF\x01", b"CDF\x02")
"""The magic numbers of NetCDF classic files (MINC1)."""

_HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
"""The signature of HDF5 files (MINC2)."""

_MINC2_ROOT = "minc-2.0"
"""The root group of a MINC2 file."""

_SNIFF_SIZE = 1 << 16
"""
The number of leading bytes of a MINC1 file searched for dimension
names.
"""

SPATIAL_DIMENSIONS = ("xspace", "yspace", "zspace")
"""The spatial MINC dimensions, in world-axis order."""

_DEFAULT_COSINES = {
    "xspace": (1.0, 0.0, 0.0),
    "yspace": (0.0, 1.0, 0.0),
    "zspace": (0.0, 0.0, 1.0),
}

# Other dimensions keep their own name and have no axis type.
_AXES = {
    "xspace": ("x", "space"),
    "yspace": ("y", "space"),
    "zspace": ("z", "space"),
    "time": ("t", "time"),
}

_Source = tx.Union[str, bytes]
"""The source of the voxels: a local path, or the file bytes."""


class MincDimension(Magic, frozen=True):
    """
    One dimension of a MINC volume, as the header describes it.

    Unrecorded attributes are `None`; the properties give the defaults.
    """

    name: str
    """The MINC name of the dimension, such as `"xspace"` or `"time"`."""

    length: int
    """The number of samples."""

    start: tx.Optional[float] = None
    """The world coordinate of the first sample."""

    step: tx.Optional[float] = None
    """The distance between samples, which may be negative."""

    direction_cosines: tx.Optional[tx.Tuple[float, ...]] = None
    """The world direction of a spatial dimension."""

    units: tx.Optional[str] = None
    """The unit of `start` and `step`, as written."""

    @property
    def is_spatial(self) -> bool:
        """Whether the dimension is `xspace`, `yspace` or `zspace`."""
        return self.name in SPATIAL_DIMENSIONS

    @property
    def origin(self) -> float:
        """The `start`, or 0 if it is not recorded."""
        return 0.0 if self.start is None else float(self.start)

    @property
    def spacing(self) -> float:
        """The `step`, or 1 if it is not recorded."""
        return 1.0 if self.step is None else float(self.step)

    @property
    def cosines(self) -> tx.Tuple[float, float, float]:
        """The `direction_cosines`, or else the world axis of the dimension."""
        if self.direction_cosines is not None:
            return tuple(float(c) for c in self.direction_cosines)
        return _DEFAULT_COSINES.get(self.name, (0.0, 0.0, 0.0))


def minc_version(head: bytes) -> tx.Optional[int]:
    """
    Return the MINC version that the leading bytes suggest, or `None`.

    NetCDF content suggests version 1, and HDF5 content version 2, although
    a MINC2 file also needs the `/minc-2.0` group.
    """
    head = bytes(head[:8])
    if head[:4] in _NETCDF_MAGICS:
        return 1
    if head == _HDF5_SIGNATURE:
        return 2
    return None


class MincParser(DataModelBase, BinaryFileParser):
    """
    The base class of objects encoded as MINC1 or MINC2 files.

    The object holds the dimensions as the file lists them, and reads the
    voxels from the source file on first access. The coordinate system and
    the voxel-to-world matrix are those of the Fortran-ordered array.

    A concrete format sets [`VERSION`][] and reads only that version. A
    class without a version reads both, and returns an object of the
    matching class among its [`VARIANTS`][].
    """

    HINTS = ("minc",)
    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".mnc",)

    VERSION: tx.ClassVar[tx.Optional[int]] = None
    """The MINC version this class reads, or `None` for both."""

    VARIANTS: tx.ClassVar[tx.Tuple[type, ...]] = ()
    """The classes to which a versionless class hands files."""

    dimensions: tx.Annotated[
        tx.Tuple[MincDimension, ...],
        tx.Doc(
            """
            The dimensions of the volume, slowest first, as the file
            lists them.
            """
        ),
    ] = ()

    _source: tx.Annotated[
        tx.Optional[_Source],
        tx.Doc(
            """
            Where the voxels are read from: the path of a local file, or
            the (decompressed) bytes of the file.
            """
        ),
    ] = None

    @property
    def version(self) -> tx.Optional[int]:
        """The MINC version of the file, 1 or 2."""
        return self.VERSION

    @property
    def shape(self) -> tx.Optional[tx.Tuple[int, ...]]:
        """The shape of the volume, in Fortran order (fastest first)."""
        if self.dimensions:
            return tuple(d.length for d in reversed(self.dimensions))
        data = getattr(self, "_data", None)
        return None if data is None else tuple(data.shape)

    @property
    def vox2world(self) -> tx.Optional[np.ndarray]:
        """
        The `(4, 4)` voxel-to-world (RAS) matrix of the Fortran-ordered array.

        Non-spatial axes, such as time, are skipped. The matrix is `None`
        unless the volume has all three spatial dimensions.
        """
        spatial = [d for d in reversed(self.dimensions) if d.is_spatial]
        if len(spatial) != 3:
            return None
        cosines = np.asarray([d.cosines for d in spatial]).T
        matrix = np.eye(4)
        matrix[:3, :3] = cosines * [d.spacing for d in spatial]
        matrix[:3, 3] = cosines @ [d.origin for d in spatial]
        return matrix

    @property
    def data(self) -> tx.Optional[ArrayProtocol]:
        """
        The voxels in Fortran order, scaled to real values.

        They are read on first access and cached, unless set explicitly.
        """
        if getattr(self, "_data", None) is not None:
            return self._data
        if self._source is None or not self.dimensions:
            return None
        array = _read_minc(self._source, self.VERSION, _read_voxels)
        self._data = get_array_backend().asarray(array.T)
        return self._data

    @data.setter
    def data(self, value: tx.Optional[ArrayProtocol]) -> None:
        self._data = value

    @property
    def system(self) -> tx.Optional[CoordinateSystem]:
        """
        The voxel coordinate system in Fortran order, derived from the
        dimensions unless set explicitly.
        """
        if getattr(self, "_system", None) is not None:
            return self._system
        if not self.dimensions:
            return None
        axes = [_axis(d.name) for d in reversed(self.dimensions)]
        return CoordinateSystem(name="voxel", axes=axes, order="F")

    @system.setter
    def system(self, value: tx.Optional[CoordinateSystem]) -> None:
        self._system = value

    @classmethod
    def _variant(cls, version: tx.Optional[int]) -> type:
        """Return the class that reads a file of a given version."""
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
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """
        Score a file from its content.

        A local MINC2 file is opened by name, so that only its header is read.
        """
        if isinstance(filename, str):
            filename = path.Path(filename)
        if _is_local(filename) and path.exists(filename):
            with filename.open("rb") as f:
                version = minc_version(f.read(8))
            if version == 2 and cls._reads(2):
                return _score(_sniff_minc2(str(filename)), error)
        return super().sniff_filename(filename, error=error, **kwargs)

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """
        Return the confidence that an open file holds a MINC file of the
        version of this class.

        A NetCDF file, gzipped or not, is certainly MINC1 when its header names
        a spatial dimension and an `image` variable, and weakly accepted
        otherwise. An HDF5 file is MINC2 when it has a `/minc-2.0` group.
        """
        score = Confidence.NO
        try:
            with preserve_position(file):
                start = file.tell()
                stream = open_compressed(file)
                head = stream.read(_SNIFF_SIZE)
                version = minc_version(head)
                if not cls._reads(version):
                    version = None
                if version == 1:
                    score = _sniff_minc1(head)
                elif version == 2:
                    file.seek(start)
                    score = _sniff_minc2(file)
        except Exception:
            score = Confidence.NO
        return _score(score, error)

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
    def _reads(cls, version: tx.Optional[int]) -> bool:
        """Tell whether this class, or one of its variants, reads a version."""
        if version is None:
            return False
        if cls.VERSION is not None:
            return cls.VERSION == version
        return any(v.VERSION == version for v in cls.VARIANTS)

    @classmethod
    def from_filename(cls, filename: path.FilenameLike, **kwargs) -> tx.Self:
        """
        Build an object from a MINC file, reading only its header.

        The voxels are read from the file, by name, on first access. A file
        that is not local, or that is gzipped, is read into memory first.
        """
        if isinstance(filename, str):
            filename = path.Path(filename)
        if not path.exists(filename):
            raise ParserExistsError(f"No such file: {filename}")
        with filename.open("rb") as f:
            head = f.read(8)
        if _is_local(filename) and minc_version(head) is not None:
            return cls._from_source(str(filename), **kwargs)
        with filename.open("rb") as f:
            return cls.from_fileobj(f, **kwargs)

    @classmethod
    def from_fileobj(cls, file: tx.IO, **kwargs) -> tx.Self:
        """
        Build an object from an open MINC file, gzipped or not, by reading it
        into memory.
        """
        with preserve_position(file):
            content = open_compressed(file).read()
        return cls._from_source(bytes(content), **kwargs)

    @classmethod
    def from_bytes(cls, content: bytes, **kwargs) -> tx.Self:
        """Build an object from the bytes of a MINC file, gzipped or not."""
        return cls.from_fileobj(BytesIO(bytes(content)), **kwargs)

    @classmethod
    def _from_source(cls, source: _Source, **kwargs) -> tx.Self:
        """Read the header, and keep the source to read the voxels later."""
        if isinstance(source, str):
            with open(source, "rb") as f:
                head = f.read(8)
        else:
            head = source[:8]
        klass = cls._variant(minc_version(head))
        dimensions = _read_minc(source, klass.VERSION, _read_dimensions)
        return klass(dimensions=dimensions, source=source, **kwargs)


# ----------------------------------------------------------------------
#   HELPERS
# ----------------------------------------------------------------------


def _axis(name: str) -> Axis:
    """Return the voxel axis that a MINC dimension stands for."""
    axis_name, axis_type = _AXES.get(name, (name, None))
    return Axis(axis_name, axis_type, unit="index")


def _score(score: float, error: tx.Union[bool, tx.Type[Exception]]) -> float:
    """Return a score, or raise if it is zero and the caller asks to."""
    if score or not error:
        return score
    if error is True:
        error = SnifferContentError
    raise error("Content is not a MINC file of this version.")


def _sniff_minc1(head: bytes) -> float:
    """Score the leading bytes of a NetCDF file as MINC1."""
    named = any(name.encode() in head for name in SPATIAL_DIMENSIONS)
    if named and b"image" in head:
        return Confidence.CERTAIN
    return Confidence.WEAK


def _sniff_minc2(file: tx.Union[str, tx.IO]) -> float:
    """Score an HDF5 file, given by path or stream, as MINC2."""
    import h5py

    try:
        with h5py.File(file, "r") as f:
            found = isinstance(f.get(_MINC2_ROOT), h5py.Group)
    except Exception:
        return Confidence.NO
    return Confidence.CERTAIN if found else Confidence.NO


def _read_minc(
    source: _Source,
    version: int,
    read: tx.Callable[[tx.Any, _minc1.Minc1File, int], tx.Any],
) -> tx.Any:
    """
    Open a MINC file with nibabel, apply `read` to it, and close it.

    The result of `read` must not refer to the file content, because a
    MINC1 file on disk is memory-mapped and the map is closed on return.
    """
    try:
        if version == 1:
            if isinstance(source, str):
                container = _netcdf_file(source, "r", mmap=True)
            else:
                container = _netcdf_file(BytesIO(source), "r", mmap=False)
        else:
            import h5py

            fileish = source if isinstance(source, str) else BytesIO(source)
            container = h5py.File(fileish, "r")
    except Exception as e:
        raise ParserContentError(f"Not a MINC{version} file: {e}") from e
    mfile, failure = None, None
    try:
        try:
            mfile = (_minc1.Minc1File if version == 1 else _minc2.Minc2File)(
                container
            )
        except (
            AttributeError,
            KeyError,
            ValueError,
            _minc1.MincError,
        ) as e:
            # Raise outside this block: a held traceback would keep
            # memory-mapped variables alive past close.
            failure = repr(e)
        if failure is not None:
            raise ParserContentError(
                f"Cannot read this MINC{version} file with nibabel: "
                f"{failure}. Irregularly spaced dimensions, dimensions "
                f"without a variable (such as vector_dimension) and "
                f"volumes without image-min/image-max are not supported."
            )
        return read(container, mfile, version)
    finally:
        mfile = None
        container.close()


def _read_voxels(
    container: tx.Any, mfile: _minc1.Minc1File, version: int
) -> np.ndarray:
    """Read the scaled voxels of an open file, in file (C) order."""
    array = mfile.get_scaled_data()
    # Copy into native byte order (MINC1 is big-endian), so that the result
    # outlives a memory-mapped file.
    return np.array(array, dtype=array.dtype.newbyteorder("="))


def _read_dimensions(
    container: tx.Any, mfile: _minc1.Minc1File, version: int
) -> tx.Tuple[MincDimension, ...]:
    """Read the dimensions of the image of an open file, slowest first."""
    if version == 1:
        image = container.variables["image"]
        names = list(image.dimensions)
        describe = [container.variables[name] for name in names]

        def attr(var: tx.Any, key: str) -> tx.Any:
            return getattr(var, key, None)

    else:
        image = container[_MINC2_ROOT]["image"]["0"]["image"]
        dimorder = _string(image.attrs.get("dimorder")) or ""
        names = dimorder.split(",")[: len(image.shape)]
        group = container[_MINC2_ROOT]["dimensions"]
        describe = [group[name] for name in names]

        def attr(var: tx.Any, key: str) -> tx.Any:
            return var.attrs.get(key, None)

    shape = tuple(int(n) for n in image.shape)
    return tuple(
        MincDimension(
            name=name,
            length=length,
            start=_float(attr(var, "start")),
            step=_float(attr(var, "step")),
            direction_cosines=_floats(attr(var, "direction_cosines")),
            units=_string(attr(var, "units")),
        )
        for name, length, var in zip(names, shape, describe)
    )


def _float(value: tx.Any) -> tx.Optional[float]:
    """Read a scalar attribute, which NetCDF stores as an array."""
    if value is None:
        return None
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    return float(value[0]) if value.size else None


def _floats(value: tx.Any) -> tx.Optional[tx.Tuple[float, ...]]:
    """Read a vector attribute of three floats, or `None`."""
    if value is None:
        return None
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    return tuple(float(v) for v in value) if value.size == 3 else None


def _string(value: tx.Any) -> tx.Optional[str]:
    """Read a string attribute, or `None` if it is empty."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        value = value.reshape(-1)[0] if value.size else b""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bytes, bytearray)):
        value = bytes(value).decode("latin-1")
    return str(value).strip("\x00 ") or None
