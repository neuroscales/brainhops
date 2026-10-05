"""
The shared MINC-reading machinery.

MINC is the volume format of the Montreal Neurological Institute (MNI)
and of the MINC toolkit. It comes in two containers, both with the
extension `.mnc`:

- **MINC1** is a NetCDF classic file (magic `CDF\\x01`, or `CDF\\x02`
  for the 64-bit offset variant). The voxels are the variable `image`,
  whose dimensions name the axes, and each spatial dimension is
  described by a variable of the same name.
- **MINC2** is an HDF5 file with a root group `/minc-2.0`. The voxels are
  the dataset `/minc-2.0/image/0/image`, whose attribute `dimorder`
  names the axes, and each dimension is described by a dataset under
  `/minc-2.0/dimensions`.

In both, the dimensions are listed slowest first (C order), in any order:
`zspace, yspace, xspace` is common (transverse slices), but sagittal or
coronal orders are just as valid. A dimension is described by its
`start`, its `step` (which may be negative), its `direction_cosines`
and its `units`. The *world* coordinates of a voxel are

```
world = sum_d (start_d + index_d * step_d) * direction_cosines_d
```

over the spatial dimensions `xspace`, `yspace` and `zspace`, in a world
space whose axes run left-to-right, posterior-to-anterior and
inferior-to-superior (RAS), in millimetres by default. A dimension
without direction cosines runs along its own world axis.

Integer voxels are scaled to real values through `image-min` and
`image-max`, which map the valid range of the type to real values,
possibly with one scaling per slice.

The headers and the voxels are read with `nibabel` (`nibabel.minc1`,
`nibabel.minc2`); MINC2 needs `h5py`, imported only when a MINC2 file
is read. `nibabel` cannot write MINC, so neither can `brainhops`.

!!! warning "Limitations, inherited from `nibabel`"
    Dimensions with irregular spacing (`spacing = "irregular"`) and
    dimensions that have no describing variable (such as MINC's
    `vector_dimension`, used for RGB volumes and displacement grids)
    are not supported, nor is a volume without `image-min` and
    `image-max` (which the MINC library always writes).
"""

__all__ = ["MincParser", "MincDimension", "minc_version"]

# stdlib
from io import BytesIO

# dependencies
import numpy as np
import typing_extensions as tx
from bagof.magic import Magic
from nibabel import minc1 as _minc1
from nibabel import minc2 as _minc2
from nibabel.externals.netcdf import netcdf_file as _netcdf_file

# internals
from brainhops._core import path
from brainhops._core.streams import open_compressed, preserve_position
from brainhops._core.typing import ArrayProtocol
from brainhops.backends import get_array_backend
from brainhops.datamodel.axes import Axis
from brainhops.datamodel.base import DataModelBase
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.io.base.nifti import _is_local
from brainhops.io.base.parsers import (
    BinaryFileParser,
    Confidence,
    ParserContentError,
    ParserExistsError,
    SnifferContentError,
)

_NETCDF_MAGICS = (b"CDF\x01", b"CDF\x02")
"""The magic numbers of NetCDF classic files (MINC1)."""

_HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
"""The signature at the start of an HDF5 file (MINC2)."""

_MINC2_ROOT = "minc-2.0"
"""The root group of a MINC2 file."""

_SNIFF_SIZE = 1 << 16
"""How many leading bytes of a MINC1 file are searched for its
dimension names (the NetCDF header lists them first)."""

SPATIAL_DIMENSIONS = ("xspace", "yspace", "zspace")
"""The names of the spatial MINC dimensions, in world-axis order."""

_DEFAULT_COSINES = {
    "xspace": (1.0, 0.0, 0.0),
    "yspace": (0.0, 1.0, 0.0),
    "zspace": (0.0, 0.0, 1.0),
}

# The axes that the usual MINC dimensions stand for. Any other dimension
# keeps its own name and has no type.
_AXES = {
    "xspace": ("x", "space"),
    "yspace": ("y", "space"),
    "zspace": ("z", "space"),
    "time": ("t", "time"),
}

_Source = tx.Union[str, bytes]
"""Where the voxels are read from: a local path, or the file's bytes."""


class MincDimension(Magic, frozen=True):
    """
    One dimension of a MINC volume, as its header describes it.

    The attributes the file does not record are `None`; the properties
    give MINC's defaults instead.
    """

    name: str
    """The MINC name of the dimension (`"xspace"`, `"time"`, ...)."""

    length: int
    """The number of samples along the dimension."""

    start: tx.Optional[float] = None
    """The world coordinate of the first sample, along the direction
    cosines (MINC's default is 0)."""

    step: tx.Optional[float] = None
    """The distance between two samples, possibly negative (MINC's
    default is 1)."""

    direction_cosines: tx.Optional[tx.Tuple[float, ...]] = None
    """The world direction of the dimension, for a spatial one (MINC's
    default is its own world axis)."""

    units: tx.Optional[str] = None
    """The unit of `start` and `step`, as written in the file."""

    @property
    def is_spatial(self) -> bool:
        """Whether the dimension is one of `xspace`, `yspace`, `zspace`."""
        return self.name in SPATIAL_DIMENSIONS

    @property
    def origin(self) -> float:
        """`start`, or 0 when the file does not record it."""
        return 0.0 if self.start is None else float(self.start)

    @property
    def spacing(self) -> float:
        """`step`, or 1 when the file does not record it."""
        return 1.0 if self.step is None else float(self.step)

    @property
    def cosines(self) -> tx.Tuple[float, float, float]:
        """`direction_cosines`, or the dimension's own world axis."""
        if self.direction_cosines is not None:
            return tuple(float(c) for c in self.direction_cosines)
        return _DEFAULT_COSINES.get(self.name, (0.0, 0.0, 0.0))


def minc_version(head: bytes) -> tx.Optional[int]:
    """
    The MINC version that the leading bytes of a file suggest: 1 for a
    NetCDF classic file, 2 for an HDF5 file, `None` otherwise.

    An HDF5 file is only a MINC2 candidate: it must still hold a
    `/minc-2.0` group.
    """
    head = bytes(head[:8])
    if head[:4] in _NETCDF_MAGICS:
        return 1
    if head == _HDF5_SIGNATURE:
        return 2
    return None


class MincParser(DataModelBase, BinaryFileParser):
    """
    Base class for objects that are encoded by a MINC1 or MINC2 file.

    It holds the dimensions of the volume ([`dimensions`][]), in the
    order the file stores them (slowest first), and reads the voxels on
    first access, scaled to real values, from the file it was loaded
    from. The voxel coordinate system and the voxel-to-world matrix are
    those of the array in F order, i.e. with the axes reversed, so that
    the fastest-varying dimension comes first.

    A concrete format sets `VERSION` (1 or 2) and is only recognised in
    files of that version. A class with no version reads both, and
    returns an object of the matching class among its `VARIANTS`.
    """

    HINTS = ("minc",)
    EXTENSIONS: tx.ClassVar[tx.Tuple[str, ...]] = (".mnc",)

    VERSION: tx.ClassVar[tx.Optional[int]] = None
    """The MINC version read by this class, or `None` for both."""

    VARIANTS: tx.ClassVar[tx.Tuple[type, ...]] = ()
    """The classes that a class with no `VERSION` hands files to."""

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

    # --- geometry -----------------------------------------------------

    @property
    def version(self) -> tx.Optional[int]:
        """The MINC version of the file (1 or 2)."""
        return self.VERSION

    @property
    def shape(self) -> tx.Optional[tx.Tuple[int, ...]]:
        """The shape of the volume in F order (fastest dimension
        first)."""
        if self.dimensions:
            return tuple(d.length for d in reversed(self.dimensions))
        data = getattr(self, "_data", None)
        return None if data is None else tuple(data.shape)

    @property
    def vox2world(self) -> tx.Optional[np.ndarray]:
        """
        The `(4, 4)` voxel-to-world (RAS) matrix of the F-ordered array.

        Its columns follow the spatial axes of [`system`][] (the
        non-spatial ones, such as time, are skipped). It is `None` unless
        the volume has the three spatial dimensions.
        """
        spatial = [d for d in reversed(self.dimensions) if d.is_spatial]
        if len(spatial) != 3:
            return None
        cosines = np.asarray([d.cosines for d in spatial]).T
        matrix = np.eye(4)
        matrix[:3, :3] = cosines * [d.spacing for d in spatial]
        matrix[:3, 3] = cosines @ [d.origin for d in spatial]
        return matrix

    # --- datamodel ----------------------------------------------------

    @property
    def data(self) -> tx.Optional[ArrayProtocol]:
        """The voxels in F order, scaled to real values, read on first
        access and cached, unless set explicitly."""
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
        """The voxel coordinate system, in F order, derived from the
        dimensions unless set explicitly."""
        if getattr(self, "_system", None) is not None:
            return self._system
        if not self.dimensions:
            return None
        axes = [_axis(d.name) for d in reversed(self.dimensions)]
        return CoordinateSystem(name="voxel", axes=axes, order="F")

    @system.setter
    def system(self, value: tx.Optional[CoordinateSystem]) -> None:
        self._system = value

    # --- variants -----------------------------------------------------

    @classmethod
    def _variant(cls, version: tx.Optional[int]) -> type:
        """The class that reads a file of this version."""
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

    # --- BinaryFileSniffer API ----------------------------------------

    @classmethod
    def sniff_filename(
        cls,
        filename: path.FilenameLike,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Score a file by its content (a MINC2 file is opened by name,
        so that only its header is read)."""
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
        Score how confident the class is that an open file object holds
        a MINC file of its version, from its content.

        A MINC1 file is a NetCDF file (gzipped or not) whose header names
        a MINC spatial dimension and an `image` variable; a NetCDF file
        that does not is only weakly accepted. A MINC2 file is an HDF5
        file with a `/minc-2.0` group.
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
        """Score how confident the class is that bytes hold a MINC file
        of its version."""
        return cls.sniff_fileobj(BytesIO(bytes(content)), error=error)

    @classmethod
    def _reads(cls, version: tx.Optional[int]) -> bool:
        """Whether this class (or one of its variants) reads a version."""
        if version is None:
            return False
        if cls.VERSION is not None:
            return cls.VERSION == version
        return any(v.VERSION == version for v in cls.VARIANTS)

    # --- BinaryFileParser API -----------------------------------------

    @classmethod
    def from_filename(cls, filename: path.FilenameLike, **kwargs) -> tx.Self:
        """
        Build the object from a MINC file.

        Only the header is read: the voxels are read from the file, by
        name, on first access. A file that is not local, or that is
        gzipped, is read into memory first.
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
        """Build the object from an open MINC file object (gzipped or
        not), which is read into memory."""
        with preserve_position(file):
            content = open_compressed(file).read()
        return cls._from_source(bytes(content), **kwargs)

    @classmethod
    def from_bytes(cls, content: bytes, **kwargs) -> tx.Self:
        """Build the object from the bytes of a MINC file (gzipped or
        not)."""
        return cls.from_fileobj(BytesIO(bytes(content)), **kwargs)

    @classmethod
    def _from_source(cls, source: _Source, **kwargs) -> tx.Self:
        """Read the header of a MINC file, and keep where it came from to
        read the voxels later."""
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
    """The voxel axis that a MINC dimension stands for."""
    axis_name, axis_type = _AXES.get(name, (name, None))
    return Axis(axis_name, axis_type, unit="index")


def _score(score: float, error: tx.Union[bool, tx.Type[Exception]]) -> float:
    """Return a score, or raise when it is zero and asked to."""
    if score or not error:
        return score
    if error is True:
        error = SnifferContentError
    raise error("Content is not a MINC file of this version.")


def _sniff_minc1(head: bytes) -> float:
    """Score the leading bytes of a NetCDF file as a MINC1 file."""
    named = any(name.encode() in head for name in SPATIAL_DIMENSIONS)
    if named and b"image" in head:
        return Confidence.CERTAIN
    return Confidence.WEAK


def _sniff_minc2(file: tx.Union[str, tx.IO]) -> float:
    """Score an HDF5 file (a path or a stream) as a MINC2 file."""
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
    Open a MINC file with `nibabel`, read it, and close it.

    `read(container, mfile, version)` is handed the container (the
    NetCDF or HDF5 file) and the `nibabel` MINC file that reads the
    voxels from it. What it returns must not refer to the file's
    content: a MINC1 file on disk is memory-mapped (so that its header is
    read without its voxels), and the map is closed on return.
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
            # Raised outside of this block: the traceback would otherwise
            # keep the (memory-mapped) variables alive past `close`.
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
    """The voxels of an open MINC file, scaled, in file (C) order."""
    array = mfile.get_scaled_data()
    # Native byte order (MINC1 is big-endian), in a copy that outlives the
    # (possibly memory-mapped) file.
    return np.array(array, dtype=array.dtype.newbyteorder("="))


def _read_dimensions(
    container: tx.Any, mfile: _minc1.Minc1File, version: int
) -> tx.Tuple[MincDimension, ...]:
    """The dimensions of the image of an open MINC file (its NetCDF or
    HDF5 container), slowest first."""
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
    """A scalar attribute as a float (NetCDF stores it as an array)."""
    if value is None:
        return None
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    return float(value[0]) if value.size else None


def _floats(value: tx.Any) -> tx.Optional[tx.Tuple[float, ...]]:
    """A vector attribute as a tuple of floats."""
    if value is None:
        return None
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    return tuple(float(v) for v in value) if value.size == 3 else None


def _string(value: tx.Any) -> tx.Optional[str]:
    """A string attribute as a `str`."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        value = value.reshape(-1)[0] if value.size else b""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bytes, bytearray)):
        value = bytes(value).decode("latin-1")
    return str(value).strip("\x00 ") or None
