"""The MINC parser."""

# stdlib
from io import BytesIO

# dependencies
import numpy as np
import typing_extensions as tx

# internals
from brainhops._core import path
from brainhops._core.streams import open_compressed, preserve_position
from brainhops._core.typing import ArrayProtocol
from brainhops.backends import get_array_backend
from brainhops.datamodel.base import DataModelBase
from brainhops.datamodel.systems import CoordinateSystem
from brainhops.io.base.parsers import (
    BinaryFileReader,
    Confidence,
    ParserContentError,
    ParserExistsError,
)
from brainhops.io.common.nifti._files import _is_local

# this format
from ._constants import _SNIFF_SIZE
from ._dimensions import (
    MincDimension,
    _read_dimensions,
)
from ._utils import (
    _axis,
    _read_minc,
    _read_voxels,
    _score,
    _sniff_minc1,
    _sniff_minc2,
    _Source,
    minc_version,
)


class MincParser(DataModelBase, BinaryFileReader):
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
