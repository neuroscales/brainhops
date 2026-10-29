"""The record of a MINC image: its header."""

# stdlib

# dependencies
import numpy as np
import typing_extensions as tx

# internals
from brainhops._core._internal_api import open_compressed, preserve_position
from brainhops.io.base._raw import Raw
from brainhops.io.base.parsers import (
    BinaryFileReader,
    Confidence,
    ParserContentError,
)

# this format
from ._constants import _MINC2_ROOT, _SNIFF_SIZE
from ._dimensions import (
    MincDimension,
    _string,
    minc1_dimensions,
    minc2_dimensions,
)
from ._netcdf import NetcdfHeader, read_netcdf_header
from ._utils import _score, _sniff_minc1, _sniff_minc2, minc_version

_UNSUPPORTED = (
    "Irregularly spaced dimensions, dimensions without a variable (such "
    "as vector_dimension) and volumes without image-min/image-max are not "
    "supported."
)
"""What nibabel, which reads the voxels, cannot read."""

_SIGNED = {"signed__": True, "unsigned": False}


class MincRaw(Raw, BinaryFileReader):
    """The header of a MINC image, as brainhops reads it.

    A MINC file is a NetCDF classic file (MINC1) or an HDF5 file (MINC2)
    whose voxels are a variable, `image`, described by one variable per
    dimension. The record holds the version of the file, the dimensions
    of the image slowest first, as the file lists them, and the type of
    the stored voxels. Other attributes of the file are not read, since
    MINC files cannot be written.

    The record never reads the voxels, nor the `image-min` and
    `image-max` values that scale them, so a file cut after its header
    can be read. It is held by
    [`MincMetadata`][brainhops.io.common.minc.MincMetadata]. A record is
    frozen, and two records are equal when their fields are.
    """

    version: int
    """The MINC version of the file, 1 or 2."""

    dimensions: tx.Tuple[MincDimension, ...] = ()
    """The dimensions of the image, slowest first, as the file lists them."""

    datatype: str = "float64"
    """The NumPy type of the stored voxels, as a type string."""

    @property
    def shape(self) -> tx.Tuple[int, ...]:
        """The shape of the stored voxels, in file (C) order."""
        return tuple(d.length for d in self.dimensions)

    @property
    def dtype(self) -> np.dtype:
        """The NumPy type of the stored voxels."""
        return np.dtype(self.datatype)

    @property
    def scaled_dtype(self) -> np.dtype:
        """The type of the voxels once nibabel scales them.

        Floating-point voxels are not scaled and keep their type, in the
        native byte order, and integers are scaled to 64-bit floats.
        """
        if self.dtype.kind == "f":
            return self.dtype.newbyteorder("=")
        return np.dtype(np.float64)

    @property
    def vox2world(self) -> tx.Optional[np.ndarray]:
        """The `(4, 4)` voxel-to-world (RAS) matrix of the Fortran-ordered
        array.

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

    @classmethod
    def from_fileobj(cls, file: tx.BinaryIO, **kwargs) -> "MincRaw":
        """Read the header of a MINC stream, gzipped or not.

        The voxels are not read, and the position of the stream is
        restored. A MINC2 stream needs `h5py`. The record has no options.

        Parameters
        ----------
        file : file object
            The stream, open in binary mode.
        **kwargs : Any
            Refused, since the record has no option.

        Returns
        -------
        MincRaw
            The record.

        Raises
        ------
        TypeError
            If an option is given.
        ParserContentError
            If the stream is not a MINC file, or a file that nibabel
            cannot read the voxels of.
        """
        _no_options(kwargs)
        with preserve_position(file):
            start = file.tell()
            version = minc_version(open_compressed(file).read(8))
            file.seek(start)
            if version == 1:
                return _read_minc1(open_compressed(file))
            if version == 2:
                return _read_minc2(file)
        raise ParserContentError("Not a MINC file (neither NetCDF nor HDF5).")

    @classmethod
    def sniff_fileobj(
        cls,
        file: tx.IO,
        error: tx.Union[bool, tx.Type[Exception]] = False,
        **kwargs,
    ) -> float:
        """Return the confidence that a stream holds a MINC file.

        A NetCDF file, gzipped or not, is certainly MINC1 when its header
        names a spatial dimension and an `image` variable, and weakly
        accepted otherwise. An HDF5 file is MINC2 when it has a
        `/minc-2.0` group. The position of the stream is restored.

        Parameters
        ----------
        file : file object
            The stream to test, open in binary mode.
        error : bool or type[Exception], default=False
            Whether to raise an error when the content is not a MINC file.
            With `True`, the error is a `SnifferContentError`, and an
            exception class is raised instead when one is given.
        **kwargs : Any
            Ignored.

        Returns
        -------
        float
            The confidence that the stream holds a MINC file.

        Raises
        ------
        SnifferContentError
            If the stream is not a MINC file and `error` is `True`.
        """
        return _score(sniff_minc(file, (1, 2)), error)


def sniff_minc(file: tx.IO, versions: tx.Collection[int]) -> float:
    """Score a stream as a MINC file of one of the given versions.

    The position of the stream is restored, and a stream that cannot be
    read scores `Confidence.NO`.

    Parameters
    ----------
    file : file object
        The stream to test, open in binary mode.
    versions : collection of int
        The MINC versions to accept, among 1 and 2.

    Returns
    -------
    float
        The confidence that the stream holds a MINC file of one of these
        versions.
    """
    try:
        with preserve_position(file):
            start = file.tell()
            head = open_compressed(file).read(_SNIFF_SIZE)
            version = minc_version(head)
            if version not in versions:
                return Confidence.NO
            if version == 1:
                return _sniff_minc1(head)
            file.seek(start)
            return _sniff_minc2(file)
    except Exception:  # noqa: BLE001
        return Confidence.NO


def _no_options(kwargs: tx.Mapping[str, tx.Any]) -> None:
    """Refuse the options of a reader that takes none.

    Raises
    ------
    TypeError
        If an option is given.
    """
    if kwargs:
        raise TypeError(f"Unknown MINC reader option(s): {', '.join(kwargs)}")


def _unsupported(version: int, reason: str) -> ParserContentError:
    """Return the error for a file whose voxels nibabel cannot read."""
    return ParserContentError(
        f"Cannot read this MINC{version} file with nibabel: {reason}. "
        f"{_UNSUPPORTED}"
    )


def _read_minc1(file: tx.BinaryIO) -> MincRaw:
    """Read the record of a MINC1 file from its NetCDF header.

    The file is checked as nibabel checks it before it reads the voxels.

    Raises
    ------
    ParserContentError
        If the header is not one that nibabel can read the voxels of.
    """
    header = read_netcdf_header(file)
    image = header.variables.get("image")
    if image is None:
        raise _unsupported(1, "the file has no image variable")
    for name in image.dimensions:
        variable = header.variables.get(name)
        if variable is None:
            raise _unsupported(1, f"the dimension {name} has no variable")
        spacing = _string(variable.attributes.get("spacing"))
        if spacing != "regular__":
            raise _unsupported(1, f"the dimension {name} is not regular")
    for name in ("image-min", "image-max"):
        if name not in header.variables:
            raise _unsupported(1, f"the file has no {name} variable")
    return MincRaw(
        version=1,
        dimensions=minc1_dimensions(header, image.dimensions),
        datatype=_minc1_datatype(header, image.dtype).str,
    )


def _minc1_datatype(header: NetcdfHeader, stored: np.dtype) -> np.dtype:
    """Return the type of the voxels of a MINC1 image.

    Integers are signed or unsigned as the `signtype` of the image says,
    and nibabel refuses integers without a `signtype`, so they are
    refused here as well.

    Raises
    ------
    ParserContentError
        If the image holds integers without a valid `signtype`.
    """
    if stored.kind != "i":
        return stored
    attributes = header.variables["image"].attributes
    signtype = _string(attributes.get("signtype"))
    if signtype not in _SIGNED:
        raise _unsupported(1, "the integer image has no valid signtype")
    if _SIGNED[signtype]:
        return stored
    return np.dtype(f">u{stored.itemsize}")


def _read_minc2(file: tx.BinaryIO) -> MincRaw:
    """Read the record of a MINC2 file with h5py, without its voxels.

    Raises
    ------
    ParserContentError
        If h5py is missing, the file is not MINC2, or nibabel cannot read
        its voxels.
    """
    try:
        import h5py
    except ImportError as e:
        raise ParserContentError("Reading MINC2 files needs h5py.") from e
    try:
        container = h5py.File(file, "r")
    except Exception as e:
        raise ParserContentError(f"Not a MINC2 file: {e}") from e
    with container:
        try:
            root = container[_MINC2_ROOT]
            group = root["image"]["0"]
            image = group["image"]
            dimensions = root["dimensions"]
        except KeyError as e:
            raise _unsupported(2, f"the file has no {e.args[0]}") from None
        dimorder = _string(image.attrs.get("dimorder")) or ""
        names = dimorder.split(",")[: len(image.shape)] if dimorder else []
        if len(names) != len(image.shape):
            raise _unsupported(2, "the image has no dimorder")
        for name in names:
            if name not in dimensions:
                raise _unsupported(2, f"the dimension {name} has no variable")
            spacing = _string(dimensions[name].attrs.get("spacing"))
            if spacing == "irregular":
                raise _unsupported(2, f"the dimension {name} is irregular")
        for name in ("image-min", "image-max"):
            if name not in group:
                raise _unsupported(2, f"the file has no {name} dataset")
        return MincRaw(
            version=2,
            dimensions=minc2_dimensions(image, dimensions, names),
            datatype=image.dtype.str,
        )
